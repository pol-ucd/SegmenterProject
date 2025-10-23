import argparse
import logging
import sys
import traceback
from abc import abstractmethod
from copy import deepcopy
from os import PathLike
from typing import Any, Dict, Union, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, autocast
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from segmenter.loss import MaskedCosineSimilarityLoss
from segmenter.utils import HDF5DatasetOptimized
from segmenter.utils.data import SSLTransformPipeline, HDF5BatchSampler, hdf5_worker_init_fn

""" DEBUG helpers - COmment out unless debugging """
import os

# Force synchronous CUDA errors to surface
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Enable autograd anomaly detection and print where NaNs appear
# torch.autograd.set_detect_anomaly(True)
""" End of debug stuff """

MASK_RATIO = 0.7
PATCH_SIZE = 4  # 4x4 patches for SegFormer MiT [1]
IMAGE_H = 512
IMAGE_W = 512
debug_run = True

# source = "/Users/polmacaonghusa/Documents/Projects/segmenter/data/Classica.h5"
# source = "/Users/polmacaonghusa/Documents/Projects/segmenter/data/pretrain_images.h5"
source = "../segmenter/data/pretrain_images.h5"

backbone_name = "nvidia/segformer-b4-finetuned-ade-512-512"

from segmenter.masks import CompositeMask


class PatchMaskingException(Exception):
    pass


class BackboneWrapperBase(nn.Module):
    """
    Base class for all backbone wrappers.
    """

    def __init__(self, backbone_name):
        super(BackboneWrapperBase, self).__init__()
        self.backbone_name = backbone_name
        self.model = None
        self.encoder = None
        self.decoder = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        pass

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def set_encoder(self, encoder):
        self.encoder = encoder

    def set_decoder(self, decoder):
        self.decoder = decoder


class SegFormerBackboneWrapper(BackboneWrapperBase):
    """
    Wrapper for SegFormer backbone model. Exposes the encodings produced by the backbone
    as a consistent forward() interface.
    """

    def __init__(self, backbone_name: Union[str, PathLike] = backbone_name):
        super().__init__(backbone_name=backbone_name)
        self.cfg = SegformerConfig.from_pretrained(backbone_name) if backbone_name else SegformerConfig()
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(
                backbone_name, config=self.cfg, ignore_mismatched_sizes=True
            )
            if backbone_name
            else SegformerForSemanticSegmentation(config=self.cfg)
        )

        """
                Can check for existence of base SegFormer model using self._get_components(self.model)
                and so on. But we know this is a SegFormerFor Semantic Segmentation model. So we 
                can just assume everything is as follow 
                """
        self.encoder = self.model.segformer.encoder
        self.decoder = self.model.decode_head

        """
        In general case can check what the names of each component is:

        encoder_components = self._get_cmponents(self.encoder)

        But we have a predefined model here.
        """
        # self.patch_embeddings = self.encoder.patch_embeddings  # List of 4 patch embedding modules
        # self.blocks = self.encoder.block  # List of 4 ModuleLists of blocks
        # self.norms = self.encoder.layer_norm  # List of 4 Norm modules

        # self.processor = SegformerImageProcessor()
        # Determine the patch size/stride for the first stage (MiT-style structure)
        # self.strides = self.encoder.config.patch_sizes
        # self.initial_stride = self.strides[0]
        # self.hidden_dims = self.encoder.config.hidden_sizes
        # self.initial_hidden_dim = self.hidden_dims[0]

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        encodings = self.model(x, output_hidden_states=True, return_dict=True)['hidden_states']
        upscaled_encodings = [F.interpolate(enc,
                                            size=x.shape[-2:],
                                            mode="bilinear",
                                            align_corners=False) for enc in encodings]
        return upscaled_encodings


    #
    # def mask2patches(self, mask: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    #     """
    #     Calculate Patch Embeddings from the backbone model patch embeddings layer.
    #     Runs the first patch embedding module only.
    #     :param mask: torch.Tensor 0-1 mask
    #     :return: Tuple[torch.Tensor(patch_embedding(B, HxW, N_features),
    #                         int(Height - e.g. 128), int(Width - e.g. 128)]
    #     """
    #     return self.initial_patch_layer(mask)
    #
    # # --------------------------------------------------------------------------
    # # 4) Function to Upscale SegFormer Embeddings
    # # --------------------------------------------------------------------------
    # @staticmethod
    # def upscale_embeddings(stage_embeddings: Union[List[torch.Tensor], Tuple[torch.Tensor]]):
    #     """Upscales all encoder outputs to the spatial resolution of the initial patches (L1)."""
    #
    #     _, _, h_upscale, w_upscale = stage_embeddings[0].shape
    #
    #     upscaled_outputs = []
    #     for features in stage_embeddings:
    #         upscaled_spatial = F.interpolate(
    #             features,
    #             size=(h_upscale, w_upscale),
    #             mode='bilinear',
    #             align_corners=False
    #         )
    #
    #         upscaled_outputs.append(upscaled_spatial)
    #     return tuple(upscaled_outputs)
    #
    # @staticmethod
    # def _get_components(item: torch.nn.Module) -> List[str]:
    #     keys = set()
    #     for key, value in item.named_parameters():
    #         key_0 = key.split('.')[0]
    #         keys.add(key_0)
    #     return list(keys)

    def __repr__(self):
        return f"{self.__class__.__name__}(backbone={self.backbone})"


class OldMoCoMSN(nn.Module):
    def __init__(self, backbone: SegFormerBackboneWrapper = SegFormerBackboneWrapper(backbone_name),
                 momentum: float = 0.99, mask_generator: CompositeMask = CompositeMask(), ):
        super().__init__()
        self.backbone_encoder = backbone
        self.momentum = float(momentum)
        self.mask_generator = mask_generator

        self.anchor_encoder = SegFormerBackboneWrapper(backbone)
        self.target_encoder = deepcopy(self.anchor_encoder)
        self._set_requires_grad(self.target_encoder, False)

    def forward(self, anchor: torch.Tensor, target: torch.Tensor) -> Tuple[Any, Any, Any]:

        with torch.no_grad():
            target_encodings = self.target_encoder(target)
            target_encodings_upscaled = self.target_encoder.upscale_embeddings(target_encodings)
        anchor_mask = self.mask_generator.generate_pixel_mask(anchor).to(anchor.device)

        visible_mask = anchor_mask.long()
        visible_anchor = anchor * (1 - visible_mask)
        anchor_encodings = self.anchor_encoder(visible_anchor)
        anchor_encodings_upscaled = self.anchor_encoder.upscale_embeddings(anchor_encodings)

        h_downscale, w_downscale = anchor_encodings[0].shape[-2:]

        mask_encodings_upscaled = F.interpolate(
            anchor_mask,
            size=(h_downscale, w_downscale),
            mode='bilinear',
            align_corners=False
        )

        # print(f"MoCoMSN.forward() Exiting CUDA memory usage : ", torch.cuda.memory_usage(device='cuda'))

        return anchor_encodings_upscaled, target_encodings_upscaled, mask_encodings_upscaled

    @staticmethod
    def _set_requires_grad(model: nn.Module, requires_grad: bool):
        for p in model.parameters():
            p.requires_grad = requires_grad

    @torch.no_grad()
    def update_momentum_encoder(self):
        for q, k in zip(self.anchor_encoder.encoder.parameters(), self.target_encoder.encoder.parameters()):
            k.data.mul_(self.momentum).add_(q.data * (1.0 - self.momentum))
        for q, k in zip(self.anchor_encoder.decoder.parameters(), self.target_encoder.decoder.parameters()):
            k.data.mul_(self.momentum).add_(q.data * (1.0 - self.momentum))


def ema_update(student_net: BackboneWrapperBase, teacher_net: BackboneWrapperBase,
               momentum: float = 0.9) -> BackboneWrapperBase:
    """
    Perform Exponential Moving Average update on the teacher network

    The teacher is updated with torch.no_grad(True)

    """
    momentum = float(momentum)

    if not 0.0 <= momentum < 1.0:
        momentum = 0.9

    with torch.no_grad():
        student_encoder = student_net.get_encoder()
        teacher_encoder = teacher_net.get_encoder()
        student_decoder = student_net.get_decoder()
        teacher_decoder = teacher_net.get_decoder()
        for q, k in zip(student_encoder.parameters(), teacher_encoder.parameters()):
            k.data.mul_(momentum).add_(q.data * (1.0 - momentum))
        for q, k in zip(student_decoder.parameters(), teacher_decoder.parameters()):
            k.data.mul_(momentum).add_(q.data * (1.0 - momentum))
    return teacher_net


#
# class BasePatchMasking(nn.Module):
#     """
#     Base class for patch masking classes
#     """
#
#     def __init__(self, encoder: SegFormerBackboneWrapper = None,
#                  mask_ratio: float = 0.75, patch_size: int = 4,
#                  image_size: int = 224):
#         super().__init__()
#         if not encoder:
#             raise PatchMaskingException("'encoding' parameter is required, and must be a subclass of "
#                                         "transformers.PreTrainedModel")
#         self.encoder = encoder
#         self.patchify = self.encoder.mask2patches
#
#         if not (0.0 <= mask_ratio < 1.0):
#             raise PatchMaskingException("mask_ratio must be in the range [0.0, 1.0) (1.0 is excluded).")
#
#         self.mask_ratio = mask_ratio
#         self.patch_size = patch_size
#         self.image_size = image_size
#
#         self.grid_size = image_size // patch_size
#         self.num_patches = self.grid_size * self.grid_size
#         self.num_visible = int(self.num_patches * (1.0 - mask_ratio))
#
#         # Patch Embedding Layer from instantiated SegFormer model
#         self.patch_embedding = encoder

#
# class ContextAwarePatchMasking(BasePatchMasking):
#     """
#     Implements structured, context-aware masking for the Anchor View,
#     mimicking techniques like those used for surgical instruments/lesions.
#
#     The masking is determined by a pixel-level input mask, and only unmasked tokens
#     are passed to the MiT encoder.
#     """
#
#     def __init__(self, encoder: SegFormerBackboneWrapper = None,
#                  mask_generator: CompositeMask = CompositeMask(),
#                  mask_ratio: float = 0.75,
#                  patch_size: int = 4,
#                  image_size: int = 512,
#                  min_overlap_threshold: float = 0.5):
#         """
#         Args:
#             patch_size: Size of the non-overlapping patches (e.g., 4 for MiT Stage 1). [1]
#             image_size: Expected input size (e.g., 224).
#             min_overlap_threshold: Minimum mask overlap required for a patch to be
#                                    considered 'masked' (and dropped).
#         """
#         super().__init__(encoder=encoder,
#                          mask_ratio=mask_ratio,
#                          patch_size=patch_size,
#                          image_size=image_size)
#         if not 0 < min_overlap_threshold < 1:
#             raise PatchMaskingException("'min_overlap_threshold' value must be in the range [0.0, 1.0].")
#         self.min_overlap_threshold = min_overlap_threshold
#
#         self.mask_generator = mask_generator
#
#     def _generate_visibility_map(self, pixel_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Converts the high-resolution (H x W) pixel mask into a patch-level
#         binary visibility map (B, N) using average pooling to check overlap.
#
#         A patch is considered 'masked' (value 1) if its average overlap
#         with the pixel_mask exceeds the threshold.
#         """
#         # B, C, H, W = pixel_mask.shape  # (B, 1, H, W)
#         b, c, h, w = pixel_mask.shape
#         if c > 1:
#             pixel_mask = pixel_mask[:, 0, ...].reshape(b, 1, h, w)
#
#         # Use average pooling to calculate the mean mask overlap for each patch area
#         # Kernel size and stride match the patch_size
#         patch_overlap = F.avg_pool2d(
#             pixel_mask.float(),
#             kernel_size=self.patch_size,
#             stride=self.patch_size
#         )  # Output shape: (B, 1, H/P, W/P)
#
#         # Binary Mask: True (1) if overlap > threshold (patch is heavily masked/dropped)
#         # We invert this logic to get the VISIBILITY map (True if patch is visible/kept)
#         # Visible = patch_overlap < threshold
#         visibility_map = (patch_overlap < self.min_overlap_threshold).squeeze(1).bool().unsqueeze(1)
#
#         return visibility_map
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Applies pixel-level masking and performs context-aware token dropping.
#
#         Args:
#             x: Augmented Anchor image tensor (B, 3, H, W).
#
#         Returns:
#             - visible_tokens: The sparse sequence of patches for the MiT encoder (B, N_visible, D).
#             - unmasked_indices: List of retained patch indices (for re-alignment).
#             - masked_indices: List of dropped patch indices.
#         NOTE:
#         Generated pixel mask has 1 = (lesion/instrument/ ..) and 0 = (background/tissue). So
#         take (1 - pixel_mask) to select background/tissue
#         """
#
#         pixel_mask = self.mask_generator.generate_pixel_mask(x)
#         if isinstance(pixel_mask, np.ndarray):
#             pixel_mask = torch.from_numpy(pixel_mask)
#
#         pixel_mask = pixel_mask.float()
#
#         # Patchify the PIXEL MASK to get patch-level visibility
#         visibility_map = self._generate_visibility_map(pixel_mask)
#
#         return visibility_map
#
#     # Use __call__ alias for the forward method
#     __call__ = forward
#
def show_batch(loader: DataLoader, n_batches: int = 1) -> None:
    """
    Plot n_batches batches of images in loader.
    :param loader: Dataloader to pull data from.
    :param n_batches: integer > 0, number of batches to display
    :return: None
    """
    for i, batch in enumerate(loader):
        if i < n_batches:
            grid = []
            n_row = 6
            for j in range(batch['targets'].shape[0]):
                grid += [batch['images'][j],
                         batch['targets'][j],
                         batch['anchors'][j]]
            image_grid = make_grid(grid, nrow=n_row)
            img = torchvision.transforms.ToPILImage()(image_grid)
            img.show()
        else:
            break
    return


def report_cuda_memory_usage(device: torch.device, label=None) -> str:
    out_str = f"Label: {label}, Device; {device}"
    # if device.type.startswith('cuda'):
    #     out_str = label if label is not None else ""
    #     out_str += f"\nMemory Usage:\n{torch.cuda.memory_usage(device=device)}\n"
    #
    #     out_str += f"Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB\n"
    #     out_str += f"Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB\n"
    # else:
    #     out_str = f"\nMDevice {device} is not a CUDA device\n"
    return out_str


def main(params: Dict[str, Any]):
    logger = logging.getLogger(__name__)

    run_once = bool(params.get('run_once', False))
    prefix = params.get('prefix', 'msn_moco')
    if prefix == '':
        prefix = 'msn_moco'

    image_size = (512, 512)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_type = 'cuda'
        scaler = torch.amp.GradScaler()
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        scaler = None

    if debug_run:
        logger.debug(report_cuda_memory_usage(device, label='Beginning of run'))

    logger.info(f'Using device: {device}')

    ds = HDF5DatasetOptimized(hdf5_path=params['dataset'],
                              transform=SSLTransformPipeline(size=image_size))

    batch_sampler = HDF5BatchSampler(ds.dataset_len,
                                     params['batch_size'],
                                     shuffle=True)

    loader = torch.utils.data.DataLoader(ds,
                                         batch_size=None,
                                         sampler=batch_sampler,
                                         shuffle=False,
                                         num_workers=params['num_workers'],
                                         worker_init_fn=hdf5_worker_init_fn
                                         )

    # Instantiate the masking utility
    mask_generator = CompositeMask(shapes_per_image=params['num_shapes'])

    # model = MSNSegFormerAdaptor(backbone=backbone_name)
    # model = MoCoMSN(backbone=backbone_name, mask_generator=mask_generator).to(device)

    """ MoCo setup """
    student_model = SegFormerBackboneWrapper(backbone_name=backbone_name).to(device)
    teacher_model = deepcopy(student_model).to(device)

    optimizer = torch.optim.AdamW(student_model.encoder.parameters(),
                                  lr=params['learning_rate'],
                                  weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'])

    criterion = MaskedCosineSimilarityLoss(reduce='mean')

    """ Set up stopping criteria - stop after 'boredom' steps do not improve loss by 'min_delta' """
    best_loss = float('inf')
    min_delta = 0.0000001
    boredom = 0
    max_boredom = 10
    best_model = None

    logger.info(f"Starting training for {params['num_epochs']} epochs")

    for epoch in range(params['num_epochs']):
        logger.info(f"Starting Epoch {epoch + 1} / {params['num_epochs']}")
        epoch_loss = []
        for idx, batch in enumerate(tqdm(loader)):
            """ Anchor images """
            x_anchor = batch['anchors'].to(device)
            x_anchor.requires_grad = True
            pixel_mask = torch.logical_not(mask_generator.generate_pixel_mask(x_anchor).bool()).to(device)
            x_anchor_masked = x_anchor*pixel_mask.float()

            local_anchors = batch['local_anchors'].to(device)
            # local_anchors.requires_grad = True
            # n_local_repeats = int(local_anchors.shape[0] / pixel_mask.shape[0])
            # local_pixel_mask = pixel_mask.repeat(n_local_repeats, 1, 1, 1).to(device)
            # local_anchors_masked = local_anchors * local_pixel_mask.float()

            """ Target images """
            z_target = batch['targets'].to(device)

            if torch.isnan(x_anchor).any() or torch.isnan(z_target).any() or torch.isnan(local_anchors).any():
                logger.error(f"NaN in input data! x_anchor: {torch.isnan(x_anchor).any()}, "
                             f"z_target: {torch.isnan(z_target).any()}, local_anchors: {torch.isnan(local_anchors).any()}")

            optimizer.zero_grad()

            with autocast(device_type=device_type):

                x_anchor_upscaled = student_model(x_anchor_masked)
                # local_anchors_upscaled = student_model(local_anchors_masked)
                # local_output = [[]]*len(x_anchor_upscaled)
                # for n in range(n_local_repeats):
                #     _output = student_model(local_anchors_masked[n*n_local_repeats:(n+1)*n_local_repeats])
                #     for i in range(len(_output)):
                #         local_output[i] += _output[i]
                # local_anchors_upscaled = tuple(torch.stack(l_o, dim=0) for l_o in local_output)

                with torch.no_grad():
                    z_target_upscaled = teacher_model(z_target)
                    # z_local_upscale = z_target_upscaled.repeat(n_local_repeats, 1, 1, 1).to(device)

                # loss = (0.5*criterion(x_anchor_upscaled, z_target_upscaled, pixel_mask) +
                #         0.5*criterion(local_anchors_upscaled, z_target_upscaled, local_pixel_mask))
                loss = criterion(x_anchor_upscaled, z_target_upscaled, pixel_mask)

                check_is_finite(logger, loss)

            epoch_loss += [loss.item()]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.encoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.encoder.parameters(), max_norm=1.0)
                optimizer.step()


            teacher_model = ema_update(student_model, teacher_model)

            if run_once:
                logger.info(f"Running one time. run_once={run_once}")
                break

        if scheduler is not None:
            scheduler.step()

        avg_loss = np.mean(epoch_loss)
        logger.info(
            f"Epoch {epoch + 1} / {params['num_epochs']}, Mean (per visible patch per encoding layer) loss : {avg_loss:.8f}")

        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            boredom = 0
            logger.info("Saving best snapshot of SegFormer state dict for fine-tuning.")
            try:
                best_model = student_model.model.state_dict()
                torch.save(best_model,
                           f'../segmenter/checkpoint/{prefix}_segformer_pretrained.pt')
            except Exception as e:
                logger.error(f"Pretraining failed to save `{prefix}_segformer_pretrained.pt`: {e}")

        else:
            logger.info(f"Getting bored after {boredom} epochs with no useful improvement")
            boredom += 1
        if boredom > max_boredom:
            logger.info(f"No improvement after {boredom} epochs, terminating")
            break

    logger.info("All done! Exiting")


def check_is_finite(logger: logging.Logger, loss: torch.Tensor)-> None:
    if not torch.isfinite(loss):
        logger.warning("Warning: loss is not finite!")

    try:
        assert loss.requires_grad and loss.grad_fn is not None, "Loss is detached from graph"
    except AssertionError:
        logger.error("Loss is detached from graph")
        logger.error(
            f"Loss: device {loss.device}, dtype:{loss.dtype}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")
    try:
        assert loss.shape == torch.Size([]), f"Loss is not a scalar tensor {loss.shape}"
    except AssertionError:
        logger.error(f"Loss is not a scalar tensor {loss.shape}")
        logger.error(
            f"Loss: device {loss.device}, dtype:{loss.dtype}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")


def get_args():
    """
    Command line arguments

    :return: Dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=source,
                        type=str, help="Path to the HDF5 file.")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, )
    parser.add_argument("-nw", "--num_workers", type=int, default=4, )
    parser.add_argument("-e", "--num_epochs", type=int, default=200, )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, )
    parser.add_argument("-p", "--prefix", type=str, default='moco_msn', )
    parser.add_argument("-ro", "--run_once", type=bool, default=False, )
    parser.add_argument("-ns", "--num_shapes", type=int, default=24, )

    args = parser.parse_args()

    params = {'dataset': args.input,
              'batch_size': args.batch_size,
              'num_workers': args.num_workers,
              'num_epochs': args.num_epochs,
              'learning_rate': args.learning_rate,
              'prefix': args.prefix,
              'run_once': bool(args.run_once),
              'num_shapes': int(args.num_shapes), }

    return params


if __name__ == "__main__":
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        force=True,  # Resets any previous configuration - in Colab for example
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )
    logger = logging.getLogger()
    try:
        params = get_args()
        main(params)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down gracefully.")
    except Exception as ex:
        logger.error(f"Unknown exception occurred. Error: {ex}")
        logger.error(traceback.format_exc())
    finally:
        # ensure log handlers are flushed.
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.info("Logger handlers flushed and closed. Exiting now.")
        sys.exit(0)
