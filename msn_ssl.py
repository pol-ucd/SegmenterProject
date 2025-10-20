import argparse
import logging
import sys
from copy import deepcopy
from os import PathLike
from typing import Any, Dict, Union, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, autocast
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from segmenter.loss import MaskedCosineSimilarityLoss
from segmenter.utils import HDF5DatasetOptimized
from segmenter.utils.data import SSLTransformPipeline, HDF5BatchSampler, hdf5_worker_init_fn

MASK_RATIO = 0.7
PATCH_SIZE = 4  # 4x4 patches for SegFormer MiT [1]
IMAGE_H = 512
IMAGE_W = 512

# source = "/Users/polmacaonghusa/Documents/Projects/segmenter/data/Classica.h5"
# source = "/Users/polmacaonghusa/Documents/Projects/segmenter/data/pretrain_images.h5"
source = "../segmenter/data/pretrain_images.h5"

backbone_name = "nvidia/segformer-b4-finetuned-ade-512-512"

from segmenter.masks import CompositeMask


class PatchMaskingException(Exception):
    pass


class MSNSegFormerAdaptor(nn.Module):
    def __init__(self, backbone: Union[str, PathLike] = backbone_name):
        super().__init__()
        self.backbone = backbone
        self.cfg = SegformerConfig.from_pretrained(backbone) if backbone else SegformerConfig()
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(
                backbone, config=self.cfg, ignore_mismatched_sizes=True
            )
            if backbone
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

        But we have a predefined model here so we can assume.
        """
        self.patch_embeddings = self.encoder.patch_embeddings  # List of 4 patch embedding modules
        self.blocks = self.encoder.block  # List of 4 ModuleLists of blocks
        self.norms = self.encoder.layer_norm  # List of 4 Norm modules

        # self.processor = SegformerImageProcessor()
        # Determine the patch size/stride for the first stage (MiT-style structure)
        self.strides = self.encoder.config.patch_sizes
        self.initial_stride = self.strides[0]
        self.hidden_dims = self.encoder.config.hidden_sizes
        self.initial_hidden_dim = self.hidden_dims[0]

    def forward(self, x, **kwargs):
        encodings = self.model(x, output_hidden_states=True, return_dict=True)['hidden_states']
        return encodings

    def mask2patches(self, mask: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Calculate Patch Embeddings from the backbone model patch embeddings layer.
        Runs the first patch embedding module only.
        :param mask: torch.Tensor 0-1 mask
        :return: Tuple[torch.Tensor(patch_embedding(B, HxW, N_features),
                            int(Height - e.g. 128), int(Width - e.g. 128)]
        """
        return self.initial_patch_layer(mask)

    # --------------------------------------------------------------------------
    # 4) Function to Upscale SegFormer Embeddings
    # --------------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def upscale_embeddings(stage_embeddings: Union[List[torch.Tensor], Tuple[torch.Tensor]]):
        """Upscales all encoder outputs to the spatial resolution of the initial patches (L1)."""

        _, _, h_upscale, w_upscale = stage_embeddings[0].shape

        upscaled_outputs = []
        for features in stage_embeddings:
            upscaled_spatial = F.interpolate(
                features,
                size=(h_upscale, w_upscale),
                mode='bilinear',
                align_corners=False
            )

            upscaled_outputs.append(upscaled_spatial)
        return tuple(upscaled_outputs)

    @staticmethod
    @torch.no_grad()
    def _get_components(item: torch.nn.Module) -> List[str]:
        keys = set()
        for key, value in item.named_parameters():
            key_0 = key.split('.')[0]
            keys.add(key_0)
        return list(keys)

    def __repr__(self):
        return f"{self.__class__.__name__}(backbone={self.backbone})"


class MoCoMSN(nn.Module):
    def __init__(self, backbone: Union[str, PathLike] = backbone_name,
                 momentum: float = 0.99):
        super().__init__()
        self.backbone = backbone
        self.momentum = float(momentum)

        self.anchor_encoder = MSNSegFormerAdaptor(backbone)
        self.target_encoder = deepcopy(self.anchor_encoder)
        self._set_requires_grad(self.target_encoder, False)

    def forward(self, anchor: torch.Tensor, target: torch.Tensor):
        anchor = anchor.to(target.device)
        target = target.to(target.device)
        anchor_encodings = self.anchor_encoder(anchor)
        anchor_encodings_upscaled = self.anchor_encoder.upscale_embeddings(anchor_encodings)

        with torch.no_grad():
            target_encodings = self.target_encoder(target)
            target_encodings_upscaled = self.target_encoder.upscale_embeddings(target_encodings)

        return anchor_encodings_upscaled, target_encodings_upscaled

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


class BasePatchMasking(nn.Module):
    """
    Base class for patch masking classes
    """

    def __init__(self, encoder: MSNSegFormerAdaptor = None,
                 mask_ratio: float = 0.75, patch_size: int = 4,
                 image_size: int = 224):
        super().__init__()
        if not encoder:
            raise PatchMaskingException("'encoding' parameter is required, and must be a subclass of "
                                        "transformers.PreTrainedModel")
        self.encoder = encoder
        self.patchify = self.encoder.mask2patches

        if not (0.0 <= mask_ratio < 1.0):
            raise PatchMaskingException("mask_ratio must be in the range [0.0, 1.0) (1.0 is excluded).")

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.image_size = image_size

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.num_visible = int(self.num_patches * (1.0 - mask_ratio))

        # Patch Embedding Layer from instantiated SegFormer model
        self.patch_embedding = encoder


class ContextAwarePatchMasking(BasePatchMasking):
    """
    Implements structured, context-aware masking for the Anchor View,
    mimicking techniques like those used for surgical instruments/lesions.

    The masking is determined by a pixel-level input mask, and only unmasked tokens
    are passed to the MiT encoder.
    """

    def __init__(self, encoder: MSNSegFormerAdaptor = None,
                 mask_generator: CompositeMask = CompositeMask(),
                 mask_ratio: float = 0.75,
                 patch_size: int = 4,
                 image_size: int = 512,
                 min_overlap_threshold: float = 0.5):
        """
        Args:
            patch_size: Size of the non-overlapping patches (e.g., 4 for MiT Stage 1). [1]
            image_size: Expected input size (e.g., 224).
            min_overlap_threshold: Minimum mask overlap required for a patch to be
                                   considered 'masked' (and dropped).
        """
        super().__init__(encoder=encoder,
                         mask_ratio=mask_ratio,
                         patch_size=patch_size,
                         image_size=image_size)
        if not 0 < min_overlap_threshold < 1:
            raise PatchMaskingException("'min_overlap_threshold' value must be in the range [0.0, 1.0].")
        self.min_overlap_threshold = min_overlap_threshold

        self.mask_generator = mask_generator

    def _generate_visibility_map(self, pixel_mask: torch.Tensor) -> torch.Tensor:
        """
        Converts the high-resolution (H x W) pixel mask into a patch-level
        binary visibility map (B, N) using average pooling to check overlap.

        A patch is considered 'masked' (value 1) if its average overlap
        with the pixel_mask exceeds the threshold.
        """
        # B, C, H, W = pixel_mask.shape  # (B, 1, H, W)
        b, c, h, w = pixel_mask.shape
        if c > 1:
            pixel_mask = pixel_mask[:, 0, ...].reshape(b, 1, h, w)

        # Use average pooling to calculate the mean mask overlap for each patch area
        # Kernel size and stride match the patch_size
        patch_overlap = F.avg_pool2d(
            pixel_mask.float(),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # Output shape: (B, 1, H/P, W/P)

        # Binary Mask: True (1) if overlap > threshold (patch is heavily masked/dropped)
        # We invert this logic to get the VISIBILITY map (True if patch is visible/kept)
        # Visible = patch_overlap < threshold
        visibility_map = (patch_overlap < self.min_overlap_threshold).squeeze(1).bool().unsqueeze(1)

        return visibility_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies pixel-level masking and performs context-aware token dropping.

        Args:
            x: Augmented Anchor image tensor (B, 3, H, W).

        Returns:
            - visible_tokens: The sparse sequence of patches for the MiT encoder (B, N_visible, D).
            - unmasked_indices: List of retained patch indices (for re-alignment).
            - masked_indices: List of dropped patch indices.
        NOTE:
        Generated pixel mask has 1 = (lesion/instrument/ ..) and 0 = (background/tissue). So
        take (1 - pixel_mask) to select background/tissue
        """

        pixel_mask = self.mask_generator.generate_pixel_mask(x)
        if isinstance(pixel_mask, np.ndarray):
            pixel_mask = torch.from_numpy(pixel_mask)

        pixel_mask = pixel_mask.float()

        # Patchify the PIXEL MASK to get patch-level visibility
        visibility_map = self._generate_visibility_map(pixel_mask)

        return visibility_map

    # Use __call__ alias for the forward method
    __call__ = forward


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


def main(params: Dict[str, Any]):
    logger = logging.getLogger(__name__)

    prefix = params.get('prefix', 'msn_moco')
    if prefix == '':
        prefix = 'msn_moco'

    image_size = (512, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # model = MSNSegFormerAdaptor(backbone=backbone_name)
    model = MoCoMSN(backbone=backbone_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params['learning_rate'],
                                  weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'])

    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    criterion = MaskedCosineSimilarityLoss(reduce='mean')

    # Instantiate the masking utility
    mask_generator = CompositeMask(mask_ratio=MASK_RATIO)

    """ Set up stopping criteria - stop after 'boredom' steps do not improve loss by 'min_delta' """
    best_loss = float('inf')
    min_delta = 0.00001
    boredom = 0
    max_boredom = 10
    best_model = None

    logger.info(f"Starting training for {params['num_epochs']} epochs")
    model.train()
    for epoch in range(params['num_epochs']):
        logger.info(f"Starting Epoch {epoch + 1} / {params['num_epochs']}")
        epoch_loss = []
        for idx, batch in enumerate(tqdm(loader)):
            """ Anchor images """
            x_anchor = batch['anchors'].to(device)
            # x_anchor_mask = mask_utility(x_anchor).to(device)

            """ Target images """
            z_target = batch['targets'].to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                x_anchor_upscaled, z_target_upscaled = model(x_anchor, z_target)

                x_anchor_mask = mask_generator.generate_pixel_mask(x_anchor_upscaled[0]).to(device)

                loss = criterion(x_anchor_upscaled, z_target_upscaled, x_anchor_mask)

            epoch_loss += [loss.cpu().detach().item()]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                model.update_momentum_encoder()

        if scheduler is not None:
            scheduler.step()

        avg_loss = np.mean(epoch_loss)
        logger.info(
            f"Epoch {epoch + 1} / {params['num_epochs']}, Mean (per pixel per encoding layer) loss : {avg_loss:.4f}")

        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            boredom = 0
            logger.info("Saving best snapshot `msn_model.online_encoder` state dict for fine-tuning.")
            try:
                best_model = model.anchor_encoder.model.state_dict()
                torch.save(best_model,
                           f'../segmenter/checkpoint/{prefix}_segformer_pretrained.pt')
            except Exception as e:
                logger.error(f"Pretraining failed to save `{prefix}_segformer_pretrained.pt`: {e}")

        else:
            boredom += 1
        if boredom > max_boredom:
            logger.info(f"No improvement after {boredom} epochs, terminating")
            break

    logger.info("All done! Exiting")


def get_args():
    """
    Command line arguments

    :return: Dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=source,
                        type=str, help="Path to the HDF5 file.")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, )
    parser.add_argument("-nw", "--num_workers", type=int, default=4, )
    parser.add_argument("-e", "--num_epochs", type=int, default=200, )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, )
    parser.add_argument("-p", "--prefix", type=str, default='moco_msn', )

    args = parser.parse_args()

    params = {'dataset': args.input,
              'batch_size': args.batch_size,
              'num_workers': args.num_workers,
              'num_epochs': args.num_epochs,
              'learning_rate': args.learning_rate,
              'prefix': args.prefix, }

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
        sys.exit(0)
    finally:
        # This block will always be executed, allowing you to clean up resources
        # ensure log handlers are flushed.
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.info("Logger handlers flushed and closed. Exiting now.")
