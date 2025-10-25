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
import transformers
from torch import nn, autocast
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel, SegformerLayer

from segmenter.core import report_cuda_memory_usage
from segmenter.loss import MaskedCosineSimilarityLoss, EncodingCosineSimilarityLoss
from segmenter.utils import HDF5DatasetOptimized
from segmenter.utils.data import SSLTransformPipeline, HDF5BatchSampler, hdf5_worker_init_fn

""" DEBUG helpers - Comment out unless debugging """
import os

# Force synchronous CUDA errors to surface
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Enable autograd anomaly detection and print where NaNs appear
torch.autograd.set_detect_anomaly(True)
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


    @staticmethod
    def _get_components(item: torch.nn.Module) -> List[str]:
        """
        Helper for exploring the layers in a model. The layers are not always named as
        expected
        :param item: An instantiate model.
        :return: a list of layer names
        """
        keys = set()
        for key, value in item.named_parameters():
            key_0 = key.split('.')[0]
            keys.add(key_0)
        return list(keys)

    def __repr__(self):
        return f"{self.__class__.__name__}(backbone={self.backbone})"



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
        self.encoder = self.model.segformer
        # Include the decoder head in case we want to replace it later
        self.decoder = self.model.decode_head


    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        encodings = self.encoder(x, output_hidden_states=True, return_dict=True)['hidden_states']
        return encodings


class SimpleDecoder(nn.Module):
    def __init__(self, in_dim, patch_size=16, num_channels=3):
        super().__init__()

        # Calculate the dimension of a flattened image patch
        self.patch_dim = patch_size * patch_size * num_channels  # e.g., 16*16*3 = 768

        # --- Decoder Layers ---
        # The decoder takes the encoded token dimension (in_dim) and maps it
        # to the patch dimension (patch_dim).

        # 1. Main Linear Layer: Maps the hidden state to the patch pixel space
        # A very simple decoder might only use this one layer.
        self.decoder_pred = nn.Linear(in_dim, self.patch_dim)

        # 2. Optional: Another layer for non-linearity (can be removed for simplicity)
        # self.optional_layer = nn.Sequential(
        #     nn.Linear(in_dim, in_dim // 2),
        #     nn.GELU(),
        #     nn.Linear(in_dim // 2, self.patch_dim)
        # )

        # Mask Token: A learnable parameter representing the content of a masked patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_dim))

        # Initialize the mask token
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, encoded_visible_tokens, mask_positions):
        """
        Args:
            encoded_visible_tokens (Tensor): (B, N_vis, D) Encoded features from the SegFormer (D is in_dim).
            mask_positions (Tensor): (B, N_mask) Indices of the masked patches in the full sequence.

        Returns:
            Tensor: (B, N_mask, patch_dim) Predicted pixel values for the masked patches.
        """
        B, N_vis, D = encoded_visible_tokens.shape
        N_mask = mask_positions.shape[1]

        # 1. Create a full sequence of features (Visible + Mask Tokens)

        # Create a batch of mask tokens for the full sequence length (N_vis + N_mask)
        # The total number of tokens (N_total) might be fixed (e.g., 256 for a 256x256 image / 16x16 patches)

        # Find the max sequence length (N_total)
        N_total = N_vis + N_mask

        # Create a tensor for the full sequence, initially filled with mask tokens
        full_sequence = torch.zeros((B, N_total, D), device=encoded_visible_tokens.device)

        # Determine the indices of the visible tokens
        visible_indices = torch.ones((B, N_total), dtype=torch.bool, device=encoded_visible_tokens.device)
        # Mark the masked positions as False
        visible_indices.scatter_(1, mask_positions, False)

        # Place the encoded visible tokens back into their original positions
        full_sequence[visible_indices] = encoded_visible_tokens.flatten(0, 1)  # B*N_vis, D

        # Fill the mask positions with the learned mask token
        # The mask token is broadcasted across the batch and sequence length
        full_sequence[~visible_indices] = self.mask_token.expand(B * N_mask, -1)  # B*N_mask, D

        # 2. Decode: Predict pixel values for ALL tokens (both visible and masked)
        # However, we only care about the masked positions for the loss calculation.
        predictions_all = self.decoder_pred(full_sequence)

        # 3. Extract Predictions for Masked Patches
        # predictions_all is (B, N_total, patch_dim)
        # The mask_positions tells us which N_mask tokens to extract for the loss

        # Get the indices in a way that can be used to gather (B, N_mask, patch_dim)
        mask_positions_expanded = mask_positions.unsqueeze(-1).expand(-1, -1, self.patch_dim)

        predicted_patches = torch.gather(predictions_all, dim=1, index=mask_positions_expanded)

        return predicted_patches  # (B, N_mask, patch_dim)


class HybridSegFormer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # --- 1. Shared Backbone: SegFormer Encoder ---
        # The base SegFormer model (encoder only)
        self.encoder = SegformerModel(config)

        # Determine the dimension of the final feature map output from the SegFormer
        # (This is typically the hidden size of the last transformer layer)
        encoder_output_dim = config.hidden_sizes[-1]

        # --- 2. Contrastive Branch Heads ---
        # Projection Head (g) for contrastive loss: maps Z to H
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.GELU(),
            nn.Linear(encoder_output_dim, 256)  # Output dimension for the contrastive embedding
        )

        # --- 3. Generative Branch Heads ---
        # Lightweight Decoder (d_phi) for reconstruction
        # This decoder takes the encoded features and positional info to reconstruct pixels
        self.reconstruction_decoder = SimpleDecoder(
            in_dim=encoder_output_dim,
            patch_size=config.patch_sizes[0]
        )

        # Loss functions
        self.contrastive_loss_fn = NTXentLoss(temperature=0.07)
        self.reconstruction_loss_fn = nn.MSELoss()

        # Hyperparameter for balancing losses
        self.lambda_recon = 1.0  # Can be tuned

    def forward_contrastive(self, x_i, x_j):
        # 1. Get features (Z) from both views
        # We take the *pooled* output for instance contrastive learning
        z_i = self.encoder(x_i).pooler_output
        z_j = self.encoder(x_j).pooler_output

        # 2. Project features (H)
        h_i = self.projection_head(z_i)
        h_j = self.projection_head(z_j)

        # 3. Compute Contrastive Loss (L_cont)
        # This function typically handles the positive pair (i, j) and negatives from the batch
        loss_cont = self.contrastive_loss_fn(h_i, h_j)

        return loss_cont, h_i, h_j

    def forward_generative(self, x):
        # 1. Masking
        # Mask a portion of the image and separate into visible tokens and masked patches
        visible_tokens, masked_patches, mask_positions = mask_and_get_visible_tokens(x)

        # 2. Encode Visible Tokens
        # The encoder only processes the visible tokens
        encoder_output = self.encoder(visible_tokens).last_hidden_state

        # 3. Decode for Reconstruction
        # The decoder predicts the content of the masked patches
        predicted_patches = self.reconstruction_decoder(encoder_output, mask_positions)

        # 4. Compute Reconstruction Loss (L_recon)
        loss_recon = self.reconstruction_loss_fn(predicted_patches, masked_patches)

        return loss_recon

    def forward(self, x_i, x_j, x_orig):
        # --- Contrastive Pass ---
        loss_cont, _, _ = self.forward_contrastive(x_i, x_j)

        # --- Generative Pass ---
        loss_recon = self.forward_generative(x_orig)

        # --- Total Hybrid Loss ---
        loss_total = loss_cont + (self.lambda_recon * loss_recon)

        return {
            'loss_total': loss_total,
            'loss_contrastive': loss_cont,
            'loss_reconstruction': loss_recon
        }


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

        for q, k in zip(student_net.parameters(), teacher_net.parameters()):
            k.data.mul_(momentum).add_(q.data * (1.0 - momentum))
    return teacher_net


def main(params: Dict[str, Any]):
    logger = logging.getLogger(__name__)

    run_once = bool(params.get('run_once', False))
    prefix = params.get('prefix', 'msn_moco')
    if prefix == '':
        prefix = 'msn_moco'

    image_size = (512, 512)
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
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

    """ MoCo setup """
    student_model = SegFormerBackboneWrapper(backbone_name=backbone_name).to(device)
    teacher_model = deepcopy(student_model).to(device)

    optimizer = torch.optim.AdamW(student_model.encoder.parameters(),
                                  lr=params['learning_rate'],
                                  weight_decay=1e-2)
    # optimizer = torch.optim.SGD(student_model.parameters(),
    #                             lr=params['learning_rate'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'])

    criterion = MaskedCosineSimilarityLoss()
    # criterion = EncodingCosineSimilarityLoss()

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
            x_anchor = batch['anchors']

            pixel_mask = torch.logical_not(mask_generator.generate_pixel_mask(x_anchor).bool())
            x_anchor_masked = x_anchor*pixel_mask.float()
            x_anchor_masked = x_anchor_masked.to(device)

            local_anchors = batch['local_anchors']
            local_anchors_masked = local_anchors*pixel_mask.float()
            local_anchors_masked = local_anchors_masked.to(device)

            pixel_mask = pixel_mask.to(device)


            """ Target images """
            z_target = batch['targets'].to(device)
            z_target.requires_grad = False

            optimizer.zero_grad()

            with autocast(device_type=device_type):

                x_anchor_upscaled = student_model(x_anchor_masked)
                local_anchors_upscaled = student_model(local_anchors_masked)

                with torch.no_grad():
                    z_target_upscaled = teacher_model(z_target)

                # loss = (0.5*criterion(x_anchor_upscaled, z_target_upscaled) +
                #         0.5*criterion(local_anchors_upscaled, z_target_upscaled))
                loss = (0.5*criterion(x_anchor_upscaled, z_target_upscaled, pixel_mask) +
                        0.5*criterion(local_anchors_upscaled, z_target_upscaled, pixel_mask))
                # loss = criterion(x_anchor_upscaled, z_target_upscaled, pixel_mask)
                # loss = criterion(x_anchor_upscaled, z_target_upscaled)

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
            torch.cuda.empty_cache()
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
