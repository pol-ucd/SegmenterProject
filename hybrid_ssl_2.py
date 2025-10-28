import logging
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm
from transformers import SegformerConfig, SegformerDecodeHead
from transformers import SegformerModel

# Assume utility functions for loss and masking are defined elsewhere
# from utils import mask_and_get_visible_tokens
from segmenter.loss import NTXentLoss, HybridLoss
from segmenter.masks import CompositeMask
from segmenter.utils import HDF5DatasetOptimized, HDF5BatchSampler
from segmenter.utils.data import SSLTransformPipeline, hdf5_worker_init_fn


def mask_and_get_visible_tokens(images, patch_size=16, mask_ratio=0.75):
    """
    Performs MAE-style masking: divides image into patches, randomly masks a ratio,
    and returns the visible tokens, the ground-truth masked patches, and mask indices.

    Args:
        images (Tensor): Input images (B, C, H, W).
        patch_size (int): Size of the square patch.
        mask_ratio (float): Ratio of patches to mask (e.g., 0.75).

    Returns:
        visible_tokens (Tensor): Encoded features of visible patches (B, N_vis, patch_dim).
        masked_patches (Tensor): Ground-truth pixel values of masked patches (B, N_mask, patch_dim).
        mask_positions (Tensor): Indices of the masked patches (B, N_mask).
    """
    B, C, H, W = images.shape

    # --- 1. Patchify and Flatten ---
    # Convert image to sequence of patches and flatten pixels within each patch
    N_H = H // patch_size
    N_W = W // patch_size
    N_total = N_H * N_W  # Total number of patches
    patch_dim = patch_size * patch_size * C  # Flattened size of one patch (e.g., 16*16*3)

    # Reshape image into patches: (B, N_H, N_W, C, P, P) -> (B, N_total, patch_dim)
    patches = images.unfold(2, patch_size, patch_size) \
        .unfold(3, patch_size, patch_size) \
        .permute(0, 2, 3, 1, 4, 5) \
        .reshape(B, N_total, patch_dim)

    # --- 2. Random Masking ---
    num_masked = int(N_total * mask_ratio)

    # Generate random noise for sorting (unique for each batch item)
    noise = torch.rand(B, N_total, device=images.device)

    # Get the indices that sort the noise; this is used for random masking
    ids_shuffle = torch.argsort(noise, dim=1)

    # --- 3. Separate Visible, Masked, and Indices ---

    # Get mask and unmask indices
    mask_positions = ids_shuffle[:, :num_masked]  # The first 'num_masked' indices are the ones we mask
    unmask_positions = ids_shuffle[:, num_masked:]  # The rest are the visible indices

    # Gather the patches based on indices
    # We use a custom utility (or torch.gather) to get the required patches

    # Create mask_positions_expanded for gathering masked patches (B, N_mask, patch_dim)
    mask_positions_expanded = mask_positions.unsqueeze(-1).expand(-1, -1, patch_dim)

    # Ground-truth masked patches (target for reconstruction)
    masked_patches = torch.gather(patches, dim=1, index=mask_positions_expanded)

    # Create unmask_positions_expanded for gathering visible patches (B, N_vis, patch_dim)
    unmask_positions_expanded = unmask_positions.unsqueeze(-1).expand(-1, -1, patch_dim)

    # Input tokens for the encoder (we will need to apply positional embedding later)
    # The encoder only sees these visible patch features
    visible_tokens = torch.gather(patches, dim=1, index=unmask_positions_expanded)

    # Note: In a full ViT/SegFormer implementation, positional embeddings would be added
    # to these 'visible_tokens' before feeding them to the encoder.

    return visible_tokens, masked_patches, mask_positions


class HybridSegFormer(nn.Module):
    def __init__(self, config, backbone, lambda_recon=0.25):
        super().__init__()

        # Shared Backbone: SegFormer Encoder ---
        # The base SegFormer model (encoder only)
        self.encoder = SegformerModel.from_pretrained(pretrained_model_name_or_path=backbone,
                                                      config=config,
                                                      ignore_mismatched_sizes=True)

        # Determine the dimension of the final feature map output from the SegFormer
        # (This is typically the hidden size of the last transformer layer)
        encoder_output_dim = config.hidden_sizes[-1]


        # --- NEW: Custom Pooling Layer for Contrastive Branch ---
        # SegFormer outputs a sequence of tokens (patches) for the last layer (B, N_tokens, D)
        # We use Global Average Pooling (GAP) across the tokens (N_tokens) to get a single vector (B, D)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Contrastive Branch Heads ---
        # Projection Head (g) for contrastive loss: maps Z to H

        # self.projection_head = nn.Sequential(
        #     nn.Linear(encoder_output_dim, encoder_output_dim),
        #     nn.GELU(),
        #     nn.Linear(encoder_output_dim, 256)  # Output dimension for the contrastive embedding
        # )
        self.projection_head = nn.Sequential(
            nn.Linear(16*16, encoder_output_dim),
            nn.GELU(),
            nn.Linear(encoder_output_dim, 256)  # Output dimension for the contrastive embedding
        )

        # Generative Branch Heads ---
        # Lightweight Decoder (d_phi) for reconstruction
        # This decoder takes the encoded features and positional info to reconstruct pixels
        self.reconstruction_decoder = SegformerDecodeHead(config)

        #Masking function
        self.mask_generator = CompositeMask()
        # Loss functions
        self.contrastive_loss_fn = NTXentLoss(temperature=0.07)
        # self.reconstruction_loss_fn = nn.MSELoss()
        loss_config = {
        "ce": {"weight": 0.2},
        "iou": {"weight": 0.8},
        }
        self.reconstruction_loss_fn = HybridLoss(**loss_config)

        # Hyperparameter for balancing losses
        self.lambda_recon = lambda_recon

    def forward_contrastive(self, x_i, x_j):
        # The encoder returns a BaseModelOutput object.
        # setting output_hidden_states=True is usually necessary in the config
        # to get all features, but SegFormerModel returns the list of multi-scale
        # features in the 'hidden_states' attribute by default.

        # 1. Get multi-scale features (list of 4 Tensors)
        output_i = self.encoder(x_i, output_hidden_states=True,
                                return_dict=True).hidden_states  # List of (B, H/16 * W/16, D_k) Tensors
        output_j = self.encoder(x_j, output_hidden_states=True,
                                return_dict=True).hidden_states

        # 2. Extract the features from the final, highest-level stage (index -1)
        # This gives a tensor of shape (B, N_tokens, D)
        final_tokens_i = output_i[-1].flatten(start_dim=-2, end_dim=-1)
        final_tokens_j = output_j[-1].flatten(start_dim=-2, end_dim=-1)

        # 3. Custom Global Average Pooling (GAP)
        # Transpose to (B, D, N_tokens) for nn.AdaptiveAvgPool1d, then squeeze the result.
        # This collapses the token dimension (N_tokens) to 1, creating the image embedding (Z).
        z_i = self.global_pool(final_tokens_i.transpose(1, 2)).squeeze(-1)  # Shape: (B, D)
        z_j = self.global_pool(final_tokens_j.transpose(1, 2)).squeeze(-1)  # Shape: (B, D)
        # z_i = self.global_pool(final_tokens_i).squeeze(-1)  # Shape: (B, D)
        # z_j = self.global_pool(final_tokens_j).squeeze(-1)  # Shape: (B, D)

        # 4. Project features (H)
        h_i = self.projection_head(z_i)
        h_j = self.projection_head(z_j)

        # 5. Compute Contrastive Loss (L_cont)
        loss_cont = self.contrastive_loss_fn(h_i, h_j)

        return loss_cont, h_i, h_j

    def forward_generative(self, x):
        _, _, h, w = x.shape
        # Masking
        # Mask a portion of the image and separate into visible tokens and masked patches
        # visible_tokens, masked_patches, mask_positions = mask_and_get_visible_tokens(x)
        pixel_mask = torch.logical_not(self.mask_generator.generate_pixel_mask(x).bool()).to(x.device)
        visible_tokens = x * pixel_mask.float()

        # Encode Visible Tokens
        # The encoder only processes the visible tokens
        encoder_output = self.encoder(visible_tokens, output_hidden_states=True, return_dict=True).hidden_states

        # Decode for Reconstruction
        # The decoder predicts the content of the masked patches

        predicted_patches = self.reconstruction_decoder(encoder_output)
        predicted_patches = F.interpolate(predicted_patches,
                                          size=(h, w),
                                          mode='bilinear',
                                          align_corners=False)

        predicted_patches = torch.argmax(predicted_patches, keepdim=True, dim=1).float()

        # Compute Reconstruction Loss (L_recon)
        loss_recon = self.reconstruction_loss_fn(predicted_patches, pixel_mask)

        return loss_recon

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # --- Contrastive Pass ---
        x_i = x['anchors']
        x_j = x['targets']
        x_orig = x['images']
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


if __name__ == '__main__':
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger()
    params = {'batch_size': 8,
              'dataset': '../segmenter/data/pretrain_images.h5',
              'num_workers': 4, }
    backbone_name = "nvidia/segformer-b4-finetuned-ade-512-512"
    num_epochs = 200

    image_size = (512, 512)
    prefix = 'hybrid_ssl'

    # Example Initialization (using placeholder values)
    ds = HDF5DatasetOptimized(hdf5_path=params['dataset'],
                              transform=SSLTransformPipeline(size=image_size))

    batch_sampler = HDF5BatchSampler(ds.dataset_len,
                                     params['batch_size'],
                                     shuffle=True)

    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size=None,
                                             sampler=batch_sampler,
                                             shuffle=False,
                                             num_workers=params['num_workers'],
                                             worker_init_fn=hdf5_worker_init_fn
                                             )
    config = SegformerConfig.from_pretrained(backbone_name)
    model = HybridSegFormer(config, backbone=backbone_name, lambda_recon=0.2)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    model.to(device)

    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'])

    # -----------------------------

    logger.info("Starting Self-Supervised Pre-training...")

    """ Set up stopping criteria - stop after 'boredom' steps do not improve loss by 'min_delta' """
    best_loss = float('inf')
    min_delta = 0.0000001
    boredom = 0
    max_boredom = 10
    best_model = None

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_epoch_loss = []
        total_contrastive_loss = []
        total_reconstruction_loss = []

        # Iterate over the dataset
        for step, data in enumerate(tqdm(dataloader)):

            # 1. Move Data to Device
            x = {}
            for key, value in data.items():
                x[key] = data[key].to(device)

            # 2. Zero the Gradients
            # Always start by clearing old gradients
            optimizer.zero_grad()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):

                # The model calculates L_cont, L_recon, and combines them into L_total
                loss_output = model(x)
                loss_total = loss_output['loss_total']

            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
                optimizer.step()


            # 5. Model Parameter Update
            # The optimizer uses the calculated gradients (from step 4)
            # to adjust the parameters (w = w - lr * dL/dw).
            # This updates the weights of the SegFormer encoder and both heads simultaneously.
            optimizer.step()

            # 6. Logging and Tracking
            total_epoch_loss += [loss_total.item()]
            total_contrastive_loss += [loss_output['loss_contrastive'].item()]
            total_reconstruction_loss += [loss_output['loss_reconstruction'].item()]

        if scheduler is not None:
            scheduler.step()

        avg_epoch_loss = np.mean(total_epoch_loss)
        avg_contrastive_loss = np.mean(total_contrastive_loss)
        avg_reconstruction_loss = np.mean(total_reconstruction_loss)

        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
                    f"Average Contrastive Loss: {avg_contrastive_loss:.4f}, "
                    f"Average Reconstruction Loss: {avg_reconstruction_loss:.4f}")


        if avg_epoch_loss + min_delta < best_loss:
            best_loss = avg_epoch_loss
            boredom = 0
            logger.info("Saving best snapshot of SegFormer state dict for fine-tuning.")
            try:
                best_model = model.encoder.state_dict()
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

    logger.info("Pre-training complete.")
