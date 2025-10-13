"""
Mixed Siamese Networks for Bring Your Own Labels (BYOL) implementation

    The following architectural styles are implemented:
    - Momentum Contrast (MoCo): using two encoders and updating one with EMA,
    - SimCLR: Gradients flow from both views to the single shared encoder
    - SimSiam: detached gradient on one branch of a shared encoder's output

    Pretrain a backbone model with unlabelled images using one of the
    MSN architectures below.

    Fine-tune the pretrained model with a small subset of annotated examples

    Validate with the remaining annotated images
"""
import random
from copy import deepcopy
from typing import Tuple, Optional, OrderedDict

import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from segmenter.masks import apply_custom_augmentations
from segmenter.models.base import MedianPool2d, AugurSegmenterBase, SegformerModelError


class MaskedTiledViewGenerator(nn.Module):
    def __init__(self, mask_composer,
                 tile_size=(64, 64),
                 return_metadata=False):
        super(MaskedTiledViewGenerator, self).__init__()
        self.mask_composer = mask_composer  # e.g., SurgicalMaskComposer, kwargs passed as params
        self.tile_size = tile_size
        self.return_metadata = return_metadata

    def tile_image(self, image):
        B, C, H, W = image.shape
        th, tw = self.tile_size
        assert H % th == 0 and W % tw == 0, "Image H,W must be divisible by tile h,w"
        tiles = image.unfold(2, th, th).unfold(3, tw, tw)
        tiles = tiles.contiguous().view(B, C, -1, th, tw)  # (B, C, N_tiles, th, tw)
        return tiles

    def apply_masks(self, tiles):
        B, C, N, th, tw = tiles.shape
        masked_tiles = []
        metadata = []

        for b in range(B):
            batch_tiles = []
            batch_meta = []
            for n in range(N):
                tile = tiles[b, :, n]
                masked_tile, mask_info = self.mask_composer(tile)
                batch_tiles.append(masked_tile)
                batch_meta.append(mask_info)
            masked_tiles.append(torch.stack(batch_tiles))  # (N, C, th, tw)
            metadata.append(batch_meta)

        masked_tiles = torch.stack(masked_tiles)  # (B, N, C, th, tw)
        return masked_tiles, metadata

    def stitch_tiles(self, masked_tiles, H, W):
        # B, N, C, th, tw = masked_tiles.shape
        B, C, N, th, tw = masked_tiles.shape
        tiles_per_row, tiles_per_col = W // tw, H // th
        tiles = masked_tiles.reshape(B, C, tiles_per_row, tiles_per_col, th, tw)

        rows = [torch.cat([tiles[:, :, r, c, ...] for r in range(tiles_per_row)], dim=2)
                for c in range(tiles_per_col)]
        # for b in range(B)]
        stitched = torch.cat(rows, dim=-1)  # (B, C, H, W)
        return stitched

    def __call__(self, image):
        B, C, H, W = image.shape
        tiles = self.tile_image(image)  # (B, C, N, th, tw)
        masked_tiles, metadata = self.apply_masks(tiles)
        masked_tiles = masked_tiles.permute(0, 2, 1, 3, 4)  # (B, N, C, th, tw)
        masked_image = self.stitch_tiles(masked_tiles, H, W)

        if self.return_metadata:
            return masked_image, metadata
        return masked_image


class SurgicalMaskComposer(nn.Module):
    def __init__(self, instrument_prob=0.3, fluid_prob=0.3, fold_prob=0.4):
        super(SurgicalMaskComposer, self).__init__()
        self.mask_types = ['instrument', 'fluid', 'fold']
        self.probs = [instrument_prob, fluid_prob, fold_prob]

    def __call__(self, tile):
        mask_type = random.choices(self.mask_types, weights=self.probs, k=1)[0]
        masked_tile, params = getattr(self, f"mask_{mask_type}")(tile)
        return masked_tile, {'type': mask_type, 'params': params}

    def mask_instrument(self, tile):
        # Simulate rigid occlusion (e.g., scalpel, grasper)
        occlusion = torch.zeros_like(tile)
        x = random.randint(0, tile.shape[2] // 2)
        y = random.randint(0, tile.shape[1] // 2)
        w = random.randint(tile.shape[2] // 4, tile.shape[2] // 2)
        h = random.randint(tile.shape[1] // 8, tile.shape[1] // 4)
        occlusion[:, y:y + h, x:x + w] = 1.0
        masked = tile * (1 - occlusion)
        return masked, {'x': x, 'y': y, 'w': w, 'h': h}

    def mask_fluid(self, tile):
        # Simulate semi-transparent smear or pooling
        alpha = random.uniform(0.3, 0.7)
        smear = torch.randn_like(tile) * 0.2 + 0.5
        masked = tile * (1 - alpha) + smear * alpha
        return masked.clamp(0, 1), {'alpha': alpha}

    def mask_fold(self, tile):
        # Simulate tissue fold with curved occlusion
        fold = torch.ones_like(tile)
        cx = random.randint(tile.shape[2] // 4, tile.shape[2] * 3 // 4)
        cy = random.randint(tile.shape[1] // 4, tile.shape[1] * 3 // 4)
        radius = random.randint(tile.shape[1] // 6, tile.shape[1] // 3)
        yy, xx = torch.meshgrid(torch.arange(tile.shape[1]), torch.arange(tile.shape[2]), indexing='ij')
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
        fold[:, mask] = 0.0
        masked = tile * fold
        return masked, {'cx': cx, 'cy': cy, 'radius': radius}

#
# class SegFormerAdapter(nn.Module):
#     """
#     Calls a pretrained SegFormer encoder and returns a (B, C, H, W) feature map.
#     Adjust token->spatial conversion to match the specific SegFormer variant you use.
#     """
#
#     def __init__(self, pretrained_name: str = None, num_classes: int = 2, k: int = 3):
#         super().__init__()
#         self.num_classes = num_classes
#         if pretrained_name is not None:
#             config = SegformerConfig.from_pretrained(pretrained_name)
#         else:
#             config = SegformerConfig()
#
#         if pretrained_name is not None:
#             self.base_model = SegformerForSemanticSegmentation.from_pretrained(
#                 pretrained_name,
#                 config=config,
#                 ignore_mismatched_sizes=True
#             )
#         else:
#             self.base_model = SegformerForSemanticSegmentation(config=config)
#
#         # Get the number of channels from the previous layer to properly
#         # define the input to our new classifier.
#         classifier_in_channels = self.base_model.decode_head.linear_fuse.out_channels
#
#         # Replace the original classifier with a custom Sequential module.
#         self.base_model.decode_head.classifier = nn.Sequential(
#             # First convolution layer to process the features.
#             nn.Conv2d(classifier_in_channels,
#                       classifier_in_channels // 4,
#                       kernel_size=3, padding=1),
#             # Batch normalization for training stability.
#             nn.BatchNorm2d(classifier_in_channels // 4),
#             # ReLU activation for non-linearity.
#             nn.ReLU(inplace=True),
#             # Final convolution to map features to the desired number of classes.
#             nn.Conv2d(classifier_in_channels // 4, num_classes, kernel_size=3, padding=1)
#         )
#         self.median = MedianPool2d(kernel_size=k, padding=k // 2)
#
#     def forward(self, x):
#
#         # The base model's forward pass handles the entire encoder and decoder.
#         # We only need the logits.
#         output = self.base_model(pixel_values=x.float()).logits
#
#         # The Segformer model's output logits are at a reduced resolution (e.g., 1/4th).
#         # We upsample them back to the original input size.
#         logits = F.interpolate(output,
#                                size=x.shape[2:],
#                                mode='bilinear',
#                                align_corners=False)  #.permute(0, 2, 3, 1).contiguous()
#
#         return self.median(logits)  # Smoothed logits
#
#     def output_dim(self):
#         return self.num_classes
#
#
# class SegFormerMSNWithMomentum(nn.Module):
#     """
#     SegFormer feature adapter for MSN-style pretraining.
#
#     - Extracts highest-level encoder features as patch embeddings.
#     - Optional projection head to lower dimensionality.
#     - Random patch masking for the online branch.
#     - Maintains a momentum (EMA) copy of the encoder (target encoder).
#     - Exposes update_momentum_encoder(step) to update EMA weights.
#     """
#
#     def __init__(
#             self,
#             pretrained_name: str = None,
#             proj_dim: int = 128,
#             mask_ratio: float = 0.6,
#             ema_momentum: float = 0.99,
#             use_last_encoder_stage: int = -1,  # select which encoder feature to use, -1 = last
#             device: torch.device = None
#     ):
#         super().__init__()
#         config = SegformerConfig.from_pretrained(pretrained_name) if pretrained_name else SegformerConfig()
#         self.base_model = SegformerForSemanticSegmentation.from_pretrained(
#             pretrained_name, config=config, ignore_mismatched_sizes=True
#         ) if pretrained_name else SegformerForSemanticSegmentation(config=config)
#
#         # encoder returns list of feature maps; choose one stage to use
#         self.encoder = self.base_model.segformer.encoder
#         self.stage_idx = use_last_encoder_stage
#
#         # projector
#         last_hidden_dim = config.hidden_sizes[self.stage_idx] if hasattr(config, "hidden_sizes") else \
#             self.encoder.layernorms[-1].normalized_shape[0]
#         self.projector = nn.Sequential(
#             nn.Linear(last_hidden_dim, last_hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(last_hidden_dim // 2, proj_dim)
#         )
#
#         # masking params
#         self.mask_ratio = float(mask_ratio)
#
#         # momentum target encoder and projector
#         self.ema_momentum = float(ema_momentum)
#         self.target_encoder = deepcopy(self.encoder)
#         self.target_projector = deepcopy(self.projector)
#
#         # ensure target params are not trainable
#         for p in self.target_encoder.parameters():
#             p.requires_grad = False
#         for p in self.target_projector.parameters():
#             p.requires_grad = False
#
#         # device for EMA ops if specified
#         self._device = device
#
#     @torch.no_grad()
#     def update_momentum_encoder(self):
#         """
#         EMA update of target encoder + projector from online weights.
#         Use inside training loop AFTER optimizer.step().
#         """
#         m = self.ema_momentum
#         # encoder modules might be nested; iterate named_parameters for robustness
#         for tgt_param, src_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
#             tgt_param.data.mul_(m).add_(src_param.data.to(tgt_param.data.dtype) * (1.0 - m))
#
#         for tgt_param, src_param in zip(self.target_projector.parameters(), self.projector.parameters()):
#             tgt_param.data.mul_(m).add_(src_param.data.to(tgt_param.data.dtype) * (1.0 - m))
#
#     def _extract_features(self, encoder_module, x):
#         """
#         Returns features from chosen encoder stage as (B, D, H', W').
#         encoder_module is either the online encoder or the target encoder.
#         """
#         # HuggingFace segformer encoder may expect inputs directly
#         # It returns a list of feature maps; keep the chosen stage
#         outputs = encoder_module(x.float())
#         feat = outputs[self.stage_idx] if isinstance(outputs, (list, tuple)) else outputs
#         return feat  # (B, D, H', W')
#
#     def _flatten_patches(self, features):
#         B, D, H, W = features.shape
#         patches = features.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, N_patches, D)
#         return patches, H, W
#
#     def _apply_projector_and_norm(self, patches, projector):
#         # patches: (B, N, D_in)
#         B, N, D = patches.shape
#         flat = patches.reshape(B * N, D)
#         projected = projector(flat)  # (B*N, proj_dim)
#         normalized = F.normalize(projected, dim=1)
#         return normalized.reshape(B, N, -1)  # (B, N, proj_dim)
#
#     def _random_mask(self, B, N, mask_ratio, device):
#         """
#         Returns boolean mask of shape (B, N) where True indicates masked (i.e., removed from online input).
#         We'll select unmasked indices for online; target uses all patches.
#         """
#         n_mask = int(round(N * mask_ratio))
#         mask = torch.zeros((B, N), dtype=torch.bool, device=device)
#         for i in range(B):
#             perm = torch.randperm(N, device=device)
#             mask[i, perm[:n_mask]] = True
#         return mask
#
#     def forward(self, x, return_aux=False):
#         """
#         Forward returns:
#         - online_embeddings_masked: (N_masked_total, proj_dim) with masked positions excluded
#         - target_embeddings_all: (N_total, proj_dim) detached (no grad)
#         - mask (optional): boolean mask (B, N_patches) where True denotes masked patches
#         """
#         device = x.device if self._device is None else self._device
#
#         # online extractor
#         features_online = self._extract_features(self.encoder, x)  # (B, D, H', W')
#         patches_online, H, W = self._flatten_patches(features_online)  # (B, N, D)
#         online_proj = self._apply_projector_and_norm(patches_online, self.projector)  # (B, N, P)
#
#         # target extractor (no grad)
#         with torch.no_grad():
#             features_target = self._extract_features(self.target_encoder, x)
#             patches_target, _, _ = self._flatten_patches(features_target)
#             target_proj = self._apply_projector_and_norm(patches_target, self.target_projector)  # (B, N, P)
#             target_proj = target_proj.detach()
#
#         B, N, P = online_proj.shape
#
#         # build mask and select unmasked online patches
#         mask = self._random_mask(B, N, self.mask_ratio, device=device)  # True = masked
#         unmask = ~mask
#         # flatten selected online patches: (sum_unmasked, P)
#         online_selected = []
#         for i in range(B):
#             online_selected.append(online_proj[i, unmask[i, :], :])
#         online_selected = torch.cat(online_selected, dim=0)
#
#         # flatten all target patches: (B*N, P)
#         target_all = target_proj.reshape(B * N, P)
#
#         if return_aux:
#             return online_selected, target_all, mask, (H, W)
#         return online_selected, target_all
#
#     def output_dim(self):
#         return self.projector[-1].out_features
#

# -------------------------
# Feature wrapper with block / spatial masking
# -------------------------
class SegFormerFeatureWrapper(nn.Module):
    """
    Extract encoder-stage patch embeddings, project+normalize, and return:
      - online_selected: (N_online, P)
      - target_all:      (B * N_patch, P)
      - mask:            (B, N_patch) boolean (True = masked)
      - hw:              (H, W) spatial dims
      - online_to_target_idx: (N_online,) LongTensor mapping each online row
                             -> index in target_all (0 .. B*N_patch-1)
    """
    def __init__(
            self,
            pretrained_name: Optional[str] = None,
            proj_dim: int = 128,
            stage_idx: int = -1,
            mask_ratio: float = 0.1,
            block_size: int = 7,
            seed: int = 0,
    ):
        super().__init__()
        cfg = SegformerConfig.from_pretrained(pretrained_name) if pretrained_name else SegformerConfig()
        self.base_model = (
            SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name, config=cfg, ignore_mismatched_sizes=True,
                torch_dtype=torch.float
            )
            if pretrained_name
            else SegformerForSemanticSegmentation(config=cfg, torch_dtype=torch.float)
        )
        self._proj_built = False
        self.encoder = self.base_model.segformer.encoder
        encoder_sizes = self.base_model.config.hidden_sizes
        encoder_hidden_size = encoder_sizes[-1] if isinstance(encoder_sizes, (list, tuple)) else encoder_sizes
        self.stage_idx = stage_idx
        self.mask_ratio = float(mask_ratio)
        self._proj_dim = int(proj_dim)
        self.seed = int(seed)
        self._build_projector(hidden_dim=encoder_hidden_size)
        # block_size in number of patches along each spatial dim
        self.block_size = int(block_size)

    def _build_projector(self, hidden_dim: int):
        self.projector = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(hidden_dim, max(hidden_dim // 2, self._proj_dim), dtype=torch.float)),
                ("relu", nn.ReLU(inplace=True)),
                ("fc2", nn.Linear(max(hidden_dim // 2, self._proj_dim), self._proj_dim, dtype=torch.float))
            ])
        )
        self._proj_built = True

    @torch.no_grad()
    def _extract_encoder_stage(self, encoder_module, x: torch.Tensor):
        outs = encoder_module(x)
        feat = outs[self.stage_idx] if isinstance(outs, (list, tuple)) else outs
        return feat.last_hidden_state  #  # (B, D, H', W')

    def _flatten_patches(self, feat: torch.Tensor):
        B, D, H, W = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, N, D)
        return patches, H, W

    def _apply_projector_and_norm(self, patches: torch.Tensor, projector: nn.Module):
        B, N, D = patches.shape
        flat = patches.reshape(B * N, D).float()
        proj = projector(flat)  # (B*N, P)
        proj = F.normalize(proj, dim=1)
        return proj.reshape(B, N, -1)

    def _spatial_block_mask(self, B: int, H: int, W: int, mask_ratio: float, block_size: int,
                            batch_index: int, epoch: int):
        """
        Create a block/spatial mask per sample. The mask is on H x W grid of patches.
        - mask_ratio: fraction of total patches to mask
        - block_size: approximate size of each block in patches (square blocks)
        Returns boolean mask of shape (B, H*W), True = masked.
        Deterministic per epoch/batch/sample through a Torch Generator seeded from (seed, epoch, batch_index, sample_idx).
        """
        N = H * W
        mask = torch.zeros((B, N), dtype=torch.bool)
        n_mask_total = int(round(N * mask_ratio))
        # Calculate number of blocks required (ceil)
        block_area = block_size * block_size
        n_blocks = max(1, (n_mask_total + block_area - 1) // block_area)

        for i in range(B):
            g = torch.Generator()
            seed_i = (self.seed + epoch * 1_000_000 + batch_index * 10_000 + i) & 0xFFFFFFFF
            g.manual_seed(seed_i)
            # For each block, choose a top-left coordinate on the HxW grid uniformly
            chosen = torch.zeros((H, W), dtype=torch.bool)
            for _b in range(n_blocks):
                top = int(torch.randint(0, max(1, H - block_size + 1), (1,), generator=g).item())
                left = int(torch.randint(0, max(1, W - block_size + 1), (1,), generator=g).item())
                bottom = min(H, top + block_size)
                right = min(W, left + block_size)
                chosen[top:bottom, left:right] = True
                # Stop early if we've masked enough
                if chosen.sum().item() >= n_mask_total:
                    break
            flat = chosen.reshape(-1)
            # If we overshot the mask total, randomly unmask a few using the generator
            current_masked = flat.sum().item()
            if current_masked > n_mask_total:
                # indices of True positions
                true_idx = torch.nonzero(flat, as_tuple=False).flatten()
                # shuffle and keep only the first n_mask_total
                perm = true_idx[torch.randperm(true_idx.numel(), generator=g)]
                keep = perm[:n_mask_total]
                new_flat = torch.zeros_like(flat)
                new_flat[keep] = True
                flat = new_flat
            mask[i, :] = flat
        return mask  # (B, N)

    def forward(self, x: torch.Tensor, epoch: int = 0, batch_index: int = 0, return_aux: bool = False):
        # Extract online features
        features_online = self._extract_encoder_stage(self.encoder, x)   # (B, D, H, W)
        B, D, H, W = features_online.shape
        if not self._proj_built:
            self._build_projector(D)

        patches_online, H, W = self._flatten_patches(features_online)   # (B, N, D)
        online_proj = self._apply_projector_and_norm(patches_online, self.projector)  # (B, N, P)

        # 2) extract target features (same encoder module used for target wrapper instance)
        with torch.no_grad():
            features_target = self._extract_encoder_stage(self.encoder, x)
            patches_target, _, _ = self._flatten_patches(features_target)
            target_proj = self._apply_projector_and_norm(patches_target, self.projector)  # (B, N, P)
            target_proj = target_proj.detach()

        # 3) build mapping indices
        N = H * W
        # global indices for target_all: for batch i and local patch j -> idx = i*N + j
        local_indices = torch.arange(N).unsqueeze(0).expand(B, N)  # (B, N)
        global_indices = (torch.arange(B).unsqueeze(1) * N) + local_indices  # (B, N)
        global_indices = global_indices.reshape(-1)  # (B*N,)

        # 4) create spatial block mask and select unmasked online patches
        mask = self._spatial_block_mask(B, H, W, self.mask_ratio, self.block_size,
                                       batch_index=batch_index, epoch=epoch)  # (B, N)
        unmask = ~mask

        # online_selected list and mapping list
        online_selected_list = []
        online_to_target_idx_list = []
        for i in range(B):
            sel_local = torch.nonzero(unmask[i], as_tuple=False).flatten()
            if sel_local.numel() == 0:
                continue
            # gather online projected embeddings for this sample
            online_selected_list.append(online_proj[i, sel_local, :])  # (n_i, P)
            # compute global indices of selected patches
            global_sel = (i * N) + sel_local  # shape (n_i,)
            online_to_target_idx_list.append(global_sel)

        if len(online_selected_list) == 0:
            online_selected = torch.empty((0, self._proj_dim), dtype=online_proj.dtype)
            online_to_target_idx = torch.empty((0,), dtype=torch.long)
        else:
            online_selected = torch.cat(online_selected_list, dim=0)                # (N_online, P)
            online_to_target_idx = torch.cat(online_to_target_idx_list, dim=0).long()  # (N_online,)

        # 5) flatten target_all
        target_all = target_proj.reshape(B * N, -1)  # (B*N, P)

        if return_aux:
            return online_selected, target_all, mask, (H, W), online_to_target_idx
        return online_selected, target_all, online_to_target_idx

class MSNSegFormerBase(nn.Module):
    def __init__(
            self,
            pretrained_model: Optional[str] = None,
            proj_dim: int = 128,
            mask_ratio: float = 0.1,
            stage_idx: int = -1,
            block_size: int = 7,
            seed: int = 0,
            tile_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        self.tile_size = tile_size

        self.online_wrapper = SegFormerFeatureWrapper(
            pretrained_name=pretrained_model,
            proj_dim=proj_dim,
            stage_idx=stage_idx,
            mask_ratio=mask_ratio,
            block_size=block_size,
            seed=seed,
        )

        self.mask_composer = SurgicalMaskComposer()
        self.view_generator = MaskedTiledViewGenerator(self.mask_composer, self.tile_size, return_metadata=True)
        self._device = self.get_device()

    def forward(self, x: torch.Tensor, epoch: int = 0, batch_index: int = 0, return_aux: bool = False):
        return self.online_wrapper(x, epoch=epoch, batch_index=batch_index, return_aux=return_aux).float()

    def get_device(self) -> torch.device:
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
        elif next(self.buffers(), None) is not None:
            # Use buffers as a fallback
            device = next(self.buffers()).device
        else:
            # Default to CPU if neither exists (e.g., an empty Sequential block)
            device = torch.device("cpu")
        return device

    def _check_device(self):
        _device = self.get_device()
        if _device != self._device:
            self.online_wrapper.to(_device)
            self.view_generator.to(_device)
            self.mask_composer.to(_device)
            self._device = _device

    @staticmethod
    def augment(batch):
        # return batch
        batch_anchor = []
        for image in batch:
            augmented = apply_custom_augmentations(image.clone())
            batch_anchor.append(augmented)
        return torch.stack(batch_anchor)


class MoCoSiameseNetwork(MSNSegFormerBase):
    def __init__(
            self,
            pretrained_model: Optional[str],
            proj_dim: int = 128,
            mask_ratio: float = 0.1,
            block_size: int = 7,
            momentum: float = 0.99,
            stage_idx: int = -1,
            seed: int = 0,
            tile_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__(pretrained_model, proj_dim=proj_dim, mask_ratio=mask_ratio,
                         stage_idx=stage_idx, block_size=block_size, seed=seed, tile_size=tile_size)
        self.momentum = float(momentum)

        self.encoder_k = deepcopy(self.online_wrapper)
        self._set_requires_grad(self.encoder_k, False)


    @staticmethod
    def _set_requires_grad(model: nn.Module, requires_grad: bool):
        for p in model.parameters():
            p.requires_grad = requires_grad

    @torch.no_grad()
    def update_momentum_encoder(self):
        m = self.momentum
        for q, k in zip(self.online_wrapper.encoder.parameters(), self.encoder_k.encoder.parameters()):
            k.data.mul_(m).add_(q.data * (1.0 - m))
        for q, k in zip(self.online_wrapper.projector.parameters(), self.encoder_k.projector.parameters()):
            k.data.mul_(m).add_(q.data * (1.0 - m))

    def forward(self, x: torch.Tensor, epoch: int = 0, batch_index: int = 0, return_aux: bool = False):
        self._check_device()
        x_q, meta_q = self.view_generator(x)
        x_k, meta_k = self.view_generator(self.augment(x))

        online_selected, _, online_to_target_indices = self.online_wrapper(x_q, epoch=epoch, batch_index=batch_index, return_aux=return_aux)
        with torch.no_grad():
            _, target_all, _ = self.encoder_k(x_k, epoch=epoch, batch_index=batch_index, return_aux=return_aux)
            target_all = target_all.detach()

        return online_selected, target_all, online_to_target_indices

    @staticmethod
    def augment(batch):
        # return batch
        batch_anchor = []
        for image in batch:
            augmented = apply_custom_augmentations(image.clone())
            batch_anchor.append(augmented)
        return torch.stack(batch_anchor)


class SimSiamSegFormer(MSNSegFormerBase):
    """
    SimSiam using the shared SegFormerFeatureWrapper from MSNSegFormerBase.

    forward accepts either:
      - two image tensors x1, x2 (B, C, H, W) already prepared, OR
      - when used in training loop you can call _siamese_pair(batch) to create views.
    Returns:
      - p1, z2_detached, p2, z1_detached  (predictor outputs and detached targets),
    and optionally raw patches when return_patches=True.
    """

    def __init__(
            self,
            pretrained_model: Optional[str] = None,
            proj_dim: int = 128,
            predictor_hidden: int = 512,
            predictor_out: Optional[int] = None,
            stage_idx: int = -1,
            block_size: int = 7,
            mask_ratio: float = 0.6,
            seed: int = 0,
            tile_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__(
            pretrained_model=pretrained_model,
            proj_dim=proj_dim,
            mask_ratio=mask_ratio,
            stage_idx=stage_idx,
            block_size=block_size,
            seed=seed,
            tile_size=tile_size,
        )

        predictor_out = proj_dim if predictor_out is None else predictor_out
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, predictor_hidden),
            nn.BatchNorm1d(predictor_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_hidden, predictor_out),
        )

    def _pool_patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, N, D_in) raw encoder patch features before projector
        Uses the shared online_wrapper.projector, then mean-pooled and L2-normalized.
        Returns (B, proj_dim).
        """
        B, N, D = patches.shape
        flat = patches.reshape(B * N, D)
        projector = self.online_wrapper.projector
        proj_flat = projector(flat)  # (B*N, proj_dim)
        proj = proj_flat.reshape(B, N, -1)  # (B, N, proj_dim)
        img_repr = proj.mean(dim=1)  # (B, proj_dim)
        img_repr = F.normalize(img_repr, dim=1)
        return img_repr

    def _siamese_pair(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build anchor (augmented) and positive (augmented + occluded) images using base utilities.
        """
        x_anchor = self.augment(batch)  # (B, C, H, W)
        x_positive_masked, _ = self.view_generator(x_anchor)
        x_positive = x_positive_masked
        return x_anchor, x_positive

    def forward(
            self,
            x1: Optional[torch.Tensor] = None,
            x2: Optional[torch.Tensor] = None,
            *,
            batch: Optional[torch.Tensor] = None,
            return_patches: bool = False,
            epoch: int = 0,
            batch_index: int = 0,
    ):
        """
        Two modes:
          - Provide x1, x2 directly (both (B, C, H, W))
          - Or provide `batch` and let _siamese_pair build views

        Returns: (p1, z2_detached, p2, z1_detached)
        If return_patches=True also returns (patches1, patches2) where patches shape = (B, N, D_in)
        """
        if batch is not None:
            x1, x2 = self._siamese_pair(batch)
        else:
            assert x1 is not None and x2 is not None, "Provide x1/x2 or batch"

        # Extract raw encoder stage features and flatten to patches
        # Use helper methods from the wrapper to get (B, N, D_in)
        features1 = self.online_wrapper._extract_encoder_stage(self.online_wrapper.encoder, x1)
        patches1, H1, W1 = self.online_wrapper._flatten_patches(features1)

        features2 = self.online_wrapper._extract_encoder_stage(self.online_wrapper.encoder, x2)
        patches2, H2, W2 = self.online_wrapper._flatten_patches(features2)

        # Image-level projected representations
        z1 = self._pool_patches_to_image(patches1)  # (B, proj_dim)
        z2 = self._pool_patches_to_image(patches2)  # (B, proj_dim)

        # Predictor outputs (trained) - use float32 for predictor stability
        p1 = self.predictor(z1.to(dtype=torch.float32))
        p2 = self.predictor(z2.to(dtype=torch.float32))

        # Normalize predictor outputs before computing cosine similarity loss
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        # Detach targets for stop-gradient
        z1_det = z1.detach()
        z2_det = z2.detach()

        if return_patches:
            return p1, z2_det, p2, z1_det, patches1, patches2
        return p1, z2_det, p2, z1_det

    @staticmethod
    def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        p: (B, D') predicted vectors (already normalized)
        z: (B, D) target vectors (should be normalized prior to use)
        Returns mean negative cosine similarity per batch.
        """
        z = F.normalize(z, dim=1)
        return - (p * z).sum(dim=1).mean()

    def compute_loss(self, p1: torch.Tensor, z2_det: torch.Tensor, p2: torch.Tensor,
                     z1_det: torch.Tensor) -> torch.Tensor:
        """
        Symmetric SimSiam loss:
          loss = 0.5 * (neg_cos(p1, z2_det) + neg_cos(p2, z1_det))
        """
        loss1 = self.negative_cosine_similarity(p1, z2_det)
        loss2 = self.negative_cosine_similarity(p2, z1_det)
        return 0.5 * (loss1 + loss2)


class SimCLRSegFormer(MSNSegFormerBase):
    """
    SimCLR-style wrapper using the shared online_wrapper from MSNSegFormerBase.

    forward returns:
      - z_anchor: (B, proj_dim)
      - z_positive: (B, proj_dim)
    If return_patches=True, also returns:
      - patches_anchor: (B, N, D_in) raw encoder patch features before projector
      - patches_positive: (B, N, D_in)
    """

    def __init__(
            self,
            pretrained_model: Optional[str] = None,
            proj_dim: int = 128,
            mask_ratio: float = 0.6,
            stage_idx: int = -1,
            block_size: int = 7,
            seed: int = 0,
            tile_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__(
            pretrained_model=pretrained_model,
            proj_dim=proj_dim,
            mask_ratio=mask_ratio,
            stage_idx=stage_idx,
            block_size=block_size,
            seed=seed,
            tile_size=tile_size,
        )

    def _pool_patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, N, D_in) raw encoder patch features before projector
        Returns: (B, proj_dim) projected and L2-normalized pooled vectors
        """
        B, N, D = patches.shape
        flat = patches.reshape(B * N, D)
        projector = self.online_wrapper.projector
        proj_flat = projector(flat)  # (B*N, proj_dim)
        proj = proj_flat.reshape(B, N, -1)  # (B, N, proj_dim)
        img_repr = proj.mean(dim=1)  # (B, proj_dim)
        img_repr = F.normalize(img_repr, dim=1)  # L2-normalize per-image
        return img_repr

    def _siamese_pair(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build anchor and positive masked views using base augment + view_generator.
        Anchor is augmented (unmasked). Positive is augmented and occluded by view_generator.
        """
        x_anchor = self.augment(batch)  # (B, C, H, W)
        x_positive_masked, meta_positive = self.view_generator(x_anchor)
        # If view_generator returns masked tiled input directly, use it as positive.
        x_positive = x_positive_masked
        return x_anchor, x_positive

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> Tuple:
        """
        Returns:
          (z_anchor, z_positive) if return_patches=False
          (z_anchor, z_positive, patches_anchor, patches_positive) if return_patches=True
        z_... shapes: (B, proj_dim)
        patches_... shapes: (B, N, D_in)
        """
        x_anchor, x_positive = self._siamese_pair(x)

        # Extract raw encoder stage features (B, D, H', W') then flatten to (B, N, D)
        features_anchor = self.online_wrapper._extract_encoder_stage(self.online_wrapper.encoder, x_anchor)
        patches_anchor, H_a, W_a = self.online_wrapper._flatten_patches(features_anchor)

        features_positive = self.online_wrapper._extract_encoder_stage(self.online_wrapper.encoder, x_positive)
        patches_positive, H_p, W_p = self.online_wrapper._flatten_patches(features_positive)

        # Project, mean-pool and normalize to image-level vectors
        z_anchor = self._pool_patches_to_image(patches_anchor)
        z_positive = self._pool_patches_to_image(patches_positive)

        if return_patches:
            return z_anchor, z_positive, patches_anchor, patches_positive
        return z_anchor, z_positive



class SupervisedSegformerSegmentation(nn.Module):
    def __init__(self, pretrained_model: str = None, num_classes: int = None,
                 checkpoint:str=None,
                 k:int=3):
        super().__init__()

        # Load the full SegformerForSemanticSegmentation model.
        # Set `ignore_mismatched_sizes=True` because we will replace the
        # final classification layer, which will have a different output size.


        msn_model = SegFormerFeatureWrapper(pretrained_name=pretrained_model)

        try:
            msn_model.load_state_dict(torch.load(checkpoint,
                                             map_location=next(msn_model.parameters()).device,
                                             weights_only=False))


        except Exception as e:
            raise ValueError(f"Could not load MSN pretrained weights from {checkpoint}")

        self.base_model = msn_model.segformer

        # Get the number of channels from the previous layer to properly
        # define the input to our new classifier.
        classifier_in_channels = self.base_model.decode_head.linear_fuse.out_channels

        # Replace the original classifier with a custom Sequential module.
        self.base_model.decode_head.classifier = nn.Sequential(
            # First convolution layer to process the features.
            nn.Conv2d(classifier_in_channels, 256, kernel_size=3, padding=1),
            # Batch normalization for training stability.
            nn.BatchNorm2d(256),
            # ReLU activation for non-linearity.
            nn.ReLU(inplace=True),
            # Final convolution to map features to the desired number of classes.
            nn.Conv2d(256, self.num_classes, kernel_size=1)
        )
        self.median = MedianPool2d(kernel_size=k, padding=k // 2)

        # --- Checkpoint Loading Logic ---
        if self.checkpoint_path:
            try:
                # Load the state dictionary from the .pt file
                state_dict = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
                self.base_model.load_state_dict(state_dict)
            except FileNotFoundError:
                # Raise exception for consistent error handling
                raise SegformerModelError(f"Checkpoint file not found at: {self.checkpoint_path}")
            except Exception as e:
                # Catch any other loading errors
                raise SegformerModelError(f"Failed to load checkpoint: {e}")

    def forward(self, pixel_values: torch.FloatTensor, labels: torch.LongTensor = None):
        """
        Forward pass for the custom Segformer model.

        Args:
            pixel_values (torch.Tensor): Input tensor of pixel values.
            labels (torch.Tensor, optional): Optional ground truth labels.

        Returns:
            torch.Tensor: The output logits from the model, upsampled to the original input size.
        """
        # The base model's forward pass handles the entire encoder and decoder.
        # We only need the logits.
        output = self.base_model(pixel_values=pixel_values.float()).logits

        # The Segformer model's output logits are at a reduced resolution (e.g., 1/4th).
        # We upsample them back to the original input size.
        logits = F.interpolate(output,
                               size=pixel_values.shape[2:],
                               mode='bilinear',
                               align_corners=False)

        # return logits
        return self.median(logits)   # Smoothed logits


if __name__ == "__main__":
    b, c, h, w = 8, 3, 512, 512
    tile_size = (32, 32)
    instrument_prob, fluid_prob, fold_prob = 0.3, 0.3, 0.3

    composer = SurgicalMaskComposer(instrument_prob=instrument_prob,
                                    fluid_prob=fluid_prob,
                                    fold_prob=fold_prob)

    gen = MaskedTiledViewGenerator(mask_composer=SurgicalMaskComposer(),
                                   tile_size=tile_size,
                                   return_metadata=True)

    test_images = torch.randn(b, c, h, w).clip(0, 1)

    tiles = gen.tile_image(test_images)
    print(f"Image shape: {test_images.shape}, Generated tile shape {tiles.shape}")

    masked_tiles, metadata = gen.apply_masks(tiles)

    print(f"After apply_masks(), mask_tiles: {masked_tiles.shape}, metadata: {len(metadata)}")

    masked_tiles = masked_tiles.permute(0, 2, 1, 3, 4)
    masked_image = gen.stitch_tiles(masked_tiles, h, w)

    print(f'After stitch_tiles(), masked_image.shape: {masked_image.shape}')

    mask = gen(test_images)
    print(mask[0].shape)  # The generated mask
    # print(mask[1])          # The generated mask metadata as Dict{'type': ..., 'params': ....}
