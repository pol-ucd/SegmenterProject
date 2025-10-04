"""
Mixed Siamese Networks for Bring Your Own Labels (BYOL) implementation

    The follwoing architectural styles are implemented:
    - Momentum Contrast (MoCo): using two encoders and updating one with EMA,
    - SimCLR: Gradients flow from both views to the single shared encoder
    - SimSiam: detached gradient on one branch of a shared encoder's output

    Pretrain a backbone model with unlabelled images using one of the
    MSN architectures below.

    Fine-tune the pretrained model with a small subset of annotated examples

    Validate with the remaining annotated images
"""
import copy
import random
from copy import deepcopy
from typing import Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import SegformerModel


class MaskedTiledViewGenerator:
    def __init__(self, mask_composer, tile_size=(64, 64), return_metadata=False):
        self.mask_composer = mask_composer  # e.g., SurgicalMaskComposer
        self.tile_size = tile_size
        self.return_metadata = return_metadata

    def tile_image(self, image):
        B, C, H, W = image.shape
        th, tw = self.tile_size
        assert H % th == 0 and W % tw == 0, "Image must be divisible by tile size"
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
        tiles_per_row = W // tw
        tiles = masked_tiles.reshape(B, tiles_per_row, -1, C, th, tw)
        rows = [torch.cat([tiles[b, r] for r in range(tiles_per_row)], dim=2)
                for b in range(B)]
        stitched = torch.cat(rows, dim=2)  # (B, C, H, W)
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


class SurgicalMaskComposer:
    def __init__(self, instrument_prob=0.3, fluid_prob=0.3, fold_prob=0.4):
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
        occlusion[:, y:y+h, x:x+w] = 1.0
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
        mask = ((xx - cx)**2 + (yy - cy)**2) < radius**2
        fold[:, mask] = 0.0
        masked = tile * fold
        return masked, {'cx': cx, 'cy': cy, 'radius': radius}


class SegFormerAdapter(nn.Module):
    """
    Calls a pretrained SegFormer encoder and returns a (B, C, H, W) feature map.
    Adjust token->spatial conversion to match the specific SegFormer variant you use.
    """
    def __init__(self, pretrained_name, out_stride=256, target_resolution=None):
        super().__init__()
        # load backbone; use SegformerModel for encoder-only outputs
        self.backbone = SegformerModel.from_pretrained(pretrained_name)
        # If using SegformerForSemanticSegmentation you may access .encoder or .last_hidden_state similarly
        self.target_resolution = target_resolution  # (H, W) or None; used to upsample features
        self.out_stride = out_stride  # e.g., 4, 8, 16 — depends on the variant

    def forward(self, x):
        # x: (B, 3, H, W) in pixel space
        # Many SegFormer variants expect pixel values normalized; ensure preprocessing matches pretrained model
        outputs = self.backbone(pixel_values=x)  # returns BaseModelOutput or dict
        # Typical key: last_hidden_state -> (B, seq_len, hidden_dim)
        last_hidden = outputs.last_hidden_state  # shape (B, N, C)
        B, N, C = last_hidden.shape

        # Convert sequence tokens back to spatial layout:
        # If SegFormer uses non-overlapping patches, seq_len = H_patch * W_patch.
        # Compute patch grid size:
        patch_h = patch_w = int(N ** 0.5)
        if patch_h * patch_w != N:
            # fall back to known stride logic: derive from input size and model stride, or raise informative error
            # Here we assume square patch layout; adjust for your variant if different.
            raise RuntimeError("Unexpected token layout: cannot reshape sequence length to square grid")

        feat = last_hidden.permute(0, 2, 1).contiguous()    # (B, C, N)
        feat = feat.view(B, C, patch_h, patch_w)            # (B, C, H_patch, W_patch)

        # Optionally upsample to the requested resolution (H, W)
        if self.target_resolution is not None:
            feat = F.interpolate(feat, size=self.target_resolution, mode="bilinear", align_corners=False)

        return feat  # (B, C, H_out, W_out)


class MoCoSegFormer(nn.Module):
    """
    Siamese Network Architecture Pattern: Momentum Contrast (MoCo)

    utilizes a core concept seen in MoCo and BYOL: two separate, yet related, encoders:

    Online Encoder (self.online_encoder):
            This is the main encoder whose weights are updated directly
            via standard backpropagation gradients.

T   arget/Momentum Encoder (self.target_encoder):
            This is a copy of the online encoder, but its weights
            are not updated via backpropagation.

    Provides a slowly evolving target, decoupling the two views in time/weight space.

    """
    def __init__(self, backbone, momentum=0.996):
        super().__init__()
        self.momentum = momentum

        # Create online and target networks
        self.online_encoder = backbone
        self.target_encoder = copy.deepcopy(self.online_encoder)

        # Disable gradients for the target network
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self):
        """
        Performs the Exponential Moving Average (EMA) update for the target network.
        This is a key component of self-supervised methods like MoCo, BYOL, and MSN.
        The update rule is: θ_t = m * θ_t + (1 - m) * θ_o
        """
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = target_param.data * self.momentum + online_param.data * (1. - self.momentum)

    def forward(self, focal_view, global_view):
        # Get online predictions from the focal view (strongly augmented)
        # Reshape to (B, C, H*W) -> (B, H*W, C) for masking

        online_features = self.online_encoder(focal_view).flatten(2).transpose(1, 2)

        # Select only the features from the MASKED patches
        # mask shape: (B, num_patches), online_features shape: (B, num_patches, C)
        masked_online_features = online_features #[mask.reshape(online_features.shape)]

        # Get target representations from the global view (weakly augmented)
        with torch.no_grad():
            self.target_encoder.eval()
            target_features = self.target_encoder(global_view).flatten(2).transpose(1, 2)
            # Detach to ensure no gradients flow back to the target encoder
            target_features = target_features.detach()

        return masked_online_features, target_features

    def get_model_stride(self):
        return self.online_encoder.segformer.config.strides[-1]


class SimSiamSegFormer(nn.Module):
    """
    SegFormer Encoder wrapped in a SimSiam architecture for pre-training.
    Uses a Projection Head and a Predictor Head.
    Processes both views symmetrically and uses .detach() on the target branch
    to stop the gradient flow.
    """

    def __init__(self, model_name: str = 'nvidia/mit-b0', projection_dim: int = 128):
        super().__init__()
        # Load the SegFormer Model (only the encoder/backbone) - Shared Encoder
        self.online_encoder = SegformerModel.from_pretrained(model_name)

        encoder_output_dim = self.online_encoder.config.hidden_sizes[-1]

        # Projection Head (g) - Maps features to embedding space (z)
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.BatchNorm1d(encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim)
        )

        # Predictor Head (h) - Maps the embedding (z) to a prediction (p)
        self.predictor_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 4),
            nn.BatchNorm1d(projection_dim // 4),
            nn.ReLU(),
            nn.Linear(projection_dim // 4, projection_dim)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes two augmented views (x1, x2) symmetrically.

        :param x1: View 1 (e.g., Anchor Image).
        :param x2: View 2 (e.g., Positive/Occluded Image).
        :return: (p1, z2_detached, p2, z1_detached)
        """

        # Helper function to compute z and p for a view
        def get_p_and_z(x):
            # Encoder
            features = self.online_encoder(x).last_hidden_state
            # Pool features over the sequence dimension (dim=1) and explicitly reshape to [B, D]
            z_pooled = features.mean(dim=1).reshape(features.shape[0], -1)
            # Projection Head (z)
            z = self.projection_head(z_pooled)
            # Predictor Head (p)
            p = self.predictor_head(z)
            return p, z

        # Process View 1 (Anchor)
        p1, z1 = get_p_and_z(x1.squeeze(1))

        # Process View 2 (Positive)
        p2, z2 = get_p_and_z(x2.squeeze(1))

        # Symmetrical output for loss calculation:
        return p1, z2.detach(), p2, z1.detach()


class SimCLRSegFormer(nn.Module):
    """
    An implementation of SimCLR architecture using a single, shared encoder for both the anchor
    and positive image streams.

    The shared encoder approach is standard in simpler Contrastive Learning frameworks like SimCLR.
    Invariance Learning: The core goal is to teach the single encoder to produce similar embeddings
    (z_anchor, and z_positive) for two different, augmented views (the original image and the occluded image)
    of the same underlying scene. By using a single set of weights, you maximize the gradient flow,
    forcing those weights to learn invariance to the occlusion augmentation simultaneously from both paths.

    Simultaneous Update: The loss is calculated based on the similarity between the positive pair,
    and dissimilarity with all other pairs in the batch (negatives). When you backpropagate the loss,
    the gradients from both the anchor view and the positive (occluded) view update the same shared
    weights in the encoder at the same time.

    There is no concept of a "momentum" encoder or separate updates.
    """

    def __init__(self, model_name: str = 'nvidia/mit-b0', projection_dim: int = 128):
        super().__init__()
        self.online_encoder = SegformerModel.from_pretrained(model_name)

        # For MiT-B0, the last feature dimension is typically 512
        encoder_output_dim = self.online_encoder.config.hidden_sizes[-1]

        # Projection Head (MLP) for Contrastive Learning
        # This maps the high-dimensional feature into a lower-dimensional embedding (z)
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim)
        )

    def forward(self, x_anchor: torch.Tensor, x_positive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes anchor and positive pairs through the shared encoder and head.
        :return: (Anchor Embeddings, Positive Embeddings)
        """
        # Anchor Stream
        # [B, H*W, D] -> We need to pool or average it to [B, D]
        anchor_features = self.online_encoder(x_anchor.squeeze()).last_hidden_state

        # Global Average Pooling across the spatial dimension (H*W)
        # [B, H*W, D] -> [B, D]
        z_anchor_pooled = anchor_features.mean(dim=1).reshape(anchor_features.shape[0], -1)

        # Positive Stream
        positive_features = self.online_encoder(x_positive.squeeze()).last_hidden_state
        # z_positive_pooled = positive_features.mean(dim=1)
        z_positive_pooled = positive_features.mean(dim=1).reshape(positive_features.shape[0], -1)

        # Projection Head
        z_anchor = self.projection_head(z_anchor_pooled)
        z_positive = self.projection_head(z_positive_pooled)

        return z_anchor, z_positive


class MoCoSiameseNetwork(nn.Module):
    """
    Implements a MoCo/BYOL/MSN-style architecture with Online and Target Encoders,
    now including the logic to mask online features.
    """

    def __init__(self, pretrained_model, projection_dim=128,
                 tile_size=(64, 64), temperature=0.2, momentum=0.999):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        self.tile_size = tile_size
        self.projection_dim = projection_dim

        # Create online and target networks
        # NOTE: SegformerModel needs an input of shape [B, C, H, W] when passing `pixel_values`
        # self.online_encoder = SegformerModel.from_pretrained(pretrained_model)
        self.online_encoder = SegFormerAdapter(pretrained_model, target_resolution=target_resolution)
        # after adapter we have channels = backbone.config.hidden_size
        encoder_output_dim = self.online_encoder.backbone.config.hidden_sizes[-1] if hasattr(self.online_encoder.backbone.config,
                                                                               "hidden_sizes") else self.online_encoder.backbone.config.hidden_size

        # encoder_output_dim = self.online_encoder.config.hidden_sizes[-1]

        # Online Predictor Head (h)
        self.online_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim // 4),
            nn.BatchNorm1d(encoder_output_dim // 4),
            nn.ReLU(),
            nn.Linear(encoder_output_dim // 4, self.projection_dim)
        )

        self.encoder_q = nn.Sequential(self.online_encoder,
                                       self.online_head)
        self.encoder_k = nn.Sequential(deepcopy(self.online_encoder),
                                       deepcopy(self.online_head))
        self._init_momentum_encoder()


        self.mask_composer = SurgicalMaskComposer()
        self.view_generator = MaskedTiledViewGenerator(self.mask_composer,
                                                       self.tile_size,
                                                       return_metadata=True)

        # self.register_buffer("queue", torch.randn(12800,
        #                                           self.projection_dim))
        # self.queue = F.normalize(self.queue, dim=1)
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _init_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """
        Performs the Exponential Moving Average (EMA) update for both the target encoder and target head.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, x):
        # Generate masked views
        x_q, meta_q = self.view_generator(x)
        x_k, meta_k = self.view_generator(x)

        # Encode query
        q = self.encoder_q(x_q)  # (B, D)
        q = F.normalize(q, dim=1)

        # Encode key (no grad)
        with torch.no_grad():
            self._update_momentum_encoder()
            k = self.encoder_k(x_k)
            k = F.normalize(k, dim=1)

        # Compute contrastive loss
        # logits = torch.mm(q, self.queue.T) / self.temperature
        # labels = torch.arange(q.size(0), device=q.device)
        # loss = F.cross_entropy(logits, labels)

        # Update queue
        # self._dequeue_and_enqueue(k)

        # return loss, {'meta_q': meta_q, 'meta_k': meta_k}
        return q, k

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[ptr:ptr + batch_size] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.queue.shape[0]

    # def forward(self, online_view: torch.Tensor, target_view: torch.Tensor, mask: torch.Tensor):
    #     """
    #     Processes focal_view (online/masked) and global_view (target/unmasked).
    #
    #     :param online_view: The anchor/online view [B, C, H, W].
    #     :param target_view: The positive/target view [B, C, H, W].
    #     :param mask: The synthetic binary mask [B, 1, H, W]. (1=occlusion area)
    #     :return: (P, Z') prediction and detached target embedding.
    #     """
    #     B, C, H, W = online_view.shape
    #
    #     # --- Online Path (Focal View / MASKED) ---
    #     # NOTE: SegFormer expects the input to be named 'pixel_values'
    #     online_output = self.online_encoder(pixel_values=online_view.squeeze(1))
    #     online_features = online_output.last_hidden_state  # [B, S, D]
    #
    #     S, D = online_features.shape[1], online_features.shape[2]
    #
    #     # --- FIX 1: Robustly calculate spatial dimensions of the feature grid ---
    #     # Ensure torch.round() is applied to a Tensor, not a float.
    #     s_tensor = torch.tensor(S, dtype=torch.float32, device=online_features.device)
    #     h_feat = int(torch.round(torch.sqrt(s_tensor)).item())
    #     w_feat = h_feat  # Assume square grid for interpolation
    #
    #     # 1. Downsample the input mask [B, 1, H, W] to feature resolution [B, 1, h_feat, w_feat]
    #     downsampled_mask = F.interpolate(
    #         mask.float(),
    #         size=(h_feat, w_feat),
    #         mode='nearest'
    #     )  # [B, 1, h_feat, w_feat]
    #
    #     # Create the boolean index: True for UNMASKED (visible) patches.
    #     patch_visibility_mask_flat = (1.0 - downsampled_mask).flatten(1).bool()  # [B, h_feat*w_feat]
    #
    #     # 3. Apply the mask and pool: For each sample, select only the visible patches
    #     online_pooled_features_list = []
    #     for i in range(B):
    #
    #         # --- FIX 2: Ensure mask length exactly matches feature sequence length S ---
    #         mask_i = patch_visibility_mask_flat[i]
    #         mask_len = mask_i.shape[0]
    #
    #         if mask_len != S:
    #             # The spatial approximation h_feat x w_feat resulted in an array
    #             # that is too long or too short. We force it to length S.
    #             if mask_len > S:
    #                 # Truncate (less likely, but safer)
    #                 current_mask = mask_i[:S]
    #             else:
    #                 # Pad with False (not visible) if mask is too short (most likely scenario)
    #                 padding = torch.zeros(S - mask_len, dtype=torch.bool, device=online_view.device)
    #                 current_mask = torch.cat([mask_i, padding])
    #         else:
    #             current_mask = mask_i
    #
    #         # Indexing: online_features[i] is [S, D]. current_mask is [S]. This is the correct operation.
    #         visible_patches = online_features[i][current_mask]  # [S_visible, D]
    #
    #         # Pool over the visible patches only (average over S_visible)
    #         if visible_patches.numel() > 0:
    #             online_pooled_features = visible_patches.mean(dim=0)  # [D]
    #         else:
    #             # Fallback: if completely masked, use the full feature mean
    #             online_pooled_features = online_features[i].mean(dim=0)  # [D]
    #
    #         online_pooled_features_list.append(online_pooled_features)
    #
    #     online_pooled_features = torch.stack(online_pooled_features_list)  # [B, D]
    #
    #     # Apply the predictor head to get the final prediction P
    #     prediction_p = self.online_head(online_pooled_features)
    #
    #     # --- Target Path (Global View / UNMASKED) ---
    #     with torch.no_grad():
    #         self.target_encoder.eval()
    #
    #         # Helper function logic moved inline for clarity
    #         target_output = self.target_encoder(pixel_values=target_view.squeeze(1))
    #         target_features = target_output.last_hidden_state  # [B, S, D]
    #
    #         # Pool over the sequence dimension (dim=1) for the target features
    #         target_pooled_features = target_features.mean(dim=1).reshape(target_features.shape[0], -1)
    #
    #         # Apply the Target Head to project
    #         target_z = self.target_head(target_pooled_features)
    #
    #         # The target embedding Z' must be detached
    #         target_z_detached = target_z.detach()
    #
    #     # Returns P (prediction) and Z' (detached target embedding)
    #     return prediction_p, target_z_detached
    #
    # def get_model_stride(self):
    #     # Approximation for the effective stride of the final feature map
    #     return self.online_encoder.config.patch_sizes[-1]
