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
from typing import Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import SegformerModel


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

    def __init__(self, model_name, momentum=0.996, projection_dim=128):
        super().__init__()
        self.momentum = momentum
        self.projection_dim = projection_dim

        # Create online and target networks
        self.online_encoder = SegformerModel.from_pretrained(model_name)

        try:
            encoder_output_dim = self.online_encoder.config.hidden_sizes[-1]
        except:
            encoder_output_dim = 256

        # Online Predictor Head (h)
        self.online_head = nn.Sequential(
            # This must match the encoder's pooled feature dimension (D)
            nn.Linear(encoder_output_dim, encoder_output_dim // 4),
            nn.BatchNorm1d(encoder_output_dim // 4),
            nn.ReLU(),
            nn.Linear(encoder_output_dim // 4, self.projection_dim)
        )

        # Target Head (Z') - Projector only. Must match the online head structure.
        self.target_head = copy.deepcopy(self.online_head)

        self.target_encoder = copy.deepcopy(self.online_encoder)

        # Disable gradients for the target network and target head
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self):
        """
        Performs the Exponential Moving Average (EMA) update for both the target encoder and target head.
        """
        # Update Target Encoder
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = target_param.data * self.momentum + online_param.data * (1. - self.momentum)

        # Update Target Head (The projection layer used for the target features)
        for online_param, target_param in zip(self.online_head.parameters(), self.target_head.parameters()):
            target_param.data = target_param.data * self.momentum + online_param.data * (1. - self.momentum)

    def forward(self, focal_view: torch.Tensor, global_view: torch.Tensor, mask: torch.Tensor):
        """
        Processes focal_view (online/masked) and global_view (target/unmasked).

        :param focal_view: The anchor/online view [B, C, H, W].
        :param global_view: The positive/target view [B, C, H, W].
        :param mask: The synthetic binary mask [B, 1, H, W]. (1=occlusion area)
        :return: (P, Z') prediction and detached target embedding.
        """
        B, _, H, W = focal_view.shape

        # Helper to extract and pool features (used only for Target Network)
        def get_pooled_features(x, encoder):
            features = encoder(x.squeeze(1)).last_hidden_state
            # Pool over the sequence dimension (dim=1) and reshape to [B, D]
            return features.mean(dim=1).reshape(features.shape[0], -1).squeeze(-1)

            # --- Online Path (Focal View / MASKED) ---

        online_output = self.online_encoder(focal_view.squeeze(1))
        online_features = online_output.last_hidden_state  # [B, S, D]

        S, D = online_features.shape[1], online_features.shape[2]
        h_feat = w_feat = int(torch.sqrt(torch.tensor(S).float()).item())

        # 1. Downsample the input mask [B, 1, H, W] to feature resolution [B, 1, h_feat, w_feat]
        downsampled_mask = F.interpolate(
            mask.float(),
            size=(h_feat, w_feat),
            mode='nearest'
        )  # [B, 1, h_feat, w_feat]

        # Create the boolean index: True for UNMASKED (visible) patches
        # FIX: Use flatten(1) to get shape [B, S], necessary for indexing.
        patch_visibility_mask = (1.0 - downsampled_mask).flatten(1).bool()  # [B, S]

        # 3. Apply the mask and pool: For each sample, select only the visible patches
        online_pooled_features_list = []
        for i in range(B):
            # Select the D-dim feature vectors where the patch is visible
            # Indexing [S, D] with a 1D boolean mask [S] is the correct way to select rows.
            print(online_features[i].shape, downsampled_mask.shape, patch_visibility_mask[i].size)
            visible_patches = online_features[i][patch_visibility_mask[i]]  # [S_visible, D]

            # Pool over the visible patches only (average over S_visible)
            if visible_patches.numel() > 0:
                online_pooled_features = visible_patches.mean(dim=0)  # [D]
            else:
                # Fallback: if completely masked, use the full feature mean
                online_pooled_features = online_features[i].mean(dim=0)  # [D]

            online_pooled_features_list.append(online_pooled_features)

        online_pooled_features = torch.stack(online_pooled_features_list)  # [B, D]

        # Apply the predictor head to get the final prediction P
        prediction_p = self.online_head(online_pooled_features.flatten(-2, -1))

        # --- Target Path (Global View / UNMASKED) ---
        with torch.no_grad():
            self.target_encoder.eval()
            target_pooled_features = get_pooled_features(global_view, self.target_encoder)

            # Apply the Target Head to project 256 down to 128 (MUST be done before comparison)
            target_z = self.target_head(target_pooled_features)

            # The target embedding Z' must be detached
            target_z_detached = target_z.detach()

        # Returns P (prediction) and Z' (detached target embedding)
        return prediction_p, target_z_detached

    def get_model_stride(self):
        # NOTE: This line assumes a 'segformer' attribute exists in the backbone,
        # which is true for HuggingFace's SegformerModel, but we should make sure
        # to use the config object if possible.
        return self.online_encoder.config.patch_sizes[-1]  # Approximation, usually 16 or 32 for the final layer
