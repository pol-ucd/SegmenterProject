import copy
from typing import Tuple

import torch
from torch import nn as nn
from transformers import SegformerModel


class SimCLRSiameseNetwork(nn.Module):
    """ Original model - used in surgical_mask_msn_segformer.py """
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
        anchor_features = self.online_encoder(x_anchor).last_hidden_state

        # Global Average Pooling across the spatial dimension (H*W)
        # [B, H*W, D] -> [B, D]
        z_anchor_pooled = anchor_features.mean(dim=1).reshape(anchor_features.shape[0], -1)

        # Positive Stream
        positive_features = self.online_encoder(x_positive).last_hidden_state
        # z_positive_pooled = positive_features.mean(dim=1)
        z_positive_pooled = positive_features.mean(dim=1).reshape(positive_features.shape[0], -1)

        # Projection Head
        z_anchor = self.projection_head(z_anchor_pooled)
        z_positive = self.projection_head(z_positive_pooled)

        return z_anchor, z_positive
