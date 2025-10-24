from abc import abstractmethod

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from transformers import SegformerConfig, SegformerModel, SegformerForSemanticSegmentation

class MedianPool2d(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.k = _pair(kernel_size)
        self.s = _pair(stride)
        self.p = _pair(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = F.unfold(x,
                           kernel_size=self.k,
                           stride=self.s,
                           padding=self.p)
        # → (B, C, k*k, L)
        patches = patches.view(B, C, self.k[0]*self.k[1], -1)

        # median over window → (B, C, L)
        med = patches.median(dim=2)[0]

        # fold back → (B, C, H_out, W_out)
        H_out = (H + 2*self.p[0] - self.k[0]) // self.s[0] + 1
        W_out = (W + 2*self.p[1] - self.k[1]) // self.s[1] + 1
        return med.view(B, C, H_out, W_out)


class SegformerModelError(Exception):
    """
    Custom exception for errors related to the Segformer model or its
    base classes, such as invalid model configurations or unexpected
    behavior during loading.
    """
    def __init__(self, message="An error occurred with the Segformer model."):
        self.message = message
        super().__init__(self.message)


class AugurSegmenterBase(nn.Module):
    default_model = 'nvidia/segformer-b4-finetuned-ade-512-512'

    def __init__(self, /, pretrained_model: str = None, num_classes: int = None,
                 checkpoint_path: str = None):
        """
        Initializes the base class for Segformer-based binary classifiers.

        Args:
            pretrained_model (str): The name or path of the pretrained
                                    Segformer model.
            num_classes (int): The number of output classes.
            checkpoint_path (str): The path of the checkpoint to be loaded.
        """
        super().__init__()
        self.pretrained_model = pretrained_model
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes or 1
        if self.pretrained_model is not None:
            try:
                self.config = SegformerConfig.from_pretrained(self.pretrained_model)
            except OSError:
                self.config = SegformerConfig()
        else:
            self.config = SegformerConfig()

        self.base_model = None
        if checkpoint_path:
            self.load_model(checkpoint_path)

    @abstractmethod
    def forward(self, pixel_values):
        """Abstract method for the forward pass."""
        pass


class SegformerBackbone(nn.Module):
    """
    A wrapper for the Hugging Face SegFormer model to be used as a backbone.
    This extracts the final hidden state (feature map).
    """

    def __init__(self, model_name='nvidia/segformer-b0-finetuned-ade-512-512', output_dim=256):
        super().__init__()
        self.segformer = SegformerModel.from_pretrained(model_name)
        self.in_features = int(self.segformer.config.hidden_sizes[-1])
        self.projection = nn.Linear(self.in_features, output_dim)

    def forward(self, x):
        outputs = self.segformer(pixel_values=x)
        features = outputs.last_hidden_state
        B, C, H, W = features.shape

        # Reshape for the nn.Linear layer: (B, C, H, W) -> (B, H*W, C)
        features = features.flatten(2).transpose(1, 2)

        # Apply the projection: (B, H*W, C) -> (B, H*W, output_dim)
        features = self.projection(features)

        # Reshape back to a 4D feature map: (B, H*W, output_dim) -> (B, output_dim, H, W)
        features = features.transpose(1, 2).reshape(B, -1, H, W)

        return features


class SupervisedSegFormer(SegformerForSemanticSegmentation):
    """
    Standard SegFormer model for semantic segmentation.
    Inherits from the Hugging Face implementation.
    """

    def __init__(self, config):
        super().__init__(config)

    def load_pretrain_weights(self, pretrain_state_dict: dict):
        """
        Loads the pre-trained encoder weights into the segmentation model's backbone.
        """
        # Get the keys for the shared encoder from the pre-trained state dict
        encoder_state_dict = {
            k: v for k, v in pretrain_state_dict.items()
            if k.startswith('encoder')
        }

        # Load the weights into the current model, ignoring the randomly initialized head
        # The strict=False handles keys missing in the decoder part
        self.load_state_dict(encoder_state_dict, strict=False)
