from abc import abstractmethod

import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


# A custom exception class to handle errors specific to a Segformer-based model.
class SegformerModelError(Exception):
    """
    Custom exception for errors related to the Segformer model or its
    base classes, such as invalid model configurations or unexpected
    behavior during loading.
    """
    def __init__(self, message="An error occurred with the Segformer model."):
        self.message = message
        super().__init__(self.message)



"""
Base class for the Segformer models
"""
class SegformerBinaryClassifierBase(nn.Module):
    default_model = 'nvidia/segformer-b4-finetuned-ade-512-512'

    def __init__(self, pretrained_model: str = None, num_classes: int = None,
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
        self.config = SegformerConfig.from_pretrained(self.pretrained_model)
        self.base_model = None
        if checkpoint_path:
            self.load_model(checkpoint_path)

    @abstractmethod
    def forward(self, pixel_values):
        """Abstract method for the forward pass."""
        pass


class CustomSegformerDecodeHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_ch in in_channels:
            self.convs.append(nn.Conv2d(in_ch, out_channels, kernel_size=1))
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(len(in_channels) * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        c1 = F.interpolate(self.convs[0](c1), size=c4.shape[2:], mode="bilinear", align_corners=False)
        c2 = F.interpolate(self.convs[1](c2), size=c4.shape[2:], mode="bilinear", align_corners=False)
        c3 = F.interpolate(self.convs[2](c3), size=c4.shape[2:], mode="bilinear", align_corners=False)
        c4 = self.convs[3](c4)
        concatenated_features = torch.cat([c1, c2, c3, c4], dim=1)
        fused_features = self.fuse_conv(concatenated_features)
        logits = self.classifier(fused_features)
        return F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)


"""
A single-class implementation for semantic segmentation.
This class is the recommended approach for its correctness and efficiency.
"""
class SegformerBinarySegmentation(SegformerBinaryClassifierBase):
    def __init__(self, pretrained_model: str = None, num_classes: int = None,
                 checkpoint_path:str=None):
        super().__init__(pretrained_model, num_classes, checkpoint_path)

        # Load the full SegformerForSemanticSegmentation model.
        # Set `ignore_mismatched_sizes=True` because we will replace the
        # final classification layer, which will have a different output size.
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained_model,
            config=self.config,
            ignore_mismatched_sizes=True
        )

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

        return logits



# class SegformerBinarySegmentation2(SegformerBinarySegmentation):
#     pass
#
#
# class SegformerBinarySegmentation3(SegformerBinarySegmentation):
#     pass
#
#
# class SegformerBinarySegmentation4(SegformerBinarySegmentation):
#     pass
