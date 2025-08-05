import sys

import torch
from torch import nn as nn, autocast
from torch.nn import functional as F
from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput # Import for consistent output

from torchinfo import summary   # Required for testing only

from utils.torch_utils import get_default_device, get_default_device_type

from abc import abstractmethod

"""
Base class for the Segformer models
"""
class SegformerBinaryClassifierBase(nn.Module):
    default_model = 'nvidia/segformer-b4-finetuned-ade-512-512'
    def __init__(self, pretrained_model: str = None):
        super().__init__()
        self.pretrained_model = pretrained_model or SegformerBinaryClassifierBase.default_model

        self.config = SegformerConfig.from_pretrained(self.pretrained_model)
        self.config.num_labels = 1  # Binary classifier

        self.base_model = None

    @abstractmethod
    def forward(self, pixel_values):
        pass

"""
Implementation from Word document - deprecated I expect!!

    A custom neural network built on top of SegformerForSemanticSegmentation
    for binary semantic segmentation.

    The original final classification layer of Segformer's decode_head is replaced
    with a custom sequence: Conv2d -> BatchNorm2d -> ReLU -> Conv2d (output 1 channel).
"""
class SegformerBinarySegmentation(SegformerBinaryClassifierBase):
    def __init__(self,
                 pretrained_model: str = None):
        """
        Initialise the custom Segformer model.
        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained
                                                 Segformer model to load from Hugging Face.
        """
        super().__init__(pretrained_model)
        self.base_model = SegformerModel.from_pretrained(self.pretrained_model,
                                                         config=self.config,
                                                         ignore_mismatched_sizes=True)
        
        hidden_dim = self.base_model.config.hidden_sizes[-1]

        self.decode_head = nn.Sequential(nn.Conv2d(hidden_dim,
                                                   out_channels=256,
                                                   kernel_size=3,
                                                   padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(in_channels=256,
                                                   out_channels=1,
                                                   kernel_size=1)
                                         )

    def forward(self, pixel_values):
        features = self.base_model(pixel_values=pixel_values).last_hidden_state  # [B, C, H/32, W/32]
        logits = self.decode_head(features)  # [B, 1, H/32, W/32]
        logits = F.interpolate(logits,
                               size=pixel_values.shape[2:],
                               mode='bilinear',
                               align_corners=False)
        return logits.float()  # [B, 1, 512, 512]

"""
Implementation from finalcode.py.

    A custom neural network built on top of SegformerForSemanticSegmentation
    for binary semantic segmentation.

    The original final classification layer of Segformer's decode_head is replaced
    with a custom sequence: Conv2d -> BatchNorm2d -> ReLU -> Conv2d (output 1 channel).
"""
class SegformerBinarySegmentation2(SegformerBinaryClassifierBase):

    def __init__(self, pretrained_model: str = None):
        super().__init__(pretrained_model)

        self.hidden_dim = int(self.config.decoder_hidden_size)

        # Ignore_mismatched_sizes=True` because we are changing the
        # effective output channels of the final classifier, even if we replace it.
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained_model,
            config=self.config,
            ignore_mismatched_sizes=True
        )

        classifier_in_size = self.base_model.decode_head.linear_fuse.out_channels
        self.base_model.decode_head.classifier = nn.Sequential(
            nn.Conv2d(classifier_in_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )


    def forward(self, pixel_values: torch.FloatTensor, labels: torch.LongTensor = None):
        """
        Forward pass for the custom Segformer model.

        Args:
            pixel_values (torch.Tensor): Input tensor of pixel values, typically
                                         of shape (batch_size, num_channels, height, width).
            labels (torch.Tensor, optional): Optional ground truth labels for loss computation.
                                             Defaults to None.

        Returns:
            transformers.modeling_outputs.SemanticSegmentationModelOutput:
                An object containing the model's outputs. The `logits` attribute
                will contain the output of our custom binary classification layer.
        """
        output = self.base_model(pixel_values=pixel_values, labels=labels)

        logits = F.interpolate(output.logits,
                               size=pixel_values.shape[2:],
                               mode='bilinear',
                               align_corners=False)

        return logits

if __name__ == '__main__':
    device = get_default_device()
    device_type = get_default_device_type()

    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    print(f"\nDummy input tensor shape: {dummy_input.shape}")

    # Create dummy labels for loss calculation (binary mask)
    dummy_labels = torch.randint(0, 2, (1, 512, 512)).long().to(device)  # Binary labels 0 or 1
    print(f"Dummy labels tensor shape: {dummy_labels.shape}")

    model1 = SegformerBinarySegmentation().to(device)
    # summary(model1)

    # Perform a forward pass with autocast
    print("1. Performing forward pass with dummy input and labels using autocast...")
    with torch.autocast(device_type=device_type, dtype=torch.float16):
        # When labels are provided, the model will also compute the loss.
        output1 = model1(pixel_values=dummy_input)

    # Print the shape of the output logits
    # Expected shape: (batch_size, num_labels=1, height, width)
    print(f"Output logits shape: {output1.shape}")


    # Verify output properties
    if output1.shape == torch.Size([1, 1, 512, 512]):
        print("Output shape is as expected for binary semantic segmentation.")
        print("\n1. Test Passed!!.")
    else:
        print("Output shape is NOT as expected. Please check the implementation.")
        print("\n1. Test Failed!!.")



    model2 = SegformerBinarySegmentation2().to(device)

    # Perform a forward pass with autocast
    print("2. Performing forward pass with dummy input and labels using autocast...")
    with torch.autocast(device_type=device_type, dtype=torch.float16):
        # When labels are provided, the model will also compute the loss.
        output2 = model2(pixel_values=dummy_input, labels=dummy_labels)


    # Print the shape of the output logits
    # Expected shape: (batch_size, num_labels=1, height, width)
    print(f"Output logits shape: {output2.shape}")


    # Verify output properties
    if output2.shape == torch.Size([1, 1, 512, 512]):
        print("Output shape is as expected for binary semantic segmentation.")
    else:
        print("Output shape is NOT as expected. Please check the implementation.")

    print("\n2. Example complete.")
