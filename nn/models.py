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
    def __init__(self, pretrained_model: str = None, num_classes: int = None):
        super().__init__()
        self.pretrained_model = pretrained_model or SegformerBinaryClassifierBase.default_model

        self.config = SegformerConfig.from_pretrained(self.pretrained_model)

        self.num_classes = num_classes or 1 # Default is a binary classifier

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
                 pretrained_model: str = None, num_classes: int = None):
        """
        Initialise the custom Segformer model.
        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained
                                                 Segformer model to load from Hugging Face.
        """
        super().__init__(pretrained_model, num_classes)

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
                                                   out_channels=self.num_classes,
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

    def __init__(self, pretrained_model: str = None, num_classes: int = None):
        super().__init__(pretrained_model, num_classes)

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
            nn.Conv2d(256, self.num_classes, kernel_size=1)
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
        output = self.base_model(pixel_values=pixel_values.float(), labels=labels)

        logits = F.interpolate(output.logits,
                               size=pixel_values.shape[2:],
                               mode='bilinear',
                               align_corners=False)

        return logits



class CustomSegformerDecodeHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_ch in in_channels:
            # We process each feature map individually to a uniform channel dimension
            self.convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(len(in_channels) * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, features):
        # 'features' is a list of tensors from the encoder
        # We process each feature map to the same channel dimension
        c1, c2, c3, c4 = features

        # Upsample each feature map to the size of the largest one (c4)
        c1 = F.interpolate(self.convs[0](c1), size=c4.shape[2:], mode="bilinear", align_corners=False)
        c2 = F.interpolate(self.convs[1](c2), size=c4.shape[2:], mode="bilinear", align_corners=False)
        c3 = F.interpolate(self.convs[2](c3), size=c4.shape[2:], mode="bilinear", align_corners=False)
        c4 = self.convs[3](c4)  # c4 is already the target size

        # Concatenate the processed and upsampled feature maps
        concatenated_features = torch.cat([c1, c2, c3, c4], dim=1)

        # Apply the final fusion convolution
        fused_features = self.fuse_conv(concatenated_features)

        # Apply the final classifier to get the single-class logits
        logits = self.classifier(fused_features)

        # Upsample the logits to the original input size
        return F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)



"""
Implementation merging the description in Word doc with code from finalcode.py.

    A custom neural network built on top of SegformerForSemanticSegmentation
    for binary semantic segmentation.

    The entire decode layer of Segformer's decode_head is replaced
    with a custom sequence: Conv2d -> BatchNorm2d -> ReLU -> Conv2d (output 1 channel).
"""
class SegformerBinarySegmentation3(SegformerBinaryClassifierBase):

    def __init__(self, pretrained_model: str = None, num_classes: int = None):
        super().__init__(pretrained_model, num_classes)

        self.hidden_dim = int(self.config.decoder_hidden_size)

        # Ignore_mismatched_sizes=True` because we are changing the
        # effective output channels of the final classifier, even if we replace it.
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained_model,
            config=self.config,
            ignore_mismatched_sizes=True
        )

        # Define the input channels from the backbone (these are specific to the "b4" model)
        in_channels = [64, 128, 320, 512]
        # Define the new custom decode head
        custom_decode_head = CustomSegformerDecodeHead(
            in_channels=in_channels,
            out_channels=768,  # Same as the original model
            num_classes=self.num_classes,  # Single class output
        )

        # Replace the original decode_head with the custom one
        self.base_model.decode_head = custom_decode_head

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
        output = self.base_model(pixel_values=pixel_values.float(), labels=labels)

        logits = F.interpolate(output.logits,
                               size=pixel_values.shape[2:],
                               mode='bilinear',
                               align_corners=False)

        return logits

"""
Implementation merging the description in Word doc with code from finalcode.py.

    A custom neural network built on top of SegformerForSemanticSegmentation
    for binary semantic segmentation.

    The outputs from SegformerForSemanticSegmentation are feed to 
    a custom classification layer: Conv2d -> BatchNorm2d -> ReLU -> Conv2d (output 1 channel).
"""

class SegformerBinarySegmentation4(SegformerBinaryClassifierBase):
    """
    A wrapper class for SegformerForSemanticSegmentation that replaces its
    decode_head with a custom Sequential layer for multi-class classification.
    """

    def __init__(self, pretrained_model: str, num_classes: int = 1):
        """
        Initializes the Segformer model with a custom classification head.

        Args:
            model_id (str): The ID of the pre-trained Segformer model to load
                            (e.g., "nvidia/segformer-b4-finetuned-ade-512-512").
            num_classes (int): The number of output classes for semantic segmentation.
        """
        super().__init__(pretrained_model, num_classes)

        in_channels_for_custom_head = self.config.num_labels
        # Ignore_mismatched_sizes=True` because we are changing the
        # effective output channels of the final classifier, even if we replace it.
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(self.pretrained_model,
                                                                           config=self.config,
                                                                           ignore_mismatched_sizes=True)
        # Define the custom decode head as a PyTorch Sequential layer.
        # This head takes the fused features from the Segformer encoder
        # and processes them for multi-class classification.
        self.classifier = nn.Sequential(
            # First Conv2d layer: maintains channel dimension, applies 3x3 convolution.
            nn.Conv2d(in_channels=in_channels_for_custom_head,
                      out_channels=in_channels_for_custom_head,
                      kernel_size=3,
                      padding=1),
            # BatchNorm2d: normalizes activations, improving training stability.
            nn.BatchNorm2d(in_channels_for_custom_head),
            # ReLU activation: introduces non-linearity.
            nn.ReLU(),
            # Final Conv2d layer: maps features to the desired number of output classes.
            # Kernel size 1x1 is common for classification layers.
            nn.Conv2d(in_channels=in_channels_for_custom_head,
                      out_channels=num_classes,
                      kernel_size=self.num_classes),
        )

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor = None):
        """
        Performs a forward pass through the Segformer model with the custom head.

        Args:
            pixel_values (torch.Tensor): The input image tensor (batch_size, channels, height, width).
            labels (torch.Tensor, optional): The ground truth labels for loss calculation. Defaults to None.

        Returns:
            transformers.modeling_outputs.SemanticSegmenterOutput: An object containing
            the model's output logits and potentially the loss if labels are provided.
        """
        # The `SegformerForSemanticSegmentation`'s forward method internally
        # handles the encoder output and passes it to the `decode_head`.
        # By replacing `decode_head`, our custom layer will be used automatically.
        outputs = self.base_model(pixel_values=pixel_values).logits
        outputs = self.classifier(outputs)
        outputs = F.interpolate(outputs,
                                size=pixel_values.shape[2:],
                                mode='bilinear', align_corners=False)
        return outputs


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

    model3 = SegformerBinarySegmentation3().to(device)

    # Perform a forward pass with autocast
    print("3. Performing forward pass with dummy input and labels using autocast...")
    with torch.autocast(device_type=device_type, dtype=torch.float16):
        # When labels are provided, the model will also compute the loss.
        output3 = model3(pixel_values=dummy_input, labels=dummy_labels)

    # Print the shape of the output logits
    # Expected shape: (batch_size, num_labels=1, height, width)
    print(f"Output logits shape: {output3.shape}")

    # Verify output properties
    if output3.shape == torch.Size([1, 1, 512, 512]):
        print("Output shape is as expected for binary semantic segmentation.")
    else:
        print("Output shape is NOT as expected. Please check the implementation.")

    print("\n3. Example complete.")

    model4 = SegformerBinarySegmentation3().to(device)

    # Perform a forward pass with autocast
    print("4. Performing forward pass with dummy input and labels using autocast...")
    with torch.autocast(device_type=device_type, dtype=torch.float16):
        # When labels are provided, the model will also compute the loss.
        output4 = model4(pixel_values=dummy_input, labels=dummy_labels)

    # Print the shape of the output logits
    # Expected shape: (batch_size, num_labels=1, height, width)
    print(f"Output logits shape: {output4.shape}")

    # Verify output properties
    if output4.shape == torch.Size([1, 1, 512, 512]):
        print("Output shape is as expected for binary semantic segmentation.")
    else:
        print("Output shape is NOT as expected. Please check the implementation.")

    print("\n4. Example complete.")

