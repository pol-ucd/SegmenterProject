import sys

import torch
from torch import nn as nn, autocast
from torch.nn import functional as F
from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput # Import for consistent output

from torchinfo import summary   # Required for testing only

from utils.torch_utils import get_default_device, get_default_device_type


"""
Implementation from Word document

    A custom neural network built on top of SegformerForSemanticSegmentation
    for binary semantic segmentation.

    The original final classification layer of Segformer's decode_head is replaced
    with a custom sequence: Conv2d -> BatchNorm2d -> ReLU -> Conv2d (output 1 channel).
"""
class SegformerBinarySegmentation(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = 'nvidia/segformer-b4-finetuned-ade-512-512'):
        """
        Initialise the custom Segformer model.
        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained
                                                 Segformer model to load from Hugging Face.
        """
        super().__init__()
        self.backbone = SegformerModel.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        hidden_dim = self.backbone.config.hidden_sizes[-1]

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
        features = self.backbone(pixel_values=pixel_values).last_hidden_state  # [B, C, H/32, W/32]
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
class SegformerBinarySegmentation2(nn.Module):
    default_model = 'nvidia/segformer-b4-finetuned-ade-512-512'
    def __init__(self, pretrained_model: str = None):
        """
        Initialise the custom Segformer model.
        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained
                                                 Segformer model to load from Hugging Face.
        """
        super().__init__()

        # `ignore_mismatched_sizes=True`
        self.pretrained_model = pretrained_model or SegformerBinarySegmentation2.default_model

        self.config = SegformerConfig.from_pretrained(self.pretrained_model)


        # self.config.num_labels = 1  # Set for binary classification output
        self.config.num_labels = 1 # if we pass the mask then it will be a 0/1 output
        # self.config.id2label = {"0": "healthy", "1": "lesion"}
        # self.config.label2id = {"healthy": 0, "lesion": 1}
        # self.config.output_hidden_states=False

        self.hidden_dim = int(self.config.decoder_hidden_size)

        # `ignore_mismatched_sizes=True` because we are changing the
        # effective output channels of the final classifier, even if we replace it.
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained_model,
            config=self.config,
            ignore_mismatched_sizes=True
        )

        self.encoder = self.base_model.segformer
        self.decoder = self.base_model.decode_head
        self.classifier = self.base_model.decode_head.classifier

        in_channels = int(self.base_model.decode_head.classifier.in_channels)

        self.custom_classification_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=1,
                      kernel_size=1)  # Final output is 1 channel for binary
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
        # encoder_outputs = self.encoder(pixel_values=pixel_values).last_hidden_state
        # print(f"encoder_outputs shape: {encoder_outputs.shape}")
        # logits = self.classifier(encoder_outputs)
        # decoder_outputs = self.decoder(encoder_outputs)
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
    # summary(model2)

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
