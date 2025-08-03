import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput # Import for consistent output

from torchinfo import summary   # Required for testing only

"""
Implmentation from Hanija's Word document

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
Implementation from Hanija's new Python code.

    A custom neural network built on top of SegformerForSemanticSegmentation
    for binary semantic segmentation.

    The original final classification layer of Segformer's decode_head is replaced
    with a custom sequence: Conv2d -> BatchNorm2d -> ReLU -> Conv2d (output 1 channel).
"""
class SegformerBinarySegmentation2(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = 'nvidia/segformer-b4-finetuned-ade-512-512'):
        """
        Initialise the custom Segformer model.
        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained
                                                 Segformer model to load from Hugging Face.
        """
        super().__init__()

        # `ignore_mismatched_sizes=True` handles the weight mismatch
        # if the original pretrained model had a different number of labels.
        self.config = SegformerConfig.from_pretrained(pretrained_model_name_or_path)
        self.config.num_labels = 1  # Set for binary classification output
        self.hidden_dim = self.config.decoder_hidden_size

        # `ignore_mismatched_sizes=True` because we are changing the
        # effective output channels of the final classifier, even if we replace it.
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path,
            config=self.config,
            ignore_mismatched_sizes=True
        )

        # We need `in_channels` to know the dimension of the features
        # coming out of the Segformer's decoder before its final classification.
        # For 'nvidia/segformer-b4-finetuned-ade-512-512', this is typically 768.
        in_channels_for_custom_head = self.segformer_model.decode_head.classifier.in_channels

        self.custom_classification_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dim,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=1,
                      kernel_size=1)  # Final output is 1 channel for binary
        )

        # We will NOT replace self.segformer_model.decode_head.classifier directly here.
        # Instead, we will manually call the decode_head's components in the forward pass
        # and then apply our custom layer. This gives us more control over contiguity.

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor = None):
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
        # The `segformer` object's forward method will now use our
        # `custom_classification_layer` because we replaced its internal `classifier`.
        print("\n1.", pixel_values.shape)
        encoder_outputs = self.segformer_model.segformer(pixel_values=pixel_values)
        hidden_states = encoder_outputs.hidden_states
        print("\n2.", encoder_outputs.shape)

        # Manually process hidden states through the decode_head's components
        # This part mimics the internal logic of SegformerDecodeHead's forward method
        # AND allows us to insert .contiguous() where needed.
        decode_head = self.segformer_model.decode_head
        height, width = pixel_values.shape[-2:]

        # Project features from different scales
        # This part processes the multi-scale features from the encoder
        # through linear layers (MLPs) as done in SegFormer's decode head.
        all_hidden_states = ()
        for i, hidden_state in enumerate(hidden_states):
            # Ensure contiguity after potential non-contiguous operations
            # if the encoder or previous layers caused it.
            # This is a common place for view errors if not handled.
            hidden_state = hidden_state.contiguous()
            projected_hidden_state = decode_head.project_features[i](hidden_state)
            all_hidden_states += (projected_hidden_state,)

        # Fuse features from different scales
        # This combines the multi-scale features by upsampling and concatenating them.
        # The `fuse_layers` typically handles this.
        # The output of fuse_layers is the aggregated feature map before the final classifier.
        fused_features = decode_head.fuse_layers(all_hidden_states)

        # Ensure the fused features are contiguous before passing to our custom layer.
        # This is the critical step to prevent the "view size is not compatible" error.
        fused_features = fused_features.contiguous()

        # 3. Apply the custom binary classification layer
        # This replaces the original decode_head.classifier
        logits = self.custom_classification_layer(fused_features)

        # 4. Upsample logits to original input size if necessary
        # The decode_head usually outputs at H/4 x W/4. We need to upsample to original size.
        if logits.shape[-2:] != (height, width):
            logits = nn.functional.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False
            )
        # 5. Prepare the output in the format expected by transformers
        loss = None
        if labels is not None:
            # For binary segmentation, typically use BCEWithLogitsLoss
            # Ensure labels match logits shape for loss calculation
            if labels.ndim == 3:  # If labels are (batch_size, H, W)
                labels = labels.unsqueeze(1).float()  # Make (batch_size, 1, H, W)
            elif labels.ndim == 4 and labels.shape[1] != 1:
                # If labels are (batch_size, C, H, W) but C is not 1, might need conversion
                # Assuming binary, so maybe labels are (B, 1, H, W) or (B, H, W)
                pass

            # Ensure labels are float for BCEWithLogitsLoss
            labels = labels.float()

            # Resize labels to match logits if necessary (e.g., if logits are downsampled)
            # However, in this setup, we upsample logits to original size before returning.
            # So, labels should match the original input size.
            if labels.shape[-2:] != logits.shape[-2:]:
                labels = nn.functional.interpolate(
                    labels,
                    size=logits.shape[-2:],
                    mode="nearest"  # Use nearest for labels to preserve integrity
                )

            # BCEWithLogitsLoss combines sigmoid and BCELoss
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits
        )

if __name__ == '__main__':
    model = SegformerBinarySegmentation()
    summary(model)
    print("+"*100)
    model = SegformerBinarySegmentation2()
    summary(model)