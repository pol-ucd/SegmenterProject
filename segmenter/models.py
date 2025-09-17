from abc import abstractmethod

import torch
from monai.networks.nets import UNet, SwinUNETR
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
# Import the SegFormer model from the transformers library
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class MedianPool2d(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.k = _pair(kernel_size)
        self.s = _pair(stride)
        self.p = _pair(padding)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # unfold → (B, C*k*k, L)
        patches = F.unfold(x, kernel_size=self.k,
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


class SegmentationModelWrapper(nn.Module):
    """
    A smart wrapper class to handle different medical image segmentation models
    transparently, encapsulating their unique initialization and ensuring a
    consistent interface for training and benchmarking.
    """

    def __init__(self, model_name: str, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.model_name = model_name.lower()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs):
        if self.model_name == 'nnunet':
            self.model = UNet(
                spatial_dims=2,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2
            )
        elif self.model_name == 'transunet':
            # Use MONAI's SwinUNETR as an alternative since it's a similar transformer-based UNet.
            self.model = SwinUNETR(
                img_size=(kwargs.get('img_dim', 224), kwargs.get('img_dim', 224)),
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                feature_size=48
            )
        elif self.model_name == 'segformer':
            # Instantiate SegFormer using the transformers library.
            # We use a pre-trained model and adjust the number of output labels.
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512",
                    num_labels=self.out_channels,
                    ignore_mismatched_sizes=True
                )
            except OSError:
                print("Could not load pre-trained model, initializing from config.")
                config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
                config.num_labels = self.out_channels
                self.model = SegformerForSemanticSegmentation(config)
        elif self.model_name == 'hrnet_cbam':
            raise NotImplementedError("HRNet+CBAM implementation needs a dedicated library or custom code.")
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SegFormer from Hugging Face expects a dictionary input
        if self.model_name == 'segformer':
            # SegFormer's output is a dictionary. We only need the logits.
            return self.model(pixel_values=x).logits
        return self.model(x)

    def get_model(self):
        return self.model

    def __repr__(self):
        return f"SegmentationModelWrapper(model_name='{self.model_name}', model={self.model.__class__.__name__})"
"""
Base class for the Segformer models
"""
class AugurSegformerClassifierBase(nn.Module):
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
        if self.pretrained_model is not None:
            self.config = SegformerConfig.from_pretrained(self.pretrained_model)
        else:
            self.config = SegformerConfig()
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
class AugurSegformerSegmentation(AugurSegformerClassifierBase):
    def __init__(self, pretrained_model: str = None, num_classes: int = None,
                 checkpoint_path:str=None,
                 k:int=3):
        super().__init__(pretrained_model, num_classes, checkpoint_path)

        # Load the full SegformerForSemanticSegmentation model.
        # Set `ignore_mismatched_sizes=True` because we will replace the
        # final classification layer, which will have a different output size.
        if self.pretrained_model is not None:
            self.base_model = SegformerForSemanticSegmentation.from_pretrained(
                self.pretrained_model,
                config=self.config,
                ignore_mismatched_sizes=True
            )
        else:
            self.base_model = SegformerForSemanticSegmentation(config=self.config)

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


# Assume you have your data loaders and loss function
# from your_dataset import MedicalImageDataset

# Step 1: Instantiate the models consistently using the wrapper
# model_nnunet = SegmentationModelWrapper(model_name='nnunet', in_channels=1, out_channels=2)
# model_transunet = SegmentationModelWrapper(model_name='transunet', in_channels=1, out_channels=2, img_dim=256)
# model_segformer = SegmentationModelWrapper(model_name='segformer', in_channels=1, out_channels=2)
# model_hrnet_cbam = SegmentationModelWrapper(model_name='hrnet_cbam', in_channels=1, out_channels=2)

# For demonstration, let's use the nnU-Net example
model = SegmentationModelWrapper(model_name='nnunet', in_channels=1, out_channels=2)

# Step 2: Set up the training loop
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

# # Dummy data for illustration
# input_tensor = torch.randn(4, 1, 224, 224)
# target_tensor = torch.randint(0, 2, (4, 1, 224, 224)).float()

# # Step 3: Use the wrapper in a unified way
# output = model(input_tensor)
# loss = loss_function(output, target_tensor)
# loss.backward()
# optimizer.step()
# print(f"Loss with {model.model_name}: {loss.item()}")