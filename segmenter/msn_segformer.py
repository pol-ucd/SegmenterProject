import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SegformerModel

from segmenter.core import freeze_seed, randimage


################################################################################
# Data Augmentation and Masking Helper
################################################################################

class DataAugmentationMSN:
    """
    Creates two views of an image for MSN: a global view (weak augmentations)
    and a focal view (strong augmentations).
    """

    def __init__(self, global_crop_size=224, focal_crop_size=96):
        # Weak augmentations for the global view (target)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=(0.5, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Strong augmentations for the focal view (online)
        self.focal_transform = transforms.Compose([
            transforms.RandomResizedCrop(focal_crop_size, scale=(0.2, 0.5),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        global_view = self.global_transform(image)
        focal_view = self.focal_transform(image)
        return global_view, focal_view


def generate_random_mask(input_size, model_stride, mask_ratio):
    """
    Generates a random mask based on the model's output feature grid.
    """
    B = input_size[0]
    H, W = input_size[2], input_size[3]

    # Calculate feature map dimensions using the model's actual stride
    num_patches_h = H // model_stride
    num_patches_w = W // model_stride
    num_patches = num_patches_h * num_patches_w

    num_masked = int(mask_ratio * num_patches)

    mask_indices = torch.rand(B, num_patches).argsort(dim=-1)[:, :num_masked]
    mask = torch.zeros(B, num_patches, dtype=torch.bool)
    mask.scatter_(1, mask_indices, True)

    return mask


class DummyEndoscopyDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(256, 256)):
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
        ])
        # Create random tensors that vaguely resemble images
        # self.data = torch.randn(num_samples, 3, img_size, img_size).uniform_(0, 1)
        data_shape = (num_samples, 3) + img_size
        self.data = randimage(data_shape).float()/255

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.transform(self.data[idx])


################################################################################
# MODULE 2: SegFormer Backbone Wrapper
################################################################################

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


################################################################################
# MODULE 3: The Masked Siamese Network (MSN)
################################################################################

class MaskedSiameseNetwork(nn.Module):
    def __init__(self, backbone, feature_dim=256, momentum=0.996):
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

    def forward(self, focal_view, global_view, mask):
        # Get online predictions from the focal view (strongly augmented)
        # Reshape to (B, C, H*W) -> (B, H*W, C) for masking

        online_features = self.online_encoder(focal_view).flatten(2).transpose(1, 2)

        # Select only the features from the MASKED patches
        # mask shape: (B, num_patches), online_features shape: (B, num_patches, C)
        masked_online_features = online_features[mask.reshape(online_features.shape)]

        # 3. Get target representations from the global view (weakly augmented)
        with torch.no_grad():
            self.target_encoder.eval()
            target_features = self.target_encoder(global_view).flatten(2).transpose(1, 2)
            # Detach to ensure no gradients flow back to the target encoder
            target_features = target_features.detach()

        return masked_online_features, target_features

    def get_model_stride(self):
        return self.online_encoder.segformer.config.strides[-1]


################################################################################
# MODULE 4: MSN Loss Function
################################################################################

def msn_loss(online_preds, target_protos, temperature=0.1):
    """
    Calculates the MSN loss.
    - online_preds: Predictions from the online network for MASKED patches.
    - target_protos: Representations from the target network for ALL patches.
    - temperature: Controls the sharpness of the target distribution.
    """
    # Normalize the prototypes and predictions
    print(online_preds.shape, target_protos.shape)
    online_preds = F.normalize(online_preds, dim=1)
    target_protos = F.normalize(target_protos, dim=1)

    # Calculate similarity scores between each masked patch and all target patches
    # Shape: (num_masked_patches, num_target_patches)
    similarity_matrix = torch.matmul(online_preds, target_protos.t())

    # Sharpen the target distribution and compute the loss
    # The target is the softmax over similarities with the target prototypes
    targets = F.softmax(similarity_matrix / temperature, dim=1)

    # The prediction is the log-softmax over the same similarities
    predictions = F.log_softmax(similarity_matrix, dim=1)

    # Cross-entropy loss
    loss = - (targets * predictions).sum(dim=1)
    return loss.mean()

# # Custom collate function to apply augmentations
# def collate_fn(batch):
#     global_views, focal_views = zip(*[augmentations(img) for img in batch])
#     return torch.stack(global_views), torch.stack(focal_views)

