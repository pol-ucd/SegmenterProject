import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerModel

from segmenter.core import freeze_seed
from segmenter.utils import SurgicalAugmentor
from segmenter.utils.surgical import SurgicalSiameseDatasetHDF5, SurgicalMaskComposer


#
# def generate_random_mask(input_size, model_stride, mask_ratio):
#     """
#     Generates a random mask based on the model's output feature grid.
#     """
#     B = input_size[0]
#     H, W = input_size[2], input_size[3]
#
#     # Calculate feature map dimensions using the model's actual stride
#     num_patches_h = H // model_stride
#     num_patches_w = W // model_stride
#     num_patches = num_patches_h * num_patches_w
#
#     num_masked = int(mask_ratio * num_patches)
#
#     mask_indices = torch.rand(B, num_patches).argsort(dim=-1)[:, :num_masked]
#     mask = torch.zeros(B, num_patches, dtype=torch.bool)
#     mask.scatter_(1, mask_indices, True)
#
#     return mask
#


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
# The Masked Siamese Network (MSN)
################################################################################

class SurgicalMaskedSiameseNetwork(nn.Module):
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


################################################################################
# MSN Loss Function
################################################################################

def msn_loss(online_preds, target_protos, temperature=0.1):
    """
    Calculates the MSN loss.
    - online_preds: Predictions from the online network for MASKED patches.
    - target_protos: Representations from the target network for ALL patches.
    - temperature: Controls the sharpness of the target distribution.
    """
    # Normalize the prototypes and predictions
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




################################################################################
# Main Training Script
################################################################################

if __name__ == '__main__':
    DATASET = '/Users/polmacaonghusa/Documents/Projects/SegmenterProject/data/all_images.hdf5'
    SHAPE = (512, 512)
    BATCH_SIZE = 8
    IMG_SIZE = 224
    FOCAL_IMG_SIZE = 96
    # MODEL_PATCH_SIZE = 16  # SegFormer-B0 uses patch size 4, but effective stride is larger. We use 16 for feature map.
    MASK_RATIO = 0.5
    FEATURE_DIM = 256  # Should match the backbone's output_dim
    LEARNING_RATE = 1e-4
    EPOCHS = 10

    # For reproducibility
    freeze_seed()

    # Create a dummy dataset (replace with your endoscopic image dataset) ---
    print("Loading dataset of surgical images...")

    # Instantiate dataset and augmentations
    mask_composer = SurgicalMaskComposer(shape=SHAPE, channels=3)
    augmentations = SurgicalAugmentor(size=SHAPE)
    full_dataset = SurgicalSiameseDatasetHDF5(hdf5_path=DATASET,
                                              mask_composer=mask_composer,
                                              augmentor=augmentations)

    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=None)

    print(f"Dataloader created with {len(full_dataset)} data records in {len(dataloader)} batches.")

    print("Initializing SegFormer backbone and MSN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segformer_backbone = SegformerBackbone(output_dim=FEATURE_DIM)
    msn_model = SurgicalMaskedSiameseNetwork(backbone=segformer_backbone).to(device)

    optimizer = torch.optim.AdamW(msn_model.online_encoder.parameters(), lr=LEARNING_RATE)

    print(f"Starting self-supervised training for {EPOCHS} epochs on {device}...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for i_batch, data_view in enumerate(dataloader):
            global_views = data_view['view1'].to(device)
            focal_views = data_view['view2'].to(device)


            # Forward pass
            online_preds, target_protos = msn_model(focal_views, global_views) #, mask)

            # The target prototypes are from the entire batch, so we need to concatenate them
            # B, N, C -> (B*N), C
            target_protos = target_protos.reshape(-1, FEATURE_DIM)

            # Calculate loss
            loss = msn_loss(online_preds, target_protos)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target network using EMA
            msn_model._update_target_network()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Average Loss: {avg_loss:.4f}")

    print("Self-supervised pre-training complete!")
    print("You can now save the `msn_model.online_encoder` state dict for fine-tuning.")

    # Example of saving the backbone for downstream tasks
    # torch.save(msn_model.online_encoder.state_dict(), 'segformer_msn_pretrained_backbone.pth')
