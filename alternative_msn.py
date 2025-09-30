from datetime import datetime
import logging
import os
import random
import sys
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
# from timm.data.tf_preprocessing import IMAGE_SIZE
from torch.utils.data import ConcatDataset

from transformers import SegformerConfig, SegformerModel, SegformerForSemanticSegmentation

from segmenter.utils import HDF5Dataset
from segmenter.utils.data import get_num_samples_from_hdf5

PRETRAIN_DATASETS = ['../segmenter/data/dresden_preprocessed.h5',
                     '../segmenter/data/all_data.h5']

FINETUNE_DATASETS = ['../segmenter/data/Classica.h5']

IMAGE_SIZE=(512, 512)


class MSNPretrainDatasetHDF5(HDF5Dataset):
    """ A dataset containing raw medical images (for pre-training)
       and annotated images/masks (for fine-tuning).
    """

    def __init__(self, hdf5_path,
                 size: Tuple[int, int] = (512, 512)):
        super().__init__(hdf5_path)
        self.image_size = size

    def __getitem__(self, idx):
        # if self.f is None:
        #     # Re-open the file handle for this specific worker/process
        #     self.f = h5py.File(self.hdf5_path, 'r')
        _data = super().__getitem__(idx)

        # Load image and convert to tensor
        image =_data['images']

        image_augment = T.Compose([T.ToTensor(),
                                   T.Resize(self.image_size,
                                            T.InterpolationMode.BICUBIC),
                                   T.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                                   ])

        image = image_augment(image)

        return image


class MSNFinetuneDatasetHDF5(HDF5Dataset):
    """ A dataset containing raw medical images (for pre-training)
       and annotated images/masks (for fine-tuning).
    """

    def __init__(self, hdf5_path,
                 indices,
                 size: Tuple[int, int] = (512, 512)):
        super().__init__(hdf5_path)
        self.indices = indices
        self.image_size = size
        self.len = len(indices)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.f is None:
            # Re-open the file handle for this specific worker/process
            self.f = h5py.File(self.hdf5_path, 'r')

        # Load image and convert to tensor
        image = self.f['images'][self.indices[idx]]
        mask = self.f['masks'][self.indices[idx]]

        image_augment = T.Compose([T.ToTensor(),
                                   T.Resize(self.image_size,
                                            T.InterpolationMode.BICUBIC),
                                   T.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                                   ])

        mask_augment = T.Compose([T.ToTensor(),
                                  T.Resize(self.image_size,
                                           T.InterpolationMode.BICUBIC)
                                  ])
        # Apply augmentations
        image = image_augment(image)
        mask = mask_augment(mask)
        return image, mask.long()




# --- 1. Data Generation Module (Simulation) ---

class MaskGenerator:
    """
    Generates synthetic masks to simulate surgical occlusions, instruments,
    and tissue folds for self-supervised pre-training.
    """

    def __init__(self, size: int = 512, instrument_ratio: float = 0.3):
        """
        Initializes the mask generator.
        :param size: The target size (H, W) for the square mask.
        :param instrument_ratio: Probability of generating an instrument-like mask.
        """
        self.size = size if len(size) > 1 else (size, size)
        self.instrument_ratio = instrument_ratio

    def _generate_circle_mask(self) -> np.ndarray:
        """Generates a soft circular/elliptical mask (random lesion/fold approx)."""
        mask = np.zeros(self.size, dtype=np.float32)

        # Random parameters
        center_x = random.randint(self.size[0] // 4, self.size[0] * 3 // 4)
        center_y = random.randint(self.size[1] // 4, self.size[1] * 3 // 4)
        radius_x = random.randint(self.size[0] // 10, self.size[0] // 4)
        radius_y = random.randint(self.size[1] // 10, self.size[1] // 4)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Ellipse equation: (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
                if ((i - center_y) ** 2 / radius_y ** 2 + (j - center_x) ** 2 / radius_x ** 2) <= 1:
                    mask[i, j] = 1.0

        # Apply slight blurring (approximates tissue folds/irregularity)
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=random.uniform(3, 8))
        mask = (mask > 0.5).astype(np.float32)  # Binarize after blurring

        return mask

    def _generate_instrument_mask(self) -> np.ndarray:
        """Generates a thin, long mask (simulating a surgical instrument)."""
        mask = np.zeros(self.size, dtype=np.float32)

        # Start and end points for a line
        start_x = random.randint(0, self.size[0] - 1)
        start_y = random.randint(0, self.size[1] - 1)
        end_x = random.randint(0, self.size[0] - 1)
        end_y = random.randint(0, self.size[1] - 1)

        # Create a line approximation
        num_points = int(np.hypot(end_x - start_x, end_y - start_y))
        x = np.linspace(start_x, end_x, num_points).astype(int)
        y = np.linspace(start_y, end_y, num_points).astype(int)

        # Clip to boundaries and set instrument path
        x = np.clip(x, 0, self.size[0] - 1)
        y = np.clip(y, 0, self.size[1] - 1)
        mask[y, x] = 1.0

        # Dilate the line to give it thickness
        from scipy.ndimage import binary_dilation
        dilation_structure = np.ones((random.randint(3, 7), random.randint(3, 7)))
        mask = binary_dilation(mask, structure=dilation_structure).astype(np.float32)

        return mask

    def generate_composite_mask(self) -> torch.Tensor:
        """Generates a composite mask based on random and surgical features."""

        # 1. Base Mask (Tissue fold / random shape)
        mask = self._generate_circle_mask()

        # 2. Add Instrument Occlusion (Surgical feature)
        if random.random() < self.instrument_ratio:
            instrument_mask = self._generate_instrument_mask()
            # Combine masks (logical OR)
            mask = np.clip(mask + instrument_mask, 0.0, 1.0)

        return torch.from_numpy(mask).unsqueeze(0).float()  # [1, H, W]

    def create_siamese_pair(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates an Anchor and a Positive image pair using a generated mask.
        :param image: The raw input medical image [C, H, W].
        :return: (Anchor Image, Positive (Occluded) Image)
        """
        # Ensure image is C, H, W
        if image.ndim == 4:  # Assume B=1 if 4D
            image = image.squeeze(0)

        mask = self.generate_composite_mask()  # [1, H, W]

        # Anchor is the original image
        anchor = image

        # Positive is the occluded image.
        # Apply mask: 1 - mask gives the occlusion area.
        # Multiplying by (1 - mask) makes the occluded area black (0).
        occlusion_mask = 1.0 - mask
        positive = anchor * occlusion_mask #.unsqueeze(0)

        return anchor, positive


# --- 2. Pre-training Module (Mixed Siamese Network) ---

class InfoNCELoss(nn.Module):
    """
    SimCLR/MoCo-style InfoNCE Loss for contrastive learning.
    This simplifies the loss to focus on bringing the positive pair (Anchor vs Occluded) closer.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """
        Calculates the InfoNCE loss for a single positive pair (anchor, positive).
        The negative samples are implicitly all other samples in the batch.
        :param z_anchor: Anchor embeddings [B, D].
        :param z_positive: Positive embeddings [B, D].
        :return: Scalar loss tensor.
        """
        # Normalize embeddings
        z_anchor = F.normalize(z_anchor, dim=1)
        z_positive = F.normalize(z_positive, dim=1)

        # Concatenate anchor and positive embeddings to form the main batch
        # [2B, D]
        features = torch.cat([z_anchor, z_positive], dim=0)

        # Compute cosine similarity matrix: [2B, 2B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs: 1 for positive, 0 otherwise
        batch_size = z_anchor.shape[0]
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)

        # The positive pairs are at (i, i+B) and (i+B, i)
        # 1. Anchor i to Positive i+B
        pos_mask_1 = torch.roll(torch.eye(batch_size, device=features.device), shifts=batch_size, dims=1)
        # 2. Positive i+B to Anchor i
        pos_mask_2 = torch.roll(torch.eye(batch_size, device=features.device), shifts=batch_size, dims=0)

        # Combine into the full mask for the 2B x 2B matrix
        # [B, B] [B, B]
        # [B, B] [B, B]
        # We need the positive pair connections:
        # A_i -> P_i and P_i -> A_i

        positive_pairs_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)

        # Anchor to Positive (i, i+B)
        positive_pairs_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        # Positive to Anchor (i+B, i)
        positive_pairs_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=features.device)

        # Exclude self-similarities (diagonal)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positive_pairs_mask = positive_pairs_mask[~mask].view(positive_pairs_mask.shape[0], -1)

        # Select the similarities for the positive pairs
        positives = similarity_matrix[positive_pairs_mask].view(2 * batch_size, -1)

        # The numerator of the InfoNCE loss (similarity with positive)
        logits = positives

        # All other samples (excluding self) are negatives
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)

        # InfoNCE Loss: -log( exp(pos) / sum(exp(all)) )
        # Using CrossEntropyLoss with logits and labels of 0 is equivalent to this:
        loss = F.cross_entropy(logits, labels)

        return loss


class SiameseSegFormer(nn.Module):
    """
    SegFormer Encoder wrapped in a Siamese architecture for pre-training.
    Uses a Projection Head to generate fixed-size embeddings.
    """

    def __init__(self, model_name: str = 'nvidia/mit-b0', projection_dim: int = 128):
        super().__init__()
        # Load the SegFormer Model (only the encoder/backbone)
        # We use SegformerModel which returns the features before the decoder head
        self.encoder = SegformerModel.from_pretrained(model_name)

        # Determine the dimension of the final encoder output
        # SegFormer outputs features from multiple stages. We take the last one.
        # For MiT-B0, the last feature dimension is typically 512
        encoder_output_dim = self.encoder.config.hidden_sizes[-1]

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
        # 1. Anchor Stream
        # The SegFormerModel returns a BaseModelOutput (features from all stages)
        # We grab the last hidden state (last_hidden_state)
        # [B, H*W, D] -> We need to pool or average it to [B, D]
        anchor_features = self.encoder(x_anchor).last_hidden_state

        # Global Average Pooling across the spatial dimension (H*W)
        # [B, H*W, D] -> [B, D]
        z_anchor_pooled = anchor_features.mean(dim=1).reshape(anchor_features.shape[0], -1)

        # 2. Positive Stream
        positive_features = self.encoder(x_positive).last_hidden_state
        # z_positive_pooled = positive_features.mean(dim=1)
        z_positive_pooled = positive_features.mean(dim=1).reshape(positive_features.shape[0], -1)

        # 3. Projection Head
        z_anchor = self.projection_head(z_anchor_pooled)
        z_positive = self.projection_head(z_positive_pooled)

        return z_anchor, z_positive


def pretrain_step(model: SiameseSegFormer, dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, loss_fn: InfoNCELoss, device: torch.device,
                  num_epochs=100):
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0
    mask_generator = MaskGenerator(size=IMAGE_SIZE)

    best_loss = float('inf')
    min_delta = 0.00001
    boredom = 0
    max_boredom = 10
    best_model = None
    for epoch in range(num_epochs):
        for batch_images in dataloader:
            # Assuming dataloader yields raw images [B, C, H, W]
            # Create Siamese Pairs dynamically for the batch
            batch_anchor = []
            batch_positive = []

            for image in batch_images:
                anchor, positive = mask_generator.create_siamese_pair(image)
                batch_anchor.append(anchor)
                batch_positive.append(positive)


            x_anchor = torch.stack(batch_anchor).to(device)
            x_positive = torch.stack(batch_positive).to(device)

            optimizer.zero_grad()

            # Forward pass through the Siamese network
            z_anchor, z_positive = model(x_anchor, x_positive)

            # Calculate InfoNCE Loss
            loss = loss_fn(z_anchor, z_positive)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Pretraining Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            boredom = 0
            logger.info("Saving best snapshot `msn_model.online_encoder` state dict for fine-tuning.")
            best_model = model.online_encoder.state_dict()
            torch.save(best_model,
                       '../segmenter/checkpoint/alternative_msn_segformer_pretrained.pth')

        else:
            boredom += 1
        if boredom > max_boredom:
            logger.info(f"No improvement after {boredom} epochs, terminating")
            break

    return best_model  # Return the pre-trained encoder weights


# --- 3. Fine-tuning & Validation Module ---

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
        logger.info("Loading pre-trained weights into SegFormer backbone...")
        # Get the keys for the shared encoder from the pre-trained state dict
        encoder_state_dict = {
            k: v for k, v in pretrain_state_dict.items()
            if k.startswith('encoder')
        }

        # Load the weights into the current model, ignoring the randomly initialized head
        # The strict=False handles keys missing in the decoder part
        self.load_state_dict(encoder_state_dict, strict=False)
        logger.info("Pre-trained encoder weights loaded successfully.")


def finetune_step(model: SupervisedSegFormer, dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device, num_epochs=100):
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0

    # Use a combined loss (Dice + Cross-Entropy) for better segmentation
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)  # Ignore index 255 for padded areas

    best_loss = float('inf')
    min_delta = 0.00001
    boredom = 0
    max_boredom = 10
    best_model = None
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:  # inputs=[B,C,H,W], labels=[B,H,W]
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            logits = outputs.logits  # Logits [B, num_labels, H/4, W/4]

            # Resize logits to match labels size (SegFormer outputs downsampled logits)
            resized_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

            # Calculate Cross Entropy Loss
            ce_loss = ce_loss_fn(resized_logits, labels)

            # For simplicity, we use only CE Loss in this example, but a Dice Loss
            # (or Tversky/Focal) is highly recommended for medical segmentation.
            loss = ce_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        logger.info(f"PFine-tuning Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            boredom = 0
            logger.info("Saving best snapshot `msn_model.online_encoder` state dict for fine-tuning.")
            best_model = model.online_encoder.state_dict()
            torch.save(best_model,
                       '../segmenter/checkpoint/alternative_msn_segformer_finetuned.pth')

        else:
            boredom += 1
        if boredom > max_boredom:
            logger.info(f"No improvement after {boredom} epochs, terminating")
            break

    return best_model  # Return the pre-trained encoder weights


def validate_step(model: SupervisedSegFormer, dataloader: torch.utils.data.DataLoader, device: torch.device):
    logger = logging.getLogger(__name__)
    model.eval()
    total_dice = 0.0
    num_classes = model.config.num_labels

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).long()

            outputs = model(inputs)
            logits = outputs.logits

            # Resize and get predictions
            resized_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = resized_logits.argmax(dim=1)  # [B, H, W]

            # Calculate Dice Score (A standard metric for medical segmentation)
            for i in range(preds.shape[0]):  # Iterate through batch
                pred = preds[i].flatten()
                label = labels[i].flatten()

                # We calculate Dice for the foreground class (assuming background=0, lesion=1)
                # This needs to handle multi-class if num_labels > 2

                # Simplified Dice for binary case (class 1)
                if num_classes == 2:
                    target_class = 1
                    intersection = torch.sum((pred == target_class) & (label == target_class)).item()
                    union = torch.sum(pred == target_class).item() + torch.sum(label == target_class).item()
                    dice = (2. * intersection) / (union + 1e-6)
                    total_dice += dice
                # Multi-class implementation would require iterating over classes

    avg_dice = total_dice / len(dataloader.dataset)
    logger.info(f"Validation Dice Score (Foreground): {avg_dice:.4f}")


# --- Example Usage (Mock Data and Setup) ---

def main():
    logger = logging.getLogger()
    logger.info("Starting pretraining run")

    # Mock Configuration
    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    NUM_CLASSES = 2  # Polyp/Lesion (1) and Background (0)
    finetune_percent = 0.1

    # Use CPU for simplicity in example
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    # Large Unannotated set for Pre-training
    pretrain_datasets = []
    for ds in PRETRAIN_DATASETS:
        pretrain_datasets.append(MSNPretrainDatasetHDF5(hdf5_path=ds))

    pretrain_dataset = ConcatDataset(pretrain_datasets)

    pretrain_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Small Annotated set for Fine-tuning
    finetune_data = FINETUNE_DATASETS[0]
    n_finetune = get_num_samples_from_hdf5(finetune_data)
    shuffled_indices = np.random.permutation(n_finetune)
    n_finetune = int(n_finetune*finetune_percent)
    finetune_indices = shuffled_indices[:n_finetune]
    validation_indices = shuffled_indices[n_finetune:]

    finetune_dataset = MSNFinetuneDatasetHDF5(hdf5_path=finetune_data,
                                              indices=finetune_indices)
    finetune_dataloader = torch.utils.data.DataLoader(
        finetune_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Annotated set for Validation
    validation_dataset = MSNFinetuneDatasetHDF5(hdf5_path=finetune_data,
                                                indices=validation_indices)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # ----------------------------------------------------
    # 1. PRE-TRAINING PHASE
    # ----------------------------------------------------

    logger.info("Starting Pre-training Phase (Siamese Network) ---")

    # Instantiate Siamese Model and Loss
    siamese_model = SiameseSegFormer(model_name='nvidia/mit-b0').to(device)
    pretrain_loss_fn = InfoNCELoss(temperature=0.1)

    # Use a large LR for pre-training (standard for self-supervised learning)
    pretrain_optimizer = torch.optim.AdamW(siamese_model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Perform a single epoch of pre-training for demonstration
    pretrain_weights = pretrain_step(
        siamese_model,
        pretrain_dataloader,
        pretrain_optimizer,
        pretrain_loss_fn,
        device
    )

    # ----------------------------------------------------
    # 2. FINE-TUNING PHASE
    # ----------------------------------------------------

    logger.info("Starting Fine-tuning Phase (Supervised Segmentation) ---")

    # Configure the standard SegFormer for the segmentation task
    segformer_config = SegformerConfig.from_pretrained('nvidia/segformer-b0', num_labels=NUM_CLASSES)

    # Instantiate the supervised model
    supervised_model = SupervisedSegFormer(segformer_config).to(device)

    # Load the pre-trained weights into the encoder
    supervised_model.load_pretrain_weights(pretrain_weights)

    # Use a small LR for fine-tuning to preserve pre-trained knowledge
    finetune_optimizer = torch.optim.AdamW(supervised_model.parameters(), lr=5e-5)

    # Perform a single epoch of fine-tuning for demonstration
    finetune_step(
        supervised_model,
        finetune_dataloader,
        finetune_optimizer,
        device
    )

    # ----------------------------------------------------
    # 3. VALIDATION PHASE
    # ----------------------------------------------------

    logger.info("Starting Validation Phase ---")

    validate_step(supervised_model, validation_dataloader, device)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    home_dir = Path.home()
    if not os.path.exists(os.path.join(home_dir, "segmenter")):
        os.makedirs(os.path.join(home_dir, "segmenter"))
    logfile = os.path.join(home_dir, "segmenter", f"alternative_msn_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        force=True,  # Resets any previous configuration - in Colab for example
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logfile)
        ]
    )
    logger = logging.getLogger(__name__)
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down gracefully.")
        sys.exit(0)
    finally:
        # This block will always be executed, allowing you to clean up resources
        # ensure log handlers are flushed.
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.info("Logger handlers flushed and closed. Exiting now.")
