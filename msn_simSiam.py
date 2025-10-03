import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerConfig

from segmenter.loss import DiceLoss
from segmenter.loss.msn import SimSiamLoss
from segmenter.masks import MaskGenerator
from segmenter.models.base import SupervisedSegFormer
from segmenter.models.msn import SimSiamSegFormer
from segmenter.utils.msn import load_data

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 12
NUM_WORKERS = 4
NUM_CLASSES = 2  # Polyp/Lesion (1) and Background (0)
finetune_percent = 0.1
IMAGE_SIZE=(512, 512)

prefix='msn_simsiam'


def pretrain_step(model: SimSiamSegFormer, dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, loss_fn: SimSiamLoss, device: torch.device,
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
        for batch_images in tqdm(dataloader):
            # Assuming dataloader yields raw images [B, C, H, W]
            # Create Siamese Pairs dynamically for the batch
            batch_anchor = []
            batch_positive = []

            for image in batch_images['images']:
                anchor, positive = mask_generator.create_siamese_pair(image)
                batch_anchor.append(anchor)
                batch_positive.append(positive)

            x1 = torch.stack(batch_anchor).to(device)
            x2 = torch.stack(batch_positive).to(device)

            optimizer.zero_grad()

            # Forward pass through the SimSiam network (returns 4 tensors for symmetric loss)
            p1, z2_detached, p2, z1_detached = model(x1, x2)

            # Calculate SimSiam Loss (Symmetric)
            loss = loss_fn(p1, z2_detached, p2, z1_detached)

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
            try:
                best_model = model.online_encoder.state_dict()
                torch.save(best_model,
                           f'../segmenter/checkpoint/{prefix}_segformer_pretrained.pth')
            except Exception as e:
                logger.error(f"Pretraining failed to save `{prefix}_segformer_pretrained.pth`: {e}")

        else:
            boredom += 1
        if boredom > max_boredom:
            logger.info(f"No improvement after {boredom} epochs, terminating")
            break

    return best_model  # Return the pre-trained encoder weights


def finetune_step(model: SupervisedSegFormer, dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device, num_epochs=100):
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0
    CE_WEIGHT, DICE_WEIGHT = 0.5, 0.5
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)  # Standard Cross Entropy
    dice_loss_fn = DiceLoss(num_classes=model.config.num_labels, ignore_index=255)  # Custom Dice Loss

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

            # Calculate Losses
            ce_loss = ce_loss_fn(resized_logits, labels)
            dice_loss = dice_loss_fn(resized_logits, labels)

            # Combined Loss
            loss = CE_WEIGHT * ce_loss + DICE_WEIGHT * dice_loss

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
            try:
                best_model = model.online_encoder.state_dict()
                torch.save(best_model,
                           f'../segmenter/checkpoint/{prefix}_segformer_finetuned.pth')
            except Exception as e:
                logger.error(f"Finetuning failed to save `{prefix}_segformer_pretrained.pth`: {e}")

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
    logger.info(f"Starting pretraining run {prefix.upper()}")

    logger.info(f"Using device: {device}")

    finetune_dataloader, pretrain_dataloader, validation_dataloader = load_data(BATCH_SIZE, finetune_percent)

    # ----------------------------------------------------
    # 1. PRE-TRAINING PHASE
    # ----------------------------------------------------

    logger.info("Loading models for Pre-training Phase (Siamese Network) ---")

    # Instantiate Siamese Model and Loss
    siamese_model = SimSiamSegFormer(model_name='nvidia/mit-b0').to(device)
    pretrain_loss_fn = SimSiamLoss()

    # Use a large LR for pre-training (standard for self-supervised learning)
    pretrain_optimizer = torch.optim.AdamW(siamese_model.parameters(), lr=1e-3, weight_decay=1e-4)
    logger.info("Starting Pre-training Phase (Siamese Network) ---")

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
    logfile = os.path.join(home_dir, "segmenter", f"{prefix}_{timestamp}.log")

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
