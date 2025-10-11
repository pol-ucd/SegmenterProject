import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerConfig

from segmenter.core import Config, get_default_device_type
from segmenter.loss import DiceLoss
from segmenter.loss.msn import NTXentLoss, MSNLoss
from segmenter.masks import MaskGenerator
from segmenter.models.base import SupervisedSegFormer
from segmenter.models.msn import SimCLRSegFormer
from segmenter.utils.msn import load_data

# Configuration
config = Config("config/msn_common.json")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone_model = "nvidia/segformer-b4-finetuned-ade-512-512"

learning_rate = 1e-05
# batch_size=config['optimizer']['params']['batch_size']

BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 2  # Polyp/Lesion (1) and Background (0)
finetune_percent = 0.1
IMAGE_SIZE=(512, 512)
N_EPOCHS = 200

prefix='msn_simclr'

# Assumes:
# - model is SimCLRSegFormer instance
# - simclr_loss is a contrastive loss function expecting (z_i, z_j) pairs, e.g., NT-Xent
# - optimizer and optional scaler are defined
# - dataloader yields dict with "images": tensor (B, C, H, W)
# - device set
def pretrain_step(model: SimCLRSegFormer, dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, loss_fn: NTXentLoss, device: torch.device,
                  scaler=None, num_epochs=100):
    logger = logging.getLogger(__name__)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler()

    model.train()
    total_loss = []
    # mask_generator = MaskGenerator(size=IMAGE_SIZE)

    best_loss = float('inf')
    min_delta = 0.00001
    boredom = 0
    max_boredom = 10
    best_model = None
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            x = batch["images"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=get_default_device_type(),
                                    dtype=torch.float16,
                                    enabled=(scaler is not None)):
                z_anchor, z_positive = model(x, return_patches=False)
                z_anchor = z_anchor.to(dtype=torch.float32)
                z_positive = z_positive.to(dtype=torch.float32)

                loss = loss_fn(z_anchor, z_positive)
                total_loss += [loss.item()]

            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                if isinstance(loss_fn, MSNLoss):
                    loss_fn.update_center(z_anchor.detach().to(torch.float32))

        if scheduler is not None:
            scheduler.step()

        avg_loss = np.mean(total_loss)
        logger.info(f"Pretraining Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            boredom = 0
            logger.info("Saving best snapshot `msn_model.online_encoder` state dict for fine-tuning.")
            try:
                best_model = model.online_wrapper.state_dict()
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
        for inputs, labels in tqdm(dataloader):
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
    logger.info(f"Starting pretraining run for {prefix.upper()}")

    # Mock Configuration
    NUM_CLASSES = 2  # Polyp/Lesion (1) and Background (0)
    finetune_percent = 0.1

    logger.info(f"Using device: {device}")

    finetune_dataloader, pretrain_dataloader, validation_dataloader = load_data(BATCH_SIZE, finetune_percent)

    # ----------------------------------------------------
    # 1. PRE-TRAINING PHASE
    # ----------------------------------------------------

    logger.info("Loading models for Pre-training Phase (Siamese Network) ---")

    # Instantiate Siamese Model and Loss
    siamese_model = SimCLRSegFormer(pretrained_model=backbone_model).to(device)
    pretrain_loss_fn = NTXentLoss(temperature=0.5)

    # Use a large LR for pre-training (standard for self-supervised learning)
    pretrain_optimizer = torch.optim.AdamW(siamese_model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=1e-4)
    logger.info("Starting Pre-training Phase (Siamese Network) ---")

    pretrain_weights = pretrain_step(
        siamese_model,
        pretrain_dataloader,
        pretrain_optimizer,
        pretrain_loss_fn,
        device=device,
        num_epochs=N_EPOCHS
    )

    logger.info("Completed Pre-training Phase (Siamese Network) ---")

    # # ----------------------------------------------------
    # # 2. FINE-TUNING PHASE
    # # ----------------------------------------------------
    #
    # logger.info("Starting Fine-tuning Phase (Supervised Segmentation) ---")
    #
    # # Configure the standard SegFormer for the segmentation task
    # segformer_config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512", num_labels=NUM_CLASSES)
    #
    # # Instantiate the supervised model
    # supervised_model = SupervisedSegFormer(segformer_config).to(device)
    #
    # # Load the pre-trained weights into the encoder
    # supervised_model.load_pretrain_weights(pretrain_weights)
    #
    # # Use a small LR for fine-tuning to preserve pre-trained knowledge
    # finetune_optimizer = torch.optim.AdamW(supervised_model.parameters(), lr=5e-5)
    #
    # # Perform a single epoch of fine-tuning for demonstration
    # finetune_step(
    #     supervised_model,
    #     finetune_dataloader,
    #     finetune_optimizer,
    #     device
    # )

    # ----------------------------------------------------
    # 3. VALIDATION PHASE
    # ----------------------------------------------------

    # logger.info(f"Starting Validation Phase for {prefix.upper()} ---")
    #
    # validate_step(supervised_model, validation_dataloader, device)

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
