import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from tqdm import tqdm

from segmenter.core import Config, set_default_device, get_default_device, get_default_device_type
from segmenter.loss import DiceLoss, MSNLoss
from segmenter.models import MoCoSiameseNetwork, SegFormerFeatureWrapper
from segmenter.core.torch import get_default_device_type
from segmenter.models.msn import SupervisedSegformerSegmentation
from segmenter.utils.msn import load_data, load_finetune

# Configuration
config = Config("config/msn_common.json")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone_model = "nvidia/segformer-b4-finetuned-ade-512-512"

learning_rate = 1e-06
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 2  # Polyp/Lesion (1) and Background (0)
finetune_percent = 0.1
IMAGE_SIZE = (512, 512)
N_EPOCHS = 200
BATCH_SIZE_FINETUNE = 8

WARMUP_EPOCHS = 5
TOTAL_EPOCHS = 200
STEPS_PER_EPOCH = 33000 // BATCH_SIZE  # Example: 33,000 images / 64 batch size = ~516 steps
WARMUP_STEPS = WARMUP_EPOCHS * STEPS_PER_EPOCH
TOTAL_STEPS = TOTAL_EPOCHS * STEPS_PER_EPOCH

prefix = 'msn_moco'


def pretrain_step(model: MoCoSiameseNetwork,
                  dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: nn.Module,
                  scaler=None,
                  device: torch.device = None,
                  num_epochs=200):
    logger = logging.getLogger(__name__)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # scaler = None
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler()

    model.train()
    total_loss = []

    best_loss = float('inf')
    min_delta = 0.00001
    boredom = 0
    max_boredom = 10
    best_model = None
    for epoch in range(num_epochs):
        for batch_idx, batch_images in enumerate(tqdm(dataloader)):
            x = batch_images["images"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=get_default_device_type(),
                                    dtype=torch.float16,
                                    enabled=(scaler is not None)):
                online_emb, target_emb, masked_indices = model(x, epoch=epoch, batch_index=batch_idx)
                # ensure float32 and detached target for center update
                online_emb = F.normalize(online_emb.to(torch.float32), dim=-1)
                target_emb = F.normalize(target_emb.to(torch.float32), dim=-1).detach()
                loss = loss_fn(online_emb, target_emb[masked_indices])
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

            # AFTER optimizer.step: EMA then center update
            model.update_momentum_encoder()
            with torch.no_grad():
                if isinstance(loss_fn, MSNLoss):
                    loss_fn.update_center(target_emb.detach().to(torch.float32))

            # print(
            #     f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f} | "
            #     f"Center norm {loss_fn.target_center.norm().item() if getattr(loss_fn, 'target_center', None) is not None else 'None'}")

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


def finetune_step(model: SupervisedSegformerSegmentation,
                  dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  num_epochs=1):
    logger = logging.getLogger(__name__)
    CE_WEIGHT, DICE_WEIGHT = 0.5, 0.5
    model.train()
    total_loss = []
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)  # Standard Cross Entropy
    dice_loss_fn = DiceLoss(num_classes=model.num_classes,
                            ignore_index=255)  # Custom Dice Loss

    best_loss = float('inf')
    min_delta = 0.00001
    boredom = 0
    max_boredom = 10
    best_model = None
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        for data_item in tqdm(dataloader):
            # inputs=[B,C,H,W], labels=[B,H,W]
            inputs = data_item["images"].to(device)
            labels = data_item['masks'].to(device).long()
            print("inputs: ", inputs.shape, "labels: ", labels.shape)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # logits = outputs.logits  # Logits [B, num_labels, H/4, W/4]
            print(outputs.shape)

            # Resize logits to match labels size (SegFormer outputs downsampled logits)
            resized_logits = F.interpolate(outputs,
                                           size=labels.shape[-2:],
                                           mode="bilinear",
                                           align_corners=False)

            # Calculate Losses
            # ce_loss = ce_loss_fn(resized_logits, labels)
            loss = dice_loss_fn(resized_logits, labels)

            # Combined Loss
            # loss = CE_WEIGHT * ce_loss + DICE_WEIGHT * dice_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += [loss.item()]

        avg_loss = np.mean(total_loss)

        logger.info(f"PFine-tuning Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            boredom = 0
            logger.info("Saving best snapshot state dict for fine-tuning.")
            try:
                best_model = model.state_dict()
                torch.save(best_model,
                           f"../segmenter/checkpoint/{prefix}_segformer_fine_tuned.pth")
            except Exception as e:
                logger.error(f"Finetuning failed to save `{prefix}_segformer_fine_tuned.pth`: {e}")

        else:
            boredom += 1
        if boredom > max_boredom:
            logger.info(f"No improvement after {boredom} epochs, terminating")
            break

    return best_model  # Return the pre-trained encoder weights


def validate_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    logger = logging.getLogger(__name__)
    model.eval()
    total_dice = []

    num_classes = model.num_classes

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
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
                    total_dice += [dice]
                # Multi-class implementation would require iterating over classes

    avg_dice = np.mean(total_dice)
    logger.info(f"Validation Dice Score (Foreground): {avg_dice:.4f}")


def main():
    logger = logging.getLogger()
    device = get_default_device()
    set_default_device(device)
    logger.info(f"Starting finetuning run {prefix.upper()}")

    logger.info(f"Using device: {device}")

    finetune_dataloader, validation_dataloader = load_finetune(batch_size=BATCH_SIZE_FINETUNE,
                                                               finetune_percent=finetune_percent)

    # ----------------------------------------------------
    # 1. FINETUNING PHASE
    # ----------------------------------------------------
    checkpoint = f'../segmenter/checkpoint/{prefix}_segformer_pretrained.pth'
    logger.info(f"Loading models for Finetuning Phase ({prefix})")

    # Load the base model
    supervised_model = SupervisedSegformerSegmentation(pretrained_model=backbone_model,
                                                       num_classes=2,
                                                       checkpoint=checkpoint).to(device)

    logger.info(f"Successfully loaded models for Finetuning Phase ({prefix})")

    # # Use a small LR for fine-tuning to preserve pre-trained knowledge
    finetune_optimizer = torch.optim.AdamW(supervised_model.parameters(), lr=5e-5)
    #
    # # Perform a single epoch of fine-tuning for demonstration
    finetune_step(
        supervised_model,
        finetune_dataloader,
        finetune_optimizer
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
