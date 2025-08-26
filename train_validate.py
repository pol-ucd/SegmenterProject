import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import GradScaler
from torch.utils.data import DataLoader

from nn.data import (split_images_and_masks,
                     SemanticSegmentationDatasetAugmentor,
                     SemanticSegmentationDatasetBasic, CheckpointManager)
from nn.models import SegformerBinarySegmentation
from nn.modules import HybridLoss
from utils.torch_utils import RunManager


def main():
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    # --- Load parameters from JSON file ---
    try:
        with open('params.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        logger.error("Error: 'params.json' file not found. Please ensure it is in the same directory.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from 'params.json': {e}")
        return

    pretrained_model = params['pretrained_model']
    checkpoint_path = params['checkpoints']['path']
    checkpoint_prefix = params['checkpoints']['prefix']
    checkpoint_patience = params['checkpoints']['patience']
    checkpoint_min_delta = params['checkpoints']['min_delta']
    if not os.path.isdir(checkpoint_path):
        logger.info(f"Checkpoint directory '{checkpoint_path}' not found. Saving to '[current directory]/checkpoints' instead.")
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints")

    test_split = params['test_split']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    n_augments = params['n_augments']
    image_size = tuple(params['image_size'])
    learning_rate = params['optimizer_settings']['learning_rate']
    l2_decay_penalty = params['optimizer_settings']['l2_decay_penalty']
    n_epochs = params['n_epochs']

    image_paths = params["datasets"]["image_paths"]
    mask_paths = params["datasets"]["mask_paths"]
    file_types = params["datasets"]["file_types"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using {device} device for model training.")
    logger.info(f"Loaded parameters: {params}")

    cp_manager = CheckpointManager(checkpoint_dir=checkpoint_path,
                                   prefix=checkpoint_prefix,
                                   patience=checkpoint_patience,
                                   min_delta=checkpoint_min_delta)

    """
    Setup the model 
    """
    model = SegformerBinarySegmentation(pretrained_model=pretrained_model,
                                        num_classes=num_classes)
    latest_checkpoint = None
    try:
        # Get the list of saved checkpoints
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_path, checkpoint_prefix + "*.pt")))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            cp_manager.load(model, latest_checkpoint, device=device)
            logger.info(f"Loaded model checkpoint {latest_checkpoint}.")
        else:
            logger.info(f"No checkpoints were saved to load in {checkpoint_path}.")
    except FileNotFoundError as e:
        logger.info(f"Unable to load checkpoint {e}")

    model.to(device)
    logger.info(f"Instantiated model with {model.num_classes} classes.")

    """
    Load the list of training and testing images and masks.
    We take the same percentage split from each separate source of images so
    that we are guaranteed to have a representation from each source in the
    training and testing sets.
    """
    found_existing_data = False
    if latest_checkpoint is not None:
        latest_csv = latest_checkpoint.split(".pt")[0] + ".csv"
        try:
            df_data = pd.read_csv(latest_csv)
            train_images = df_data[df_data.phase == "T"]["image"].values
            train_masks = df_data[df_data.phase == "T"]["mask"].values
            val_images = df_data[df_data.phase == "V"]["image"].values
            val_masks = df_data[df_data.phase == "V"]["mask"].values
            found_existing_data = True
        except FileNotFoundError as e:
            logger.info(f"No existing list of test and training data found for  {latest_csv}.")
            found_existing_data = False

    if not found_existing_data:
        logger.info(f"Loading new training and testing images and masks.")
        try:
            (train_images,
             train_masks,
             val_images,
             val_masks) = split_images_and_masks(image_paths=image_paths,
                                                 mask_paths=mask_paths,
                                                 file_types=file_types,
                                                 split=test_split)
        except FileNotFoundError as e:
            logger.error(f"Error loading data: {e}. Please ensure the data directories are correctly set up.")
            return  # Exit if data cannot be loaded
        except Exception as e:
            logger.error(f"An unexpected error occurred during data splitting: {e}")
            return

        """
        Save the names of the files in the validation & training subset for 
        later use
        """
        csv_path = os.path.join(checkpoint_path,
                                cp_manager.get_prefix() + "_" + cp_manager.get_timestamp() + ".csv")
        df_dict = {"image": np.concatenate([train_images, val_images]),
                   "mask": np.concatenate([train_masks, val_masks]),
                   "phase": ["T"] * len(train_images) + ["V"] * len(val_images)}
        print(csv_path)
        pd.DataFrame(df_dict).to_csv(csv_path, index=False)

    """ 
    Data sets and loaders
    """
    """
    Only use the SemanticSegmentationDatasetAugmentor class for 
    training data sine it randomly augments the available data
    to create more training data - and so is not suitable for 
    test or validation
    """
    train_ds = SemanticSegmentationDatasetAugmentor(
        train_images,
        train_masks,
        n_augments=n_augments,
        image_size=image_size
    )

    """
    Use the SemanticSegmentationDatasetBasic class for 
    validation or test. It does not perform any augmentations
    other than resizing and standard normalisation of the 
    images. Masks are not normalised.
    """
    val_ds = SemanticSegmentationDatasetBasic(
        val_images,
        val_masks,
        image_size=image_size
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(val_loader)}")

    denom = 0.0
    for weight in params['loss_weights']:
        denom += params['loss_weights'][weight]

    loss_fn = HybridLoss(weight_ce=params['loss_weights']['weight_ce'] / denom,
                         weight_dice=params['loss_weights']['weight_dice'] / denom,
                         weight_focal=params['loss_weights']['weight_focal'] / denom,
                         weight_tversky=params['loss_weights']['weight_tversky'] / denom,
                         weight_iou=params['loss_weights']['weight_iou'] / denom,
                         weight_boundary=params['loss_weights']['weight_boundary'] / denom)

    # Initial freeze all parameters of the model
    logger.info("Freezing encoder layers...")
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Unfreeze the decoder head and segmentation head because we replaced these
    logger.info("Unfreezing decoder and segmentation head...")
    for param in model.base_model.decode_head.parameters():
        param.requires_grad = True

    # Only pass the parameters that require gradients to the optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=l2_decay_penalty  # L2 regularization to prevent large weights
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['scheduler']['T_max'])
    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, T_max = 50)

    """
    Only use GradScaler if we have CUDA
    """
    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()

    trainer = RunManager(model,
                         optimizer,
                         criterion=loss_fn,
                         scaler=scaler,
                         train_loader=train_loader,
                         eval_loader=val_loader,
                         save_preds=False,
                         save_preds_path=""
                         )
    train_params = {}
    eval_params = {}

    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")

        train_metrics = trainer.train(**train_params)
        val_metrics = trainer.evaluate(**eval_params)

        train_loss = train_metrics['loss']
        train_miou = train_metrics['iou']
        train_dice = train_metrics['dice']
        val_loss = val_metrics['loss']
        val_miou = val_metrics['iou']
        val_dice = val_metrics['dice']

        logger.info(f"Training Losses  : | Compound: {train_loss:.4f} | Dice: {train_dice:.4f} | IOU: {train_miou:.4f}")
        logger.info(f"Evaluation Losses: | Compound: {val_loss:.4f} | Dice: {val_dice:.4f} | IOU: {val_miou:.4f}")

        scheduler.step(epoch + 1)

        stop_training = cp_manager.save(model, 1 - val_miou)
        if stop_training:
            logger.info(f"Training stopped early at epoch {epoch} with mIOU Score: {val_miou:.4f}")
            break


if __name__ == "__main__":
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        force=True,  # Resets any previous configuration - in Colab for example
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )
    logger = logging.getLogger()
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
