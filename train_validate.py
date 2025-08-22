import json
import logging

import pandas as pd
import torch
from torch import GradScaler
from torch.utils.data import DataLoader

from nn.data import (split_images_and_masks,
                     SemanticSegmentationDatasetAugmentor,
                     SemanticSegmentationDatasetBasic)
from nn.models import SegformerBinarySegmentation
from nn.modules import EarlyStopping, HybridLoss
from utils.torch_utils import RunManager


def main():
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log") # Uncomment to also log to a file
        ]
    )
    logger = logging.getLogger(__name__)

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

    test_split = params['test_split']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    n_augments = params['n_augments']
    image_size = tuple(params['image_size'])
    learning_rate = params['optimizer_settings']['learning_rate']
    l2_decay_penalty = params['optimizer_settings']['l2_decay_penalty']
    n_epochs = params['n_epochs']
    stopper_patience = params['stopper_patience']
    pretained_model = params['pretrained_model']
    save_model_name = params['save_model_name']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using {device} device for model training.")
    logger.info(f"Loaded parameters: {params}")

    """
    Load the list of training and testing images and masks.
    We take the same percentage split from each separate source of images so
    that we are guaranteed to have a representation from each source in the
    training and testing sets.
    """
    try:
        train_images, train_masks, val_images, val_masks = split_images_and_masks(split=test_split)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}. Please ensure the data directories are correctly set up.")
        return  # Exit if data cannot be loaded
    except Exception as e:
        logger.error(f"An unexpected error occurred during data splitting: {e}")
        return

    """
    Save the names of the files in the validation/test subset for 
    later use
    """
    df_file = pd.DataFrame({"val_image": val_images,
                            "val_masks": val_masks, })
    df_file.to_csv("validate_files.csv",
                   index=False)

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

    """
    Setup the model 

    """
    # model = SegformerBinarySegmentation4(pretrained_model=pretained_model,
    #                                      num_classes=num_classes)  # [B, num_classes, H, W]
    # model.to(device)
    model = SegformerBinarySegmentation(num_classes=num_classes).to(device)
    logger.info(f"Instantiated model with {model.num_classes} classes.")

    loss_fn = HybridLoss(weight_ce=params['loss_weights']['weight_ce'] / 2.1,
                         weight_dice=params['loss_weights']['weight_dice'] / 2.1,
                         weight_focal=params['loss_weights']['weight_focal'] / 2.1,
                         weight_tversky=params['loss_weights']['weight_tversky'] / 2.1,
                         weight_iou=params['loss_weights']['weight_iou'] / 2.1)

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

    early_stopper = EarlyStopping(patience=stopper_patience, min_delta=0.0001,
                                  mode='min', verbose=True,
                                  save_path=save_model_name)

    for epoch in range(n_epochs):
        logger.info("-" * 20)
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

        early_stopper(val_miou, model, epoch)

        if early_stopper.early_stop:
            logger.info(f"Training stopped early at epoch {epoch}")
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        logging.info("Processing completed. Exiting.")

