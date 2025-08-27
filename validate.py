import json
import logging
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from nn.data import SemanticSegmentationDatasetBasic
from nn.models import SegformerBinarySegmentation
from nn.modules import HybridLoss
from nn.torch_utils import RunManager


def main():
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

    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    image_size = tuple(params['image_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    Load the list of training and testing images and masks.
    We take the same percentage split from each separate source of images so
    that we are guaranteed to have a representation from each source in the
    training and testing sets.
    """


    df_files = pd.read_csv("validate_files.csv")

    logger.info(f"Using {device} device for model training.")
    val_images = df_files.val_image.values
    val_masks = df_files.val_masks.values

    val_ds = SemanticSegmentationDatasetBasic(
        val_images,
        val_masks,
        image_size=image_size
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)


    logger.info(f"Validation batches: {len(val_loader)}")

    model = SegformerBinarySegmentation(num_classes=num_classes)
    model.load_state_dict(torch.load("best_dice_model.pth", map_location=device))
    model.to(device)

    loss_fn = HybridLoss(weight_ce=0.5,
                         weight_dice=0.0,
                         weight_focal=0.2,
                         weight_tversky=0.3)

    trainer = RunManager(model,
                         optimizer=None,
                         criterion=loss_fn,
                         scaler=None,
                         train_loader=None,
                         eval_loader=val_loader,
                         save_preds=False,
                         save_preds_path=""
                         )

    eval_params = {}

    val_metrics = trainer.evaluate(**eval_params)

    val_loss = val_metrics['loss']
    val_miou = val_metrics['iou']
    val_dice = val_metrics['dice']

    logger.info(
        f"Evaluation Losses: | Combined Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IOU: {val_miou:.4f}")

if __name__ == "__main__":

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

