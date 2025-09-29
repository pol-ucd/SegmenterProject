import glob
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from segmenter.loss.hybrid import HybridLoss
from segmenter.torch_utils import CheckpointManager, RunManager2
from segmenter.utils.data import (get_num_samples_from_hdf5, HDF5ImageDataset)


def main():
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger()
    home = Path.home()
    if not os.path.exists(os.path.join(home, "segmenter")):
        logger.warning("Creating 'segmenter' directory in $HOME directory.")
        os.makedirs(os.path.join(home, "segmenter"))
    if not os.path.exists(os.path.join(home, "segmenter/data")):
        logger.warning("Creating 'data' directory in $HOME/segmenter directory.")
        os.makedirs(os.path.join(home, "segmenter/data"))
    if os.path.isfile(os.path.join(os.path.join(home, "segmenter"),
                                                "lesion_params.json")):
        params_file = os.path.join(os.path.join(home, "segmenter"),
                                   "lesion_params.json")
    else:
        params_file = "lesion_params.json"




    """ Load datasets for test and training """
    # Now we correctly split the indices and create new datasets
    train_datasets = []
    test_datasets = []

    n_records = 0
    for hdf5_file in hdf5_files:
        len_hdf5 = get_num_samples_from_hdf5(hdf5_file)
        # Ensure reproducibility of training and test with a fixed seed
        torch.manual_seed(42)
        shuffled_indices = torch.randperm(len_hdf5)
        test_indices = shuffled_indices[:int(len_hdf5 * test_split)]
        train_indices = shuffled_indices[int(len_hdf5 * test_split):]


        train_datasets.append(HDF5ImageDataset(
            hdf5_path=hdf5_file,
            indices=train_indices,
            is_train_split=True,
            image_size=image_size,
            n_augment=n_augments
        ))


        test_datasets.append(HDF5ImageDataset(
            hdf5_path=hdf5_file,
            indices=test_indices,
            is_train_split=False,
            image_size=image_size,
            n_augment=0
        ))

        n_records += len_hdf5
    logger.info(f"Using {n_records} total records for training and testing.")

    final_train_dataset = train_datasets[0]
    final_test_dataset = test_datasets[0]

    train_loader = DataLoader(
        final_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        final_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    logger.info(f"Successfully loaded training and testing dataset for {n_records} records.")
    logger.info(f"Number of batches in the training DataLoader: {len(train_loader)}")
    logger.info(f"Number of batches in the test DataLoader: {len(test_loader)}")

    cp_manager = CheckpointManager(checkpoint_dir=checkpoint_path,
                                   prefix=checkpoint_prefix,
                                   patience=checkpoint_patience,
                                   min_delta=checkpoint_min_delta)
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

    loss_params = params['loss']
    loss_fn = HybridLoss(loss_params['loss_params'])

    # Initial freeze all parameters of the model
    # logger.info("Freezing encoder layers...")
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    #
    # # Unfreeze the decoder head and segmentation head because we replaced these
    # logger.info("Unfreezing decoder and segmentation head...")
    # for param in model.base_model.decode_head.parameters():
    #     param.requires_grad = True

    # Only pass the parameters that require gradients to the optimizer
    optimizer = torch.optim.AdamW(
        # params=filter(lambda p: p.requires_grad, model.parameters()),
        params=model.parameters(),
        lr=learning_rate,
        # weight_decay=l2_decay_penalty  # L2 regularization to prevent large weights
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['scheduler']['T_max'])



    trainer = RunManager2(
                         config_path=params_file
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

        # scheduler.step(epoch + 1)

        stop_training = cp_manager.save(model, 1 - val_miou)
        if stop_training:
            logger.info(f"Training stopped early at epoch {epoch} with mIOU Score: {val_miou:.4f}")
            break


if __name__ == "__main__":
    # --- Logging Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    home_dir = Path.home()
    if not os.path.exists(os.path.join(home_dir, "segmenter")):
        os.makedirs(os.path.join(home_dir, "segmenter"))
    logfile = os.path.join(home_dir, "segmenter", f"training_{timestamp}.log")

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
