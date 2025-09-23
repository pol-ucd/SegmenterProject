import glob
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from segmenter.data import (get_num_samples_from_hdf5, HDF5ImageDataset)
from segmenter.modules import HybridLoss
from segmenter.torch_utils import RunManager, CheckpointManager


def check_scores(metric:dict[str,list])-> bool:
    all_lens = np.array([len(v) for v in metric.values()])
    base_len = all_lens[0]
    if not np.all(all_lens == base_len):
        for k,v in metric.items():
            print(f"{k}: {len(v)}")
    return np.all(all_lens == base_len)



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
                                   "classica_base_segformer_params.json")):
        params_file = os.path.join(os.path.join(home, "segmenter"),
                                   "classica_base_segformer_params.json")
    else:
        params_file = "classica_base_segformer_params.json"

    # --- Load parameters from JSON file ---
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: {params_file} file not found. Please ensure it is in the same directory.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {params_file}: {e}")
        return

    # Control all the random seeds we will use for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_params = params['model']['params']
    pretrained_model = model_params['pretrained_model']

    checkpoint_path = params['checkpoints']['path']
    checkpoint_prefix = params['checkpoints']['prefix']
    checkpoint_patience = params['checkpoints']['patience']
    checkpoint_min_delta = params['checkpoints']['min_delta']
    if not os.path.isdir(checkpoint_path):
        logger.info(
            f"Checkpoint directory '{checkpoint_path}' not found. Saving to '[current directory]/checkpoints' instead.")
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints")

    """ Configure the run """
    run_params = params['run']
    num_classes = run_params['num_classes']
    batch_size = run_params['batch_size']
    num_workers = run_params['num_workers']
    n_augments = run_params['n_augments']
    image_size = tuple(run_params['image_size'])
    n_epochs = run_params['n_epochs']

    """ Optimiser settings """
    opt_params = params['optimizer']['params']
    learning_rate = opt_params['learning_rate']
    l2_decay_penalty = opt_params['l2_decay_penalty']

    """ Data settings """
    hdf5_path = params["datasets"]["hdf5_dir"]
    hdf5_file = [os.path.join(hdf5_path, _h) for _h in params["datasets"]["hdf5_files"]][0]
    # logger.info(f"Loaded parameters: {params}")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # logger.info(f"Using {device} device for model training.")

    """ Load datasets for test and training """
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    n_folds = 5
    metrics = {"case": [],
               "test_split": [],
               "test_iteration": [],
               "is_test": [],
               "dice": [],
               "iou": [],
               "precision": [],
               "recall": []}

    # scorer = Scorer()

    with h5py.File(hdf5_file, 'r', swmr=True) as hdf:
        original_names_hdf = hdf['original_name']
        original_names = np.array([h.decode('utf-8') for h in original_names_hdf])
        n_records = len(original_names)

    records_offset = 0
    for test_split in test_sizes:

        logger.info(f"Testing split {test_split} for {n_folds} iterations.")

        ss = ShuffleSplit(n_splits=n_folds, test_size=test_split, random_state=42)
        len_hdf5 = get_num_samples_from_hdf5(hdf5_file)
        shuffled_indices = np.random.permutation(len_hdf5)

        for idx, (train_index, test_index) in enumerate(ss.split(shuffled_indices)):

            test_names = original_names[test_index]
            train_names = original_names[train_index]

            final_eval_dataset = HDF5ImageDataset(
                hdf5_path=hdf5_file,
                indices=test_index,
                is_train_split=False,
                image_size=image_size,
                n_augment=0
            )

            """ 
            Create a non-augmented test dataset from the training records so
            we get metrics for the unaugmented records only when we train.
            """
            final_test_dataset = HDF5ImageDataset(
                hdf5_path=hdf5_file,
                indices=train_index,
                is_train_split=False,
                image_size=image_size,
                n_augment=0
            )

            final_train_dataset = HDF5ImageDataset(
                hdf5_path=hdf5_file,
                indices=train_index,
                is_train_split=True,
                image_size=image_size,
                n_augment=n_augments
            )

            logger.info(f"Starting fold [{idx + 1}/{n_folds}] for test split [{test_split}].")

            train_loader = DataLoader(
                final_train_dataset,
                batch_size=batch_size,
                shuffle=False,  # Already randomly shuffled
                num_workers=num_workers
            )

            test_loader = DataLoader(
                final_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

            eval_loader = DataLoader(
                final_eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

            logger.info(f"Successfully loaded training and testing dataset for {n_records} records.")

            logger.info(f"Number of batches in the training DataLoader: "
                        f"{len(train_loader)} / {len(final_train_dataset)} records")
            logger.info(f"Number of batches in the test DataLoader: "
                        f"{len(test_loader)} / {len(final_test_dataset)} records")
            logger.info(f"Number of batches in the evaluation DataLoader: "
                        f"{len(eval_loader)} / {len(final_eval_dataset)} records")

            cp_manager = CheckpointManager(checkpoint_dir=checkpoint_path,
                                           prefix=checkpoint_prefix,
                                           patience=checkpoint_patience,
                                           min_delta=checkpoint_min_delta,
                                           set_point=0,
                                           verbose=False)
            """
            Setup the model 
            """
            model_config = SegformerConfig.from_pretrained(pretrained_model)
            model_config.num_labels = num_classes
            model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model,
                config=model_config,
                ignore_mismatched_sizes=True
            )

            try:
                # Get the list of saved checkpoints
                checkpoints = sorted(glob.glob(os.path.join(checkpoint_path, checkpoint_prefix + "*.pt")))
                if checkpoints:
                    latest_checkpoint = checkpoints[-1]
                    cp_manager.load(model, latest_checkpoint, device=device)
                    logger.info(f"Loaded model checkpoint {latest_checkpoint}.")
                else:
                    latest_checkpoint = None
                    logger.info(f"No checkpoints were saved to load in {checkpoint_path}.")
            except FileNotFoundError as e:
                logger.info(f"Unable to load checkpoint {e}")
                latest_checkpoint = None

            model.to(device)
            loss_params = params['loss_function']
            loss_fn = HybridLoss(loss_params['params'])

            # Only pass the parameters that require gradients to the optimizer
            optimizer = torch.optim.AdamW(
                # params=filter(lambda p: p.requires_grad, model.parameters()),
                params=model.parameters(),
                lr=learning_rate,
                # weight_decay=l2_decay_penalty  # L2 regularization to prevent large weights
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['scheduler']['T_max'])

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
                                 scheduler=scheduler,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 eval_loader=eval_loader,
                                 save_preds=False,
                                 save_preds_path="",
                                 config_path=params_file,
                                 verbose=False
                                 )
            train_params = {}
            eval_params = {}

            for epoch in range(n_epochs):
                logger.info(f"Epoch {epoch + 1}/{n_epochs}")

                train_metrics, test_metrics = trainer.train(**train_params)
                train_loss = train_metrics['loss']
                train_iou = train_metrics['iou']
                train_miou = np.mean(train_iou)
                train_dice = train_metrics['dice']
                train_mdice = np.mean(train_dice)

                test_iou = test_metrics['iou']
                test_dice = test_metrics['dice']
                test_precision = test_metrics['precision']
                test_recall = test_metrics['recall']
                logger.info(
                    f"Training Losses  : | Compound: {train_loss:.4f} | Dice: {train_mdice:.4f} | IOU: {train_miou:.4f}")

                val_metrics = trainer.evaluate(**eval_params)
                val_loss = val_metrics['loss']
                val_iou = val_metrics['iou']
                val_miou = np.mean(val_iou)
                val_dice = val_metrics['dice']
                val_mdice = np.mean(val_dice)
                val_precision = val_metrics['precision']
                val_recall = val_metrics['recall']
                logger.info(
                    f"Evaluation Losses: | Compound: {val_loss:.4f} | Dice: {val_mdice:.4f} | IOU: {val_miou:.4f}")

                stop_training, is_saved = cp_manager.save(model,
                                                          val_miou,
                                                          prefix=f"split_{test_split}_fold_{idx}")

                if is_saved:

                    """ Back off the last time we saved so we're only saving the best one """
                    for k,v in metrics.items():
                        metrics[k] = v[:records_offset]
                    records_offset += len(original_names)

                    metrics["case"] += test_names.tolist()
                    metrics["case"] += train_names.tolist()

                    metrics["is_test"] += [1] * len(test_names)
                    metrics["is_test"] += [0] * len(train_names)

                    metrics["test_split"] += [test_split] * n_records
                    metrics["test_iteration"] += [idx] * n_records

                    metrics["dice"] += val_dice + test_dice
                    metrics["iou"] += val_iou + test_iou
                    metrics["precision"] += val_precision + test_precision
                    metrics["recall"] += val_recall + test_recall
                    logger.info(f"Saving model: split: {test_split}, fold: {idx+1}")
                    assert check_scores(metrics), "Scores don't match!"

                if stop_training:
                    logger.info(f"Training stopped early at epoch {epoch} with mIOU Score: {val_miou:.4f}")
                    break

    try:
        pd.DataFrame(metrics).to_csv(f"classica_evaluate_base_segformer_metrics_{timestamp}.csv")
    except Exception as e:
        logger.error(f"Error saving scores to CSV. The following exception was detected:{e}")


if __name__ == "__main__":
    # --- Logging Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    home_dir = Path.home()
    if not os.path.exists(os.path.join(home_dir, "segmenter")):
        os.makedirs(os.path.join(home_dir, "segmenter"))
    logfile = os.path.join(home_dir, "segmenter", f"training_base_segformer_{timestamp}.log")

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
