import glob
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmenter.loss import IoULoss
from segmenter.loss.hybrid import HybridLoss
from segmenter.models.models import AugurSegformerSegmentation
from segmenter.torch_utils import RunManager, CheckpointManager
from segmenter.utils.data import (get_num_samples_from_hdf5, HDF5ImageDataset, hdf5_worker_init_fn,
                                  HDF5DatasetOptimized, HDF5BatchSampler, SSLTransformPipeline)


def check_scores(metric:dict[str,list])-> bool:
    all_lens = np.array([len(v) for v in metric.values()])
    base_len = all_lens[0]
    if not np.all(all_lens == base_len):
        for k,v in metric.items():
            print(f"{k}: {len(v)}")
    return np.all(all_lens == base_len)


class EpochStopper:
    def __init__(self, max_boredom: int = 5, min_delta: float = 0.0001):

        self.best_loss = float('inf')
        self.min_delta = min_delta
        self.max_boredom = max_boredom
        self.current_boredom = 0

    def __call__(self, epoch: int, score: np.floating[Any]) -> bool:
        """ Set up stopping criteria - stop after 'boredom' steps do not improve loss by 'min_delta' """
        is_stopping = False
        if score + self.min_delta < self.best_loss:
            self.best_loss = score
            self.boredom = 0
        else:
            self.boredom += 1
        if self.boredom >= self.max_boredom:
            is_stopping = True
        return is_stopping

    forward = __call__



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
                                   "classica_params.json")):
        params_file = os.path.join(os.path.join(home, "segmenter"),
                                   "classica_params.json")
    else:
        params_file = "classica_params.json"

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
    pretrained_checkpoint = model_params['pretrained_checkpoint']

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
    num_workers = min(run_params['num_workers'], 1)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logger.info(f"Using {device} device for model training.")

    """ Load datasets for test and training """
    # test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    test_sizes = [0.1]
    n_folds = 5
    metrics = {"case": [],
               "test_split": [],
               "test_iteration": [],
               "is_test": [],
               "dice": [],
               "iou": [],
               "precision": [],
               "recall": []}

    results_csv_path = os.path.join(Path.home(), "segmenter")
    results_csv_name = os.path.join(results_csv_path, "classica_custom_segformer_results.csv")
    results = pd.DataFrame.from_dict(metrics)
    results.to_csv(results_csv_name)

    with h5py.File(hdf5_file, 'r', swmr=True) as hdf:
        original_names_hdf = hdf['original_name']
        original_names = np.array([h.decode('utf-8') for h in original_names_hdf])
        n_records = len(original_names)

    ds = HDF5DatasetOptimized(hdf5_path=hdf5_file,
                              transform=SSLTransformPipeline(size=image_size))



    records_offset = 0
    for test_split in test_sizes:
        n_test = int(n_records * test_split)
        n_train = int(n_records - n_test)
        logger.info(f"Testing split {test_split} for {n_folds} iterations.")

        for idx, fold in enumerate(range(n_folds)):

            batch_sampler = HDF5BatchSampler(ds.dataset_len,
                                             batch_size=1,
                                             shuffle=True)

            dataloader = torch.utils.data.DataLoader(ds,
                                                     batch_size=None,
                                                     sampler=batch_sampler,
                                                     shuffle=False,
                                                     num_workers=num_workers,
                                                     worker_init_fn=hdf5_worker_init_fn
                                                     )

            all_data = [d for d in dataloader]


            model = AugurSegformerSegmentation(pretrained_model=pretrained_model,
                                               checkpoint_path=pretrained_checkpoint,
                                               num_classes=num_classes).to(device)

            model.to(device=device)

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


            logger.info(f"Starting fold [{idx + 1}/{n_folds}] for test split [{test_split}].")
            stopper = EpochStopper(max_boredom=3, min_delta=0.01)
            train_names = []
            test_names = []
            for epoch_idx, epoch in enumerate(range(n_epochs)):
                logger.info(f"Epoch [{epoch_idx + 1}/{n_epochs}].")
                total_epoch_train_loss = []
                total_epoch_test_loss = []
                iou_epoch_train_loss = []
                iou_epoch_test_loss = []

                for data in all_data[:n_train]:
                    model.train()

                    x = {}
                    for key, value in data.items():
                        print(key)
                        if key in ['images', 'anchors', 'targets', 'local_targets', 'masks']:
                            x[key] = data[key].to(device)
                        elif key == 'original_name':
                            x[key] = [d.decode('utf-8') for d in data[key]]
                        else:
                            x[key] = data[key]

                    train_names += [x_k for x_k in x['original_name'] if x_k not in train_names]
                    logger.info(f"Epoch [{epoch_idx + 1}/{n_epochs}]. Training with {len(train_names)} names.")

                    optimizer.zero_grad()

                    mask_out = model(x['images'])
                    loss_train = loss_fn(mask_out, x['masks'])

                    loss_train.backward()

                    optimizer.step()

                    scheduler.step()

                    total_epoch_train_loss += [loss_train.item()]
                    with torch.no_grad():
                        iou_epoch_train_loss += [IoULoss()(mask_out, x['masks']).item()]

                    b, _, _, _ = mask_out.shape

                for data in all_data[n_train:]:
                    model.eval()
                    x = {}
                    for key, value in data.items():
                        print(key)
                        if key in ['images', 'anchors', 'targets', 'local_targets', 'masks']:
                            x[key] = data[key].to(device)
                        elif key == 'original_name':
                            x[key] = [d.decode('utf-8') for d in data[key]]
                        else:
                            x[key] = data[key]

                    test_names += [x_k for x_k in x['original_name'] if x_k not in test_names]

                    with torch.no_grad():
                        mask_out = model(x['images'])
                        loss_test = loss_fn(mask_out, x['masks'])

                        total_epoch_test_loss += [loss_test.item()]

                        iou_epoch_test_loss += [IoULoss()(mask_out, x['masks']).item()]

                scheduler.step()
                logger.info(f"Epoch {epoch + 1}/{n_epochs} completed. "
                            f"Training Total Loss: {np.mean(total_epoch_train_loss):.4f} "
                            f"Training IoU Loss: {np.mean(iou_epoch_train_loss):.4f} "
                            f"Test Total Loss: {np.mean(total_epoch_test_loss):.4f} "
                            f"Test IoU Loss: {np.mean(iou_epoch_test_loss):.4f} ")
                if stopper.forward(epoch=epoch, score=np.mean(total_epoch_train_loss)):
                    logger.info(f"Epoch {epoch + 1}/{n_epochs} completed. ")
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
