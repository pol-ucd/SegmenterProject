import argparse

import numpy as np
import torch
from torch import GradScaler

from losses import (DiceLoss as DL)
from nn.data import data_load
from nn.models import SegformerBinarySegmentation4
from utils.torch_utils import TrainingManager, get_default_device


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=False, save_path=None):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' for loss, 'max' for accuracy or IoU.
            verbose (bool): If True, prints updates.
            save_path (str): If set, saves best model to this path.
        """
        assert mode in ['min', 'max'], "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = None

        self._init_comparator()

    def _init_comparator(self):
        if self.mode == 'min':
            self.compare = lambda current, best: current < best - self.min_delta
            self.best_score = np.inf
        else:
            self.compare = lambda current, best: current > best + self.min_delta
            self.best_score = -np.inf

    def __call__(self, current_score, model=None, epoch=None):
        if self.compare(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                print(f"New best score: {current_score:.4f} at epoch {epoch}")
            if self.save_path and model is not None:
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"Model saved to {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = None


# TODO: this function needs to be reworked. Ignoring it for now.

def validate_positive_integer(value):
    """
    Custom type function for argparse to ensure an integer is greater than zero.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"'{value}' is an invalid positive integer value. Must be greater than zero.")
    return ivalue


def validate_01_float(value):
    """
    Custom type function for argparse to ensure a float is between 0 and 1.0
    """
    fvalue = float(value)
    if fvalue <= 0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(f"'{value}' is an invalid float value. Must be in the range [0, 1].")
    return fvalue


def process_args():
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "--train_path",
        type=str,
        nargs='?',
        default="data/Polyp Segmentation/train",
        help="The path to the training data."
    )

    parser.add_argument(
        "--val_path",
        type=str,
        nargs='?',
        default="data/Polyp Segmentation/valid",
        help="The path to the validation data."
    )

    parser.add_argument(
        "--n_epochs",
        type=validate_positive_integer,
        nargs="?",  # Makes the argument optional
        default=4,  # Sets the default value if not provided
        help="The number of training epochs to run (default: 4)."
    )

    parser.add_argument(
        "--n_batch",
        type=validate_positive_integer,
        nargs="?",  # Makes the argument optional
        default=4,  # Sets the default value if not provided
        help="The number of records per mini batch for training and validation (default: 4)."
    )

    parser.add_argument(
        "--test_split",
        type=validate_01_float,
        nargs="?",  # Makes the argument optional
        default=0.3,  # Sets the default value if not provided
        help="The fraction of records to hold back for validation (default: 0.3)."
    )

    # Parse the arguments from the command line
    return parser.parse_args()


def main():
    args = process_args()

    print(args)

    # Set the default device to the best available GPU ... or CPU if no GPU available
    device = get_default_device()
    device='cpu'
    args.n_epochs = 1
    print(f"Using {device} device for model training.")

    """
    I've implemented a data_load function that
    can generate a train/test split if needed - but for now I'm just taking 100% 
    of the training and 100% validation data and using them to train and then to 
    validate respectively.
    """
    (train_loader,
     _) = data_load(args.train_path,
                    # test_split=args.test_split,
                    test_split=0.0,  # Use 100% for training
                    batch_size=args.n_batch,
                    verbose=True)

    (_,
     val_loader) = data_load(args.val_path,
                             # test_split=args.test_split,
                             test_split=1.0,  # Use 100% for testing/validation
                             batch_size=args.n_batch,
                             verbose=True)

    n_val = len(val_loader)
    n_train = len(train_loader)

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(val_loader)}")

    pretained_model = 'nvidia/segformer-b4-finetuned-ade-512-512'
    # model = SegformerBinarySegmentation().to(device)  #Old Word doc model
    model = SegformerBinarySegmentation4(pretrained_model=pretained_model, num_classes=1).to(device)
    # loss_fn = CombinedLoss()
    loss_fn = DL(mode='binary')


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    """
    Only use GradScaler if we have CUDA
    """
    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()

    trainer = TrainingManager(model,
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
    best_dice_loss = 1.0

    early_stopper = EarlyStopping(patience=7, min_delta=0.001, mode='max', verbose=True,
                                  save_path="best_model_classica.pt")

    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        print()
        train_metrics = trainer.train(**train_params)
        val_metrics = trainer.evaluate(**eval_params)
        train_loss = train_metrics['loss'] / n_train
        train_miou = train_metrics['dice'] / n_train
        train_dice = train_metrics['dice'] / n_train
        val_loss = val_metrics['loss'] / n_val
        val_miou = val_metrics['dice'] / n_val
        val_dice = val_metrics['dice'] / n_val

        print(
            f"Training Losses: | Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IOU: {train_miou:.4f}")
        print(
            f"Evaluation Losses: | Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IOU: {val_miou:.4f}")
        print()

        scheduler.step(epoch + 1)

        early_stopper(val_miou, model, epoch)

        if early_stopper.early_stop:
            print(f"Training stopped early at epoch {epoch}")
            break


if __name__ == "__main__":
   main()