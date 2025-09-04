"""
Utilities to help with PyTorch
"""
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch import autocast
from tqdm import tqdm

from segmenter.modules import LossFactory, HybridLoss

# Pre-define a mapping of class names to their actual classes
# This avoids needing a separate factory file
MODEL_MAP = {
    "SegformerForSemanticSegmentation": None  # Will be imported on use
}
OPTIMIZER_MAP = {
    "AdamW": torch.optim.AdamW
}
CRITERION_MAP = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "HybridLoss": HybridLoss
}
SCHEDULER_MAP = {
    "LambdaLR": torch.optim.lr_scheduler.LambdaLR
}

def get_default_device_type() -> str:
    """
    Pick GPU if available, else CPU
    Chooses MPS for Apple MPS devices, or CUDA device if available
    """
    # _device = "cpu"
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"  # For Apple devices with MPS support
    else:
        _device = "cpu"
    return _device

def get_default_device() -> torch.device:
    return torch.device(get_default_device_type())


def set_default_device(device: torch.device):
    if device.type == "cuda":
        torch.set_default_dtype(torch.float16)
    elif device.type == "mps" or device.type == "cpu":
        torch.set_default_dtype(torch.float32)

    if torch.amp.autocast_mode.is_autocast_available(device.type):
        torch.autocast(device.type,
                       dtype=torch.bfloat16).__enter__()
    return

"""
class TrainingManager

Wraps train() and evaluate() methods 

Models, optimizers etc. are instantiated first and then passed as object instances to 
a TrainingManager instance

"""
class RunManager:
    def __init__(self,
                 model=None,
                 optimizer=None,
                 criterion=None,
                 scaler=None,
                 scheduler=None,
                 train_loader=None,
                 eval_loader=None,
                 save_preds=False,
                 save_preds_path=None,
                 device='cpu'):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        if self.model is None:
            raise ValueError('Please provide a valid model in TrainingManager')

        self.device = next(self.model.parameters()).device
        self.dice_loss_fn = LossFactory.create('dice')
        self.iou_loss_fn = LossFactory.create('iou')

        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.scheduler = scheduler

        self.train_loader = train_loader

        if eval_loader is None and train_loader is None:
            raise ValueError('Please provide at least one valid training or validation data loader')
        self.eval_loader = eval_loader

        if save_preds is True and save_preds_path is not None:
            self.save_preds = save_preds
            self.save_preds_path = save_preds_path
        else:
            self.save_preds = False

        # self.dice_loss = DL(mode='binary')
        # self.iou_loss = JL(mode='binary')


    def train(self, **train_params: object) -> dict[str, float]:
        """
        Trains one epoch using the data provided in self.train_loader
        :return: total loss and dice score
        """

        self.model.train()
        total_loss = []
        total_dice_loss = []
        total_iou_loss = []
        total_metrics = {'loss': 0.0, 'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

        for images, masks in tqdm(self.train_loader, colour='green'):

            if images.device != self.device:
                images = images.to(self.device)

            if masks.device != self.device:
                masks = masks.to(self.device)


            with autocast(device_type=get_default_device_type(), dtype=torch.float16):
                logits = self.model(pixel_values=images) # logits; [B, num_classes, H, W]

                loss = self.criterion(logits, masks) # Mask: [B, H, W]
                total_loss += [loss.item()]
                total_dice_loss += [self.dice_loss_fn(logits, masks.squeeze(1)).item()]
                total_iou_loss += [self.iou_loss_fn(logits, masks.squeeze(1)).item()]

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward() # Fails on MPS, works on CPU/CUDA
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        total_metrics['loss'] = np.mean(total_loss)
        total_metrics['dice'] = np.mean(total_dice_loss)
        total_metrics['iou'] = np.mean(total_iou_loss)

        return total_metrics

    def evaluate(self, **eval_params: object) -> dict[str, float]:
        """
        Evaluate using the data provided in self.eval_loader
        :return: total loss and dice score
        """

        self.model.eval()
        total_loss = []
        total_dice_loss = []
        total_iou_loss = []
        total_metrics = {'loss': 0.0, 'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

        with torch.no_grad():
            for images, masks in tqdm(self.eval_loader, colour='yellow'):
                if images.device != self.device:
                    images = images.to(self.device)

                if masks.device != self.device:
                    masks = masks.to(self.device)

                with autocast(device_type=get_default_device_type(), dtype=torch.float16):
                    logits = self.model(pixel_values=images)
                    loss = self.criterion(logits, masks)

                    total_dice_loss += [self.dice_loss_fn(logits, masks.squeeze(1)).item()]
                    total_iou_loss += [self.iou_loss_fn(logits, masks.squeeze(1)).item()]
                    total_loss += [loss.item()]

                if self.save_preds is True and self.save_preds_path is not None:
                    print("Saving predictions is not implemented yet")

            total_metrics['loss'] = np.mean(total_loss)
            total_metrics['dice'] = np.mean(total_dice_loss)
            total_metrics['iou'] = np.mean(total_iou_loss)

        return total_metrics


class CheckpointError(Exception):
    """
    Custom exception for errors related to loading or saving CSV files
    within the CSVHandler class.
    """

    def __init__(self, message="An error occurred with the CSV file operation."):
        self.message = message
        super().__init__(self.message)


class CheckpointManager:
    """
    A class to manage PyTorch model checkpoints with built-in early stopping.

    This class handles saving the model state when validation accuracy improves and
    provides a mechanism to stop training if performance plateaus.

    Args:
        checkpoint_dir (str): The directory where checkpoints will be saved.
        prefix (str): A string prefix for checkpoint filenames.
        patience (int): The number of epochs to wait for improvement before stopping.
        min_delta (float): The minimum change in accuracy to qualify as an improvement.
        warm_start (bool): Whether or not to start training from scratch or load the last checkpoint
    """

    def __init__(self, checkpoint_dir: str,
                 prefix: str ="model_checkpoint",
                 patience: int =5,
                 min_delta: float =0.0,
                 warm_start: bool =False):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Ensure the checkpoint directory exists and if not, revert to local
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir)
            except OSError as e:
                self.logger.error(f"Error creating checkpoint directory {checkpoint_dir}: {e}")
                self.checkpoint_dir = os.getcwd()
                self.logger.info(f"Reverting to current working directory for checkpoint: {self.checkpoint_dir}")

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prefix = prefix
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = float('-inf')  # Initialize with a very low value
        self.epochs_without_improvement = 0
        self.stop_training = False

    def save(self, model, current_accuracy) -> bool:
        """
        Saves the model checkpoint if the current accuracy is the best seen so far.

        This method also updates the internal state for early stopping.

        Args:
            model (torch.nn.Module): The PyTorch model to save.
            current_accuracy (float): The current validation accuracy.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if current_accuracy > self.best_accuracy + self.min_delta:
            # New best accuracy found, save the model and reset the counter
            self.best_accuracy = current_accuracy
            self.epochs_without_improvement = 0

            # Generate a timestamp for the filename
            filename = f"{self.prefix}_{self.timestamp}.pt"
            filepath = os.path.join(self.checkpoint_dir, filename)

            json_filename =  f"{self.prefix}_{self.timestamp}.json"
            json_filepath = os.path.join(self.checkpoint_dir, json_filename)
            json_data = {"best_accuracy": current_accuracy,
                         "timestamp": self.timestamp,
                         "patience": self.patience,
                         "epochs_without_improvement": self.epochs_without_improvement}

            # Save the model's state dictionary
            torch.save(model.state_dict(), filepath)
            # Save the model's current best_accuracy for warm restart
            with open(json_filepath, 'w') as fp:
                json.dump(json_data, fp, sort_keys=True, indent=4)

            self.logger.info(f"Checkpoint saved: {filepath} with accuracy: {current_accuracy:.4f}")
        else:
            # No significant improvement, increment the counter
            self.epochs_without_improvement += 1
            self.logger.info(f"No improvement. Epochs without improvement: {self.epochs_without_improvement}")

        # Check if the patience limit has been reached
        if self.epochs_without_improvement >= self.patience:
            self.stop_training = True
            self.logger.info(f"Early stopping triggered. Training will be stopped after this epoch.")

        return self.stop_training

    def load(self, model:torch.nn.Module, filename: str,
             device: torch.device =torch.device("cpu")) -> torch.nn.Module:
        """
        Loads a model's state from a checkpoint file.

        Args:
            model (torch.nn.Module): The PyTorch model instance to load the state into.
            filename (str): The name of the checkpoint file to load.
            device (torch.device): The device on which to load the checkpoint file.
        Returns:
            torch.nn.Module: The model with the loaded state.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")


        json_filepath = filepath.replace(".pt", ".json")
        try:
            with open(json_filepath, 'r') as fp:
                json_data  = json.load(fp)
                self.best_accuracy = json_data["best_accuracy"]
                try:
                    self.patience = json_data["patience"]
                    self.epochs_without_improvement = json_data["epochs_without_improvement"]
                    self.timestamp = json_data["timestamp"]
                except KeyError:
                    pass
                self.logger.info(f"Warm start with loss: {self.best_accuracy}")
        except FileNotFoundError:
            self.logger.warning(f"Checkpoint JSON configuration file not found: {filepath}")

        # Load the state dictionary and apply it to the model
        model.load_state_dict(torch.load(filepath,
                                         # map_location=device,
                                         map_location=next(model.parameters()).device,
                                         weights_only=False))
        self.logger.info(f"Checkpoint loaded successfully from: {filepath}")
        return model

    """ Getters and Setters """
    def get_checkpoint_dir(self):
        return self.checkpoint_dir

    def get_timestamp(self):
        return self.timestamp

    def get_prefix(self):
        return self.prefix

    def get_patience(self):
        return self.patience

    def get_min_delta(self):
        return self.min_delta

    def set_patience(self, patience):
        self.patience = patience

    def set_min_delta(self, min_delta):
        self.min_delta = min_delta
