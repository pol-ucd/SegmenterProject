"""
Utilities to help with PyTorch
"""
import importlib
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch import autocast, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from segmenter.modules import LossFactory, HybridLoss, ImageLightingAugmentation

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
class RunManager

Manages the training and evaluation loop for a PyTorch model.

    This class is parameterized via a JSON configuration file, dynamically
    instantiating the model, optimizer, criterion, and data loaders.

"""
class RunManager:
    def __init__(self,
                 model=None,
                 optimizer=None,
                 criterion=None,
                 scaler=None,
                 scheduler=None,
                 train_loader=None,
                 test_loader=None,
                 eval_loader=None,
                 save_preds=False,
                 save_preds_path=None,
                 config_path: str = None,
                 verbose=False):
        self.device = get_default_device()
        self.config = self._load_config(config_path)

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
        self.test_loader = test_loader

        if eval_loader is None and train_loader is None:
            raise ValueError('Please provide at least one valid training or validation data loader')
        self.eval_loader = eval_loader

        if save_preds is True and save_preds_path is not None:
            self.save_preds = save_preds
            self.save_preds_path = save_preds_path
        else:
            self.save_preds = False
        self.verbose = verbose

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)

    def _scores(self, predicted_mask, expected_mask):
        metrics = {"dice": [],
                   "iou": [],
                   "precision": [],
                   "recall": []}
        b, _, _ = predicted_mask.shape

        predicted_mask = predicted_mask.cpu().detach().numpy()
        expected_mask = expected_mask.cpu().detach().numpy()
        for batch in range(b):
            predicted = predicted_mask[batch].reshape(-1)
            expected = expected_mask[batch].reshape(-1)

            metrics['dice'].append(f1_score(expected, predicted, average='macro', zero_division=0))
            metrics['iou'].append(jaccard_score(expected, predicted, average='macro', zero_division=0))
            metrics['precision'].append(precision_score(expected, predicted, average='macro', zero_division=0))
            metrics['recall'].append(recall_score(expected, predicted, average='macro', zero_division=0))

        return metrics

    def train(self, **train_params: object) -> tuple[dict[str, list], dict[str, list]]:
        """
        Trains one epoch using the data provided in self.train_loader

        aggr
        :return: total loss and dice score
        """
        light_aug = ImageLightingAugmentation().to(self.device)

        self.model.train()
        total_loss = []
        total_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}
        test_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}

        if self.train_loader is not None:
            for images, masks in self.train_loader:

                if images.device != self.device:
                    images = images.to(self.device)

                if masks.device != self.device:
                    masks = masks.to(self.device)

                images = light_aug(images)

                with autocast(device_type=get_default_device_type(), dtype=torch.float16):
                    # logits = self.model(pixel_values=images) # logits; [B, num_classes, H, W]
                    if isinstance(self.model, SegformerForSemanticSegmentation):
                        outputs_tuple = self.model(pixel_values=dummy_input, return_dict=False)
                        logits = outputs_tuple[0]
                    else:
                        logits = self.model(pixel_values=images)

                    loss = self.criterion(logits, masks) # Mask: [B, H, W]
                    total_loss += [loss.item()/logits.shape[0]]
                    pred_masks = F.softmax(logits,
                                           dim=1).argmax(dim=1)
                    exp_masks = masks.argmax(dim=1)
                    b_m = self._scores(pred_masks, exp_masks)
                    for k, v in total_metrics.items():
                        v += b_m[k]

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

        total_metrics['loss'] = np.mean(total_loss) if total_loss else None

        if self.test_loader is not None:
            test_metrics = self.predict(self.test_loader, **train_params)

        return total_metrics, test_metrics

    def evaluate(self, **eval_params: object) -> dict[str, list]:
        """
        Evaluate using the data provided in self.eval_loader
        :return: total loss and dice score
        """
        return self.predict(self.eval_loader, **eval_params)

    def predict(self, loader: DataLoader, **predict_params: object) -> dict[str, list]:

        self.model.eval()
        total_loss = []
        total_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}

        if self.eval_loader is not None:
            with torch.no_grad():
                for images, masks in loader:
                    if images.device != self.device:
                        images = images.to(self.device)

                    if masks.device != self.device:
                        masks = masks.to(self.device)

                    with autocast(device_type=get_default_device_type(),
                                  dtype=torch.float16):

                        if isinstance(self.model, SegformerForSemanticSegmentation):
                            outputs_tuple = self.model(pixel_values=dummy_input, return_dict=False)
                            logits = outputs_tuple[0]
                        else:
                            logits = self.model(pixel_values=images)
                        loss = self.criterion(logits, masks)
                        total_loss += [loss.item()/logits.shape[0]]

                        pred_masks = F.softmax(logits,
                                               dim=1).argmax(dim=1)
                        exp_masks = masks.argmax(dim=1)
                        b_m = self._scores(pred_masks, exp_masks)
                        for k, v in total_metrics.items():
                            v += b_m[k]

                    if self.save_preds is True and self.save_preds_path is not None:
                        print("Saving predictions is not implemented yet")

        total_metrics['loss'] = np.mean(total_loss) if total_loss else None

        return total_metrics


def get_module_class(class_name: str) -> nn.Module:
    """
    Parse a string name of a module.class and return the module and class
    as a tuple after checking everything is valid.
    :param class_name: a string name of a module.class e.g segmenter.models.AugurSegformerSegmentation
    :return: the class ( with the importlib already performed so the class can be directly instantiated)
    """
    module_name, class_name = class_name.rsplit('.', maxsplit=1)
    try:
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module {module_name} not found")
    return class_


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
                 patience: int = 5,
                 min_delta: float = 0.0,
                 set_point: float = 1.0,
                 verbose=False):
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
        self.set_point = set_point  # Restart at this level
        self.best_accuracy = float('-inf')  # Initialize with a very low value
        self.epochs_without_improvement = 0
        self.stop_training = False
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Checkpoint manager loaded with prefix: {self.prefix} and timestamp: {self.timestamp}")

    def save(self, model, current_accuracy, prefix:str=None) -> tuple[bool, bool]:
        """
        Saves the model checkpoint if the current accuracy is the best seen so far.

        This method also updates the internal state for early stopping.

        Args:
            model (torch.nn.Module): The PyTorch model to save.
            current_accuracy (float): The current validation accuracy.

        Returns:
            bool: True if training should stop, False otherwise.
            bool: True if model was saved on this step, False otherwise.
        """
        is_saved = False
        if current_accuracy > self.best_accuracy + self.min_delta:
            # New best accuracy found, save the model and reset the counter
            self.best_accuracy = current_accuracy
            self.epochs_without_improvement = 0

            # Generate a timestamp for the filename
            filename = f"{self.prefix}_{self.timestamp}.pt"
            if prefix is not None:
                filename = f"{prefix}_{filename}"
            filepath = os.path.join(self.checkpoint_dir, filename)

            json_filename =  f"{self.prefix}_{self.timestamp}.json"
            if prefix is not None:
                json_filename = f"{prefix}_{json_filename}"
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
            if self.verbose:
                self.logger.info(f"Checkpoint saved: {filepath} with score: {current_accuracy:.4f}")
            is_saved = True
        else:
            # No significant improvement, increment the counter
            self.epochs_without_improvement += 1
            if self.verbose:
                self.logger.info(f"No improvement. Epochs without improvement: {self.epochs_without_improvement}")

        # Check if the patience limit has been reached
        if self.epochs_without_improvement >= self.patience:
            self.stop_training = True
            if self.verbose:
                self.logger.info(f"Early stopping triggered. Training will be stopped after this epoch.")

        return self.stop_training, is_saved

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
        # filepath = os.path.join(self.checkpoint_dir, filename)
        # if not os.path.exists(filepath):
        #     raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        json_filename = filename.replace(".pt", ".json")
        try:
            with open(json_filename, 'r') as fp:
                json_data  = json.load(fp)
                self.best_accuracy = min(json_data["best_accuracy"], self.set_point)
                try:
                    # self.patience = json_data["patience"]
                    # self.epochs_without_improvement = json_data["epochs_without_improvement"]
                    self.timestamp = json_data["timestamp"]
                except KeyError:
                    pass
                if self.verbose:
                    self.logger.info(f"Warm start with loss: {self.best_accuracy}")
        except FileNotFoundError:
            self.logger.warning(f"Checkpoint JSON configuration file not found: {json_filename}")

        if self.verbose:
            self.logger.info(f"Loading model parameters from checkpoint {filename}")
        # Load the state dictionary and apply it to the model
        model.load_state_dict(torch.load(filename,
                                         # map_location=device,
                                         map_location=next(model.parameters()).device,
                                         weights_only=False))
        if self.verbose:
            self.logger.info(f"Checkpoint loaded successfully from: {filename}")
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
