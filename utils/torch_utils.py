"""
Utilities to help with PyTorch
"""
import torch
from torch import autocast

from nn.modules import CombinedLoss
from losses import (JaccardLoss as JL, DiceLoss as DL)
from tqdm import tqdm


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
class TrainingManager:
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


        self.model = model
        if self.model is None:
            raise ValueError('Please provide a valid model in TrainingManager')

        self.device = next(self.model.parameters()).device

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = CombinedLoss()

        self.scaler = scaler
        self.scheduler = scheduler

        if train_loader is None:
            raise ValueError('Invalid data loader in train_loader parameter. Please provide a valid data loader')
        self.train_loader = train_loader

        if eval_loader is None:
            raise ValueError('Invalid data loader in eval_loader parameter. Please provide a valid data loader')
        self.eval_loader = eval_loader

        if save_preds is True and save_preds_path is not None:
            self.save_preds = save_preds
            self.save_preds_path = save_preds_path
        else:
            self.save_preds = False

        self.dice_loss = DL(mode='binary')
        self.iou_loss = JL(mode='binary')

    def train(self, **train_params: object) -> dict[str, float]:
        """
        Trains one epoch using the data provided in self.train_loader
        :return: total loss and dice score
        """
        # TODO: implement parameters
        self.model.train()
        total_loss = 0.0
        total_dice_loss = 0.0
        total_iou_loss = 0.0
        total_metrics = {'loss': 0.0, 'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

        for images, masks in tqdm(self.train_loader):

            if images.device != self.device:
                images = images.to(self.device)

            if masks.device != self.device:
                masks = masks.to(self.device)

            n_batch = images.shape[0]
            with autocast(device_type=get_default_device_type(), dtype=torch.float16):
                logits = self.model(pixel_values=images)
                total_loss += self.criterion(logits, masks.float()).item()
                total_dice_loss += self.dice_loss(logits, masks.float()).item()
                total_iou_loss += self.iou_loss(logits, masks.float()).item()

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(self.criterion).backward() # Fails on MPS, works on CPU/CUDA
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.criterion.backward() # Fails on MPS, works on CPU/CUDA
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        total_metrics['loss'] = total_loss
        total_metrics['dice'] = total_dice_loss
        total_metrics['iou'] = total_iou_loss

        return total_metrics

    def evaluate(self, **eval_params: object) -> dict[str, float]:
        """
        Evaluate using the data provided in self.eval_loader
        :return: total loss and dice score
        """
        # TODO: implement parameters
        self.model.eval()
        total_loss = 0.0
        total_dice_loss = 0.0
        total_iou_loss = 0.0
        total_metrics = {'loss': 0.0, 'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

        with torch.no_grad():
            for images, masks in self.eval_loader:
                if images.device != self.device:
                    images = images.to(self.device)

                if masks.device != self.device:
                    masks = masks.to(self.device)
                n_batch = images.shape[0]
                with autocast(device_type=get_default_device_type(), dtype=torch.float16):
                    logits = self.model(pixel_values=images)
                    loss = self.criterion(logits, masks)

                if self.save_preds is True and self.save_preds_path is not None:
                    # TODO: implement saving later
                    print(logits.shape, logits.max(), logits.min())

                total_dice_loss += self.dice_loss(logits, masks.float()).item()
                total_iou_loss += self.iou_loss(logits, masks.float()).item()
                total_loss += loss.item()

        total_metrics['loss'] = total_loss
        total_metrics['dice'] = total_dice_loss
        total_metrics['iou'] = total_iou_loss

        return total_metrics
