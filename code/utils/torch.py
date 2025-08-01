"""
Utilities to help with PyTorch
"""
import torch
from torch import device

from nn.models import SegformerBinarySegmentation

from nn.modules import CombinedLoss, DiceScore
from tqdm import tqdm


def get_default_device() -> device:
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
    return torch.device(_device)


def set_default_device(device):
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
                 train_loader=None,
                 eval_loader=None,
                 device='cpu'):
        self.device = device
        if self.device is None:
            self.device = 'cpu'  # Default to the safest option

        self.model = model
        if self.model is None:
            self.model = SegformerBinarySegmentation().to(device)

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = CombinedLoss()

        self.scaler = scaler

        if train_loader is None:
            raise ValueError('Invalid data loader in train_loader parameter. Please provide a valid data loader')
        self.train_loader = train_loader

        if eval_loader is None:
            raise ValueError('Invalid data loader in eval_loader parameter. Please provide a valid data loader')
        self.eval_loader = eval_loader

        self.dice_score = DiceScore()

    def train(self):
        """
        Trains one epoch using the data provided in self.loader
        :return: total loss and dice score
        """
        self.model.train()
        total_loss = 0
        total_dice = 0

        for images, masks in tqdm(self.train_loader):

            images = images.to(self.device)
            masks = masks.clone().detach().requires_grad_(True).to(self.device)

            # with autocast(device_type=self.device):
            #     logits = self.model(pixel_values=images)
            #     loss = self.criterion(logits, masks)
            #     dice = self.dice_score(logits, masks).detach().item()
            logits = self.model(pixel_values=images)
            loss = self.criterion(logits, masks)
            dice = self.dice_score(logits, masks)
            total_loss += loss.item()
            total_dice += dice.item()

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return total_loss, total_dice

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_dice_score = 0
        total_metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}

        with torch.no_grad():
            for images, masks in self.eval_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                logits = self.model(pixel_values=images)
                loss = self.criterion(logits, masks)

                total_dice_score += self.dice_score(logits, masks).item()
                total_loss += loss.item()

        total_metrics['dice'] = total_dice_score

        # for k in total_metrics:
        #     total_metrics[k] += metrics[k]
        #     avg_loss = total_loss / len(loader)
        #     avg_metrics = {k: total_metrics[k] / len(loader) for k in total_metrics}
        # print(
        #     f"Val Loss: {avg_loss:.4f} | Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f}")
        # return avg_loss, avg_metrics
        return total_loss, total_metrics
