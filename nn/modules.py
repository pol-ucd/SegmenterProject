import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import TverskyLoss as TL, FocalLoss as FL, DiceLoss, JaccardLoss


class HybridLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0,
                 weight_focal=1.0, weight_tversky=1.0):
        super(HybridLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_tversky = weight_tversky
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        loss_ce = self.ce_loss(pred, target)
        loss_dice = dice_loss(pred, target)
        loss_focal = focal_loss(pred, target)
        loss_tversky = tversky_loss(pred, target, alpha=0.2, beta=0.4, smooth=1e-6)

        total_loss = (
            self.weight_ce * loss_ce +
            self.weight_dice * loss_dice +
            self.weight_focal * loss_focal +
            self.weight_tversky * loss_tversky
        )
        return total_loss


def dice_loss(pred, target, epsilon=1e-6):
    pred = torch.softmax(pred, dim=1)  # [B, C, H, W]
    target_onehot = F.one_hot(target.squeeze(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target.squeeze(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    pt = torch.where(target_onehot == 1, pred, 1 - pred)
    focal_term = alpha * (1 - pt) ** gamma
    bce = -torch.log(pt + 1e-6)
    return (focal_term * bce).mean()


def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1e-6):
    probs = torch.softmax(pred, dim=1)
    target_oh = F.one_hot(target.squeeze(), num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
    TP = (probs * target_oh).sum(dim=(0, 2, 3))  # [C]
    FP = (probs * (1 - target_oh)).sum(dim=(0, 2, 3))  # [C]
    FN = ((1 - probs) * target_oh).sum(dim=(0, 2, 3))  # [C]

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss_tversky = 1.0 - tversky
    return loss_tversky.mean()

def iou_loss(pred, target, epsilon=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target.squeeze(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred + target_onehot - (pred * target_onehot)
    union = union.sum(dim=(2, 3))
    iou = (intersection + epsilon) / (union + epsilon)
    return 1 - iou.mean()


"""
Implements Hanija's Combined Loss for Binary image classification.
Is also callable so it can be used to evaluate loss with no_grad()
"""

class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super(CombinedLoss, self).__init__()
        if weights is None:
            weights = {'bce': 0.2, 'tversky': 0.4, 'focal': 0.4, 'dice': 0.6, 'jaccard': 0.6}
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TL(alpha=0.2, beta=0.4, mode='binary')
        self.focal = FL(mode='binary')
        self.dice = DiceLoss(mode='binary')
        self.iou = JaccardLoss(mode='binary')

    def forward(self, pred, target):
        return self._do_calculation(pred, target)

    def _do_calculation(self, pred, target):
        bce = self.bce(pred, target.unsqueeze(1).float())
        pred = pred.transpose(3, 1)
        tversky = self.tversky(pred, target.float())
        focal = self.focal(pred, target.float())
        dice = self.dice(pred, target.float())
        jaccard = self.iou(pred, target.float())
        return (self.weights['bce'] * bce + self.weights['tversky'] * tversky
                + self.weights['focal'] * focal + self.weights['dice'] * dice
                + self.weights['jaccard'] * jaccard)


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
