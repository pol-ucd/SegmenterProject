import torch
import torch.nn as nn
from torch.nn import functional as F

from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss

"""
Implements Dice Loss for Binary image classification.
Is also callable so it can be used to evaluate loss with no_grad()
"""
class DiceScore(nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        return self._do_calculation(pred, target)

    def __call__(self, pred, target):
        return self._do_calculation(pred, target)

    def _do_calculation(self, pred, target):
        pred = torch.sigmoid(pred)
        n_samples = pred.size(0)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice


class IOUScore(nn.Module):
    def __init__(self, smooth=1):
        super(IOUScore, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        return self._do_calculation(pred)

    def __call__(self, pred, target):
        return self._do_calculation(pred)

    def _do_calculation(self, pred):
        outputs = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
        intersection = (outputs & pred).float().sum()
        union = (outputs | pred).float().sum()
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou


"""
Implements Hanija's Combined Loss for Binary image classification.
Is also callable so it can be used to evaluate loss with no_grad()
"""
class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super(CombinedLoss, self).__init__()
        if weights is None:
            weights = {'bce': 0.5, 'tversky': 0.3, 'focal': 0.2}
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(mode='binary', alpha=0.2, beta=0.4)
        self.focal = FocalLoss(mode='binary')

    def forward(self, pred, target):
        return self._do_calculation(pred, target)

    def __call__(self, pred, target):
        return self._do_calculation(pred, target)

    def _do_calculation(self, pred, target):
        target = target.squeeze()
        pred = pred.squeeze()
        if pred.shape[-2:] != target.shape[-2:]:
            targets = F.interpolate(target, size=pred.shape[-2:], mode='nearest')
        loss = (self.weights['bce'] * self.bce(pred, target)
                + self.weights['tversky'] * self.tversky(torch.sigmoid(pred), target)
                + self.weights['focal'] * self.focal(torch.sigmoid(pred), target))
        return loss

