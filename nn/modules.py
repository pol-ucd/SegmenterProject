import torch
import torch.nn as nn

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
        pred = (pred > 0.5).bool()
        target = (target > 0).bool()
        intersection = (pred & target).float().sum()
        union = (pred | target).float().sum()
        dice = (2.0 * intersection + self.smooth) / (union + intersection + self.smooth)
        return dice


class IOUScore(nn.Module):
    def __init__(self, smooth=1):
        super(IOUScore, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        return self._do_calculation(pred, target)

    def __call__(self, pred, target):
        return self._do_calculation(pred, target)

    def _do_calculation(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).bool()
        target = (target > 0).bool()
        intersection = (pred & target).float().sum()
        union = (pred | target).float().sum()
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
        # print(f"1. Is contiguous? target: {target.is_contiguous()}, pred: {pred.is_contiguous()}")
        # target = target.squeeze()
        # pred = pred.squeeze()
        #
        # print(f"2. Is contiguous? target: {target.is_contiguous()}, pred: {pred.is_contiguous()}")
        pred = pred.transpose(3, 1)
        logits = torch.sigmoid(pred).contiguous()
        bce = self.bce(pred, target.float())
        tversky = self.tversky(logits, target.float())
        focal = self.focal(logits, target.float())

        return self.weights['bce'] * bce + self.weights['tversky'] * tversky + self.weights['focal'] * focal

