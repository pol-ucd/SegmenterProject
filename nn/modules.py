import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """
    :param - float alpha controls the penalty for false positives.
    :param - float beta controls the penalty for false negatives.
    :param - float smooth avoids division by zero and stabilizes learning.

    For example, setting alpha=0.7, beta=0.3 biases the loss
    to penalize false positives more harshly — making the model
    more conservative.

    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()

        tversky_score = (true_pos + self.smooth) / \
                        (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)

        return 1 - tversky_score  # Tversky loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # for logits, ensure values are in [0, 1]
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives
        true_pos = (inputs * targets).sum()

        # False Positives: predicted 1, actual 0
        false_pos = (inputs * (1 - targets)).sum()

        # False Negatives: predicted 0, actual 1
        false_neg = ((1 - inputs) * targets).sum()

        dice_score = (2 * true_pos + self.smooth) / (2 * true_pos + false_pos + false_neg + self.smooth)

        return 1 - dice_score


class FocalLoss(nn.Module):
    """
    param: - alpha: balancing factor to reduce impact of easy examples (typically 0.25).
    param: - gamma: focusing factor to emphasize hard examples (usually 2.0).
    params: - reduction: 'mean', 'sum', or 'none' depending on your setup.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = BinaryCrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        # Convert targets to float tensor
        targets = targets.type(torch.float32)

        # Apply sigmoid if inputs are logits
        inputs = torch.sigmoid(inputs)

        # Compute binary cross entropy

        bce_loss = self.bce(inputs, targets)
        # bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Compute p_t
        pt = inputs * targets + (1 - inputs) * (1 - targets)

        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IOULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IOULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # make sure inputs are between 0 and 1
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives: predicted 1, actual 1
        true_pos = (inputs * targets).sum()

        # False Positives: predicted 1, actual 0
        false_pos = (inputs * (1 - targets)).sum()

        # False Negatives: predicted 0, actual 1
        false_neg = ((1 - inputs) * targets).sum()

        # IoU = TP / (TP + FP + FN)
        iou = (true_pos + self.smooth) / (true_pos + false_pos + false_neg + self.smooth)
        return 1 - iou  # IoU Loss


class BinaryCrossEntropyLoss(nn.Module):
    """
    Supports 'mean', 'sum', or no reduction.
    """
    def __init__(self, reduction='mean'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)
        # Compute BCE manually
        bce = - (targets * torch.log(inputs + 1e-8) + (1 - targets) * torch.log(1 - inputs + 1e-8))

        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        else:
            return bce

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

    # def __call__(self, pred, target):
    #     return self._do_calculation(pred, target)

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
        # self.bce = nn.BCEWithLogitsLoss()
        # self.bce = BinaryCrossEntropyLoss()
        # self.tversky = TverskyLoss(alpha=0.2, beta=0.4)
        # self.focal = FocalLoss()
        # self.tversky = TverskyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self._do_calculation(pred, target)

    # def __call__(self, pred, target):
    #     return self._do_calculation(pred, target)

    def _do_calculation(self, pred, target):
        # print(f"1. Is contiguous? target: {target.is_contiguous()}, pred: {pred.is_contiguous()}")
        # target = target.squeeze()
        # pred = pred.squeeze()
        #
        # print(f"2. Is contiguous? target: {target.is_contiguous()}, pred: {pred.is_contiguous()}")
        pred = pred.transpose(3, 1)
        logits = torch.sigmoid(pred).contiguous()
        # bce = self.bce(pred, target.float())
        # tversky = self.tversky(logits, target.float())
        # focal = self.focal(logits, target.float())
        dice = self.dice(logits, target)
        # return self.weights['bce'] * bce + self.weights['tversky'] * tversky + self.weights['focal'] * focal
        return dice
