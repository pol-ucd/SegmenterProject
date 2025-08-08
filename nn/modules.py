from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import TverskyLoss as TL, FocalLoss as FL

class BaseLossClass(nn.Module, ABC):
    def __init__(self):
        super(BaseLossClass, self).__init__()

    def forward(self, logits, true):
        self.n_batch = true.shape[0]
        if self.n_batch != logits.shape[0]:
            raise ValueError("logits and targets must have the same batch size as dim 0")
        self.pos_prob = torch.sigmoid(logits)

        self.true_pos = (true * self.pos_prob).sum()
        self.false_neg = ((1 - true) * self.pos_prob).sum()
        self.false_pos = (true * (1 - self.pos_prob)).sum()


class TverskyLoss(BaseLossClass):
    """Computes the Tversky loss [1].


    :param alpha: controls the penalty for false positives.
    :param beta: controls the penalty for false negatives.
    :param eps: added to the denominator for numerical stability.

    :returns tversky_loss: the Tversky loss.

    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha:float =0.5, beta:float =0.5, smooth:float =1.0):
        """
        Assumes inputs are
        a. logits, and,
        b. 4-d shape (with the batch dimension as shape[0])

        returns the cumulative batch loss (you normalise return value to get mean or cumulative loss)

        :param alpha: float - regulariser for false positives
        :param beta: float - regulariser for false negatives
        :param smooth: float - smoothing factor to avoid divide by zero

        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        :param inputs - a tensor of shape [B, H, W] or [B, 1, H, W].
        :param logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        :returns float - (1 - tversky_score)*batch_size
        """
        super(TverskyLoss, self).forward(inputs, targets)
        # n_batch = inputs.shape[0]
        # if n_batch != targets.shape[0]:
        #     raise ValueError("inputs and targets must have the same batch size as dim 0")
        #
        # true_pos = (inputs * targets).sum()
        # false_neg = ((1 - inputs) * targets).sum()
        # false_pos = (inputs * (1 - targets)).sum()

        tversky_score = (self.true_pos + self.smooth) / \
                        (self.true_pos + self.alpha * self.false_pos + self.beta * self.false_neg + self.smooth)

        return self.n_batch - tversky_score  # Tversky loss


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
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TL(alpha=0.2, beta=0.4, mode='binary')
        self.focal = FL(mode='binary')


    def forward(self, pred, target):
        return self._do_calculation(pred, target)

    # def __call__(self, pred, target):
    #     return self._do_calculation(pred, target)

    def _do_calculation(self, pred, target):
        # pred = pred.transpose(3, 1)
        bce = self.bce(pred, target.float())
        tversky = self.tversky(pred, target.float())
        focal = self.focal(pred, target.float())
        return self.weights['bce'] * bce + self.weights['tversky'] * tversky + self.weights['focal'] * focal

if __name__ == "__main__":
    from segmentation_models_pytorch.losses import TverskyLoss as TL
    """ Unit testing """
    n_batch = 5

    test_target = torch.randint(low=0, high=1, size=(n_batch, 3, 256, 256)).float()
    test_prob = torch.rand(n_batch, 3, 256, 256)
    test_logits = torch.log(test_prob/(1 - test_prob))
    loss_fn = TverskyLoss()
    loss_fn2 = TL(mode="binary")
    loss = loss_fn(test_logits, test_target).item()
    loss2 = (loss_fn2(test_logits, test_target)).item()
    print(loss, loss2)

