import logging
from abc import abstractmethod, ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def boundary_loss(pred, target):
    """
    Calculates the multi-class boundary loss.

    This refactored version calculates a signed distance map for each class
    individually and sums their contributions to the final loss.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
                             Expected shape: (N, C, H, W) where N is batch size,
                             C is number of classes, and H, W are dimensions.
        target (torch.Tensor): The ground truth mask tensor with class indices.
                                  Expected shape: (N, H, W).

    Returns:
        torch.Tensor: The calculated multi-class boundary loss.
    """
    # Ensure inputs are on the same device
    device = pred.device
    num_classes = pred.shape[1]

    # Convert true_mask from class indices to a one-hot encoded tensor
    # Shape changes from (N, H, W) to (N, C, H, W)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    total_loss = 0.0

    # Iterate through each sample in the batch
    for i in range(pred.shape[0]):
        # Iterate through each class to calculate its contribution to the loss
        for c in range(num_classes):
            # Extract the single-channel prediction and true mask for the current class
            pred_c = pred[i, c, :, :]
            target_c = target_one_hot[i, c, :, :]

            # Move the true_mask to CPU and convert to a NumPy array for scipy
            target_np = target_c.cpu().numpy()

            # Calculate the signed distance map
            # This is a CPU-only operation, which can be a bottleneck.
            # For production, it's recommended to pre-calculate this during data loading.
            dist_map_positive = distance_transform_edt(target_np)
            dist_map_negative = distance_transform_edt(1 - target_np)
            dist_map_np = dist_map_positive - dist_map_negative

            # Convert the distance map back to a PyTorch tensor and move to the original device
            dist_map = torch.from_numpy(dist_map_np).float().to(device)

            # Apply sigmoid to the prediction to get a probability map
            pred_sig = torch.sigmoid(pred_c)

            # Calculate the per-class loss and add to the total
            loss_c = (pred_sig * dist_map).mean()
            total_loss += loss_c

    # Calculate the mean loss over all samples and classes
    final_loss = total_loss / (pred.shape[0] * num_classes)

    return final_loss


def focal_loss(pred, target, alpha=0.25, gamma=2.0, smooth=1e-6):
    """
    Computes the Focal Loss.
    """
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    pt = torch.where(target_onehot == 1, pred, 1 - pred)
    focal_term = alpha * (1 - pt) ** gamma
    bce = -torch.log(pt + smooth)
    return (focal_term * bce).mean()


def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1e-6):
    """
    Computes the Tversky Loss.
    """
    probs = torch.softmax(pred, dim=1)
    target_oh = F.one_hot(target, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
    TP = (probs * target_oh).sum(dim=(0, 2, 3))
    FP = (probs * (1 - target_oh)).sum(dim=(0, 2, 3))
    FN = ((1 - probs) * target_oh).sum(dim=(0, 2, 3))

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss_tversky = 1.0 - tversky
    return loss_tversky.mean()


def dice_loss(pred, true_mask):
    """
    Calculates the multi-class Dice Loss.

    This implementation is fully vectorized for efficiency and follows the
    standard Dice Loss formula. It's designed to be used with the same
    inputs as the boundary loss function.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
                             Expected shape: (N, C, H, W).
        true_mask (torch.Tensor): The ground truth mask tensor with class indices.
                                  Expected shape: (N, H, W).

    Returns:
        torch.Tensor: The calculated multi-class Dice Loss.
    """
    # A small constant to prevent division by zero.
    epsilon = 1e-6

    # Get the device and number of classes
    device = pred.device
    num_classes = pred.shape[1]

    # Apply softmax to the predictions to get probabilities per class
    # Alternatively, you could use sigmoid if your model doesn't use softmax
    # on the output layer. For multi-class, softmax is more common.
    pred_prob = F.softmax(pred, dim=1)

    # Convert true_mask from class indices to a one-hot encoded tensor
    # Shape changes from (N, H, W) to (N, C, H, W)
    true_mask_one_hot = F.one_hot(true_mask.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Flatten the tensors for easier element-wise operations and summation
    # Shape changes from (N, C, H, W) to (N*C, H*W)
    pred_flat = pred_prob.view(-1, pred.shape[2] * pred.shape[3])
    true_flat = true_mask_one_hot.view(-1, true_mask_one_hot.shape[2] * true_mask_one_hot.shape[3])

    # Calculate the intersection (true positives)
    intersection = (pred_flat * true_flat).sum(dim=1)

    # Calculate the union (predicted positives + true positives)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1)

    # Calculate the Dice score and loss
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    dice_loss_val = 1.0 - dice_score

    # Average the loss across all samples and classes
    return dice_loss_val.mean()


def iou_loss(pred, target, epsilon=1e-6):
    """
    Computes the IoU Loss.
    """
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred + target_onehot - (pred * target_onehot)
    union = union.sum(dim=(2, 3))
    iou = (intersection + epsilon) / (union + epsilon)
    return 1 - iou.mean()



class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all losses.
    """
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.epsilon = 1e-6

    @abstractmethod
    def forward(self, pred, target):
        pass


class BoundaryLoss(BaseLoss):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target):
        return boundary_loss(pred, target)


class DiceLoss(BaseLoss):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        return dice_loss(pred, target, self.epsilon)


class FocalLoss(BaseLoss):
    def __init__(self, alpha=None, gamma=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 0.25
        self.gamma = gamma if gamma is not None else 2.0

    def forward(self, pred, target):
        return focal_loss(pred, target,
                          alpha=self.alpha,
                          gamma=self.gamma,
                          smooth=self.epsilon)


class TverskyLoss(BaseLoss):
    def __init__(self, alpha=None, beta=None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 0.5
        self.beta = beta if beta is not None else 0.5

    def forward(self, pred, target):
        return tversky_loss(pred, target, alpha=self.alpha, beta=self.beta)


class IoULoss(BaseLoss):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        return iou_loss(pred, target, self.epsilon)


class HybridLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0,
                 weight_focal=1.0, weight_tversky=1.0, weight_iou=1.0, weight_boundary=1.0):
        super(HybridLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_tversky = weight_tversky
        self.weight_iou = weight_iou
        self.weight_boundary = weight_boundary
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss(alpha=0.2, beta=0.4)
        self.iou_loss = IoULoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred, target):
        target_squeezed = target.squeeze(1)
        loss_ce = self.ce_loss(pred, target_squeezed)
        loss_dice = self.dice_loss(pred, target_squeezed)
        loss_focal = self.focal_loss(pred, target_squeezed)
        loss_tversky = self.tversky_loss(pred, target_squeezed)
        loss_iou = self.iou_loss(pred, target_squeezed)
        loss_boundary = self.boundary_loss(pred, target_squeezed)

        total_loss = (
                self.weight_ce * loss_ce +
                self.weight_dice * loss_dice +
                self.weight_focal * loss_focal +
                self.weight_tversky * loss_tversky +
                self.weight_iou * loss_iou +
                self.weight_boundary * loss_boundary
        )
        return total_loss


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
        self.tversky = TverskyLoss(alpha=0.2, beta=0.4)
        self.focal = FocalLoss()
        # Note: 'mode' is not a valid argument for the DiceLoss and IoULoss classes here,
        # they are designed for multi-class and handle binary via one-hot encoding.
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target.unsqueeze(1).float())

        # The following losses require the target to be handled differently, as they
        # were written to expect integer labels for one-hot encoding.
        # We assume the input 'pred' is logits and 'target' is the integer label mask.
        target_squeezed = target.squeeze(1)
        tversky = self.tversky(pred, target_squeezed)
        focal = self.focal(pred, target_squeezed)
        dice = self.dice(pred, target_squeezed)
        jaccard = self.iou(pred, target_squeezed)

        return (self.weights['bce'] * bce + self.weights['tversky'] * tversky
                + self.weights['focal'] * focal + self.weights['dice'] * dice
                + self.weights['jaccard'] * jaccard)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=False, save_path=None):
        """
        Early Stopping Implementation
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' for loss, 'max' for accuracy or IoU.
            verbose (bool): If True, logs updates.
            save_path (str): If set, saves best model to this path.
        """
        assert mode in ['min', 'max'], "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path
        self.logger = logging.getLogger(self.__class__.__name__)

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
                self.logger.info(f"New best score: {current_score:.4f} at epoch {epoch + 1}")
            if self.save_path and model is not None:
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    self.logger.info(f"Model saved to {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f"No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = None
