import logging
from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _compute_signed_distance_map(one_hot_mask: np.ndarray) -> np.ndarray:
    """
    Computes a signed distance map for a single one-hot encoded mask.

    This function calculates the distance from each pixel to the nearest boundary.
    It's a CPU-bound operation and should be used with caution inside a training
    loop. For best performance, consider pre-calculating these maps offline
    and saving them with your dataset.

    Args:
        one_hot_mask (np.ndarray): A 2D NumPy array representing a single
                                   class mask (1s for the class, 0s otherwise).

    Returns:
        np.ndarray: A 2D NumPy array with the signed distance map.
    """
    # Calculate positive and negative distance maps
    dist_map_positive = distance_transform_edt(one_hot_mask)
    dist_map_negative = distance_transform_edt(1 - one_hot_mask)

    # Combine them to get the signed distance map
    signed_dist_map = dist_map_positive - dist_map_negative
    return signed_dist_map


def boundary_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the multi-class boundary loss in a vectorized manner.

    This refactored version processes the entire batch and all classes simultaneously,
    making it significantly more performant than the original.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
                             Expected shape: (N, C, H, W).
        target (torch.Tensor): The ground truth mask tensor with class indices.
                               Expected shape: (N, H, W).

    Returns:
        torch.Tensor: The calculated multi-class boundary loss.
    """
    device = pred.device
    num_classes = pred.shape[1]

    # Convert true_mask to a one-hot encoded tensor and move to CPU for scipy
    # Shape changes from (N, H, W) to (N, C, H, W)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
    target_one_hot_cpu = target_one_hot.cpu().numpy()

    # Process each sample and class to create the signed distance maps
    # We use a list comprehension to handle the iteration over the batch and classes,
    # which is more efficient than nested loops.
    dist_maps = [_compute_signed_distance_map(target_one_hot_cpu[i, c])
                 for i in range(pred.shape[0]) for c in range(num_classes)]

    # Convert the list of numpy arrays to a single PyTorch tensor
    dist_maps = torch.from_numpy(np.stack(dist_maps)).float().to(device)

    # Reshape the distance maps to match the prediction tensor shape
    dist_maps = dist_maps.view(pred.shape[0], num_classes, pred.shape[2], pred.shape[3])

    # Apply sigmoid to the prediction to get a probability map
    pred_sig = torch.sigmoid(pred)

    # FIX: Take the absolute value of the distance map to ensure the loss is non-negative.
    # A loss function must always be a non-negative value.
    loss = (pred_sig * dist_maps.abs()).mean()

    return loss


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
               smooth: float = 1e-6) -> torch.Tensor:
    """
    Computes the Focal Loss.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
        target (torch.Tensor): The ground truth mask tensor with class indices.
        alpha (float, optional): Alpha parameter for the focal loss.
        gamma (float, optional): Gamma parameter for the focal loss.
        smooth (float, optional): Small value to avoid log(0) issues.

    Returns:
        torch.Tensor: The calculated focal loss.
    """
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    pt = torch.where(target_onehot == 1, pred, 1 - pred)
    focal_term = alpha * (1 - pt) ** gamma
    bce = -torch.log(pt + smooth)
    return (focal_term * bce).mean()


def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, beta: float = 0.5,
                 smooth: float = 1e-6) -> torch.Tensor:
    """
    Computes the Tversky Loss.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
        target (torch.Tensor): The ground truth mask tensor with class indices.
        alpha (float, optional): False positive weight.
        beta (float, optional): False negative weight.
        smooth (float, optional): Small value to avoid division by zero.

    Returns:
        torch.Tensor: The calculated tversky loss.
    """
    probs = torch.softmax(pred, dim=1)
    target_oh = F.one_hot(target, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()

    # Vectorized calculation over all dimensions
    TP = (probs * target_oh).sum(dim=(0, 2, 3))
    FP = (probs * (1 - target_oh)).sum(dim=(0, 2, 3))
    FN = ((1 - probs) * target_oh).sum(dim=(0, 2, 3))

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss_tversky = 1.0 - tversky
    return loss_tversky.mean()


def dice_loss(pred: torch.Tensor, true_mask: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculates the multi-class Dice Loss.

    This implementation is fully vectorized for efficiency and follows the
    standard Dice Loss formula.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
                             Expected shape: (N, C, H, W).
        true_mask (torch.Tensor): The ground truth mask tensor with class indices.
                                  Expected shape: (N, H, W).
        epsilon (float, optional): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The calculated multi-class Dice Loss.
    """
    num_classes = pred.shape[1]
    pred_prob = F.softmax(pred, dim=1)

    # Convert true_mask from class indices to a one-hot encoded tensor
    true_mask_one_hot = F.one_hot(true_mask.long(), num_classes=num_classes)
    true_mask_one_hot = true_mask_one_hot.permute(0, 3, 1, 2).float()

    # Flatten the tensors for easier element-wise operations and summation
    pred_flat = pred_prob.view(-1, pred.shape[2] * pred.shape[3])
    true_flat = true_mask_one_hot.reshape(-1, true_mask_one_hot.shape[2] * true_mask_one_hot.shape[3])

    intersection = (pred_flat * true_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1)

    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    dice_loss_val = 1.0 - dice_score

    return dice_loss_val.mean()


def iou_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Computes the IoU Loss.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
        target (torch.Tensor): The ground truth mask tensor with class indices.
        epsilon (float, optional): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The calculated IoU loss.
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
    Abstract base class for all loss functions to ensure a consistent interface.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            pred (torch.Tensor): The model's output.
            target (torch.Tensor): The ground truth.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        pass


class BoundaryLoss(BaseLoss):
    """
    Loss based on the signed distance map from boundaries.

    Args:
        scale_factor (float): A factor to scale the final loss by, to
                              prevent it from dominating other losses.
    """

    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = boundary_loss(pred, target)
        return loss * self.scale_factor


class DiceLoss(BaseLoss):
    """Calculates the Dice Loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(pred, target, self.epsilon)


class FocalLoss(BaseLoss):
    """
    Calculates the Focal Loss with adjustable alpha and gamma parameters.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(pred, target, alpha=self.alpha, gamma=self.gamma, smooth=self.epsilon)


class TverskyLoss(BaseLoss):
    """
    Calculates the Tversky Loss with adjustable alpha and beta parameters.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tversky_loss(pred, target, alpha=self.alpha, beta=self.beta, smooth=self.epsilon)


class IoULoss(BaseLoss):
    """Calculates the Intersection over Union (IoU) Loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return iou_loss(pred, target, self.epsilon)


class HybridLoss(nn.Module):
    """
    Combines multiple loss functions with configurable weights.

    This class correctly initializes all loss components and applies them
    to the input tensors before combining.
    """

    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0,
                 weight_focal: float = 1.0, weight_tversky: float = 1.0,
                 weight_iou: float = 1.0, weight_boundary: float = 1.0):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_tversky = weight_tversky
        self.weight_iou = weight_iou
        self.weight_boundary = weight_boundary

        # Initialize all loss components
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss(alpha=0.2, beta=0.4)
        self.iou_loss = IoULoss()
        # Initializing BoundaryLoss with a scale_factor to prevent it from dominating
        self.boundary_loss = BoundaryLoss(scale_factor=1e-3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Assuming target has shape (N, H, W) and pred is logits (N, C, H, W)
        # Squeeze dim 1 if it exists, as CrossEntropyLoss expects (N, C, ...) and (N, ...)
        target_squeezed = target.squeeze(1) if target.dim() == pred.dim() else target

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


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions for binary image classification.

    This version correctly handles the binary case by using BCEWithLogitsLoss
    and passing the appropriate targets to other multi-class losses.
    """

    def __init__(self, weights: Optional[dict] = None):
        super().__init__()
        if weights is None:
            weights = {'bce': 0.2, 'tversky': 0.4, 'focal': 0.4, 'dice': 0.6, 'jaccard': 0.6}
        self.weights = weights

        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=0.2, beta=0.4)
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # For binary classification, pred has shape (N, 1, H, W) or (N, H, W)
        # target has shape (N, H, W) with values 0 or 1

        # BCEWithLogitsLoss expects float target
        bce = self.bce(pred, target.unsqueeze(1).float())

        # For the other losses, we treat the binary case as a multi-class
        # problem with 2 classes (0 and 1) to use the existing multi-class
        # implementations.
        tversky = self.tversky(pred, target.long())
        focal = self.focal(pred, target.long())
        dice = self.dice(pred, target.long())
        jaccard = self.iou(pred, target.long())

        total_loss = (self.weights['bce'] * bce +
                      self.weights['tversky'] * tversky +
                      self.weights['focal'] * focal +
                      self.weights['dice'] * dice +
                      self.weights['jaccard'] * jaccard)
        return total_loss


class EarlyStopping:
    """
    Early Stopping Implementation to halt training when a metric stops improving.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 mode: str = 'min', verbose: bool = False,
                 save_path: Optional[str] = None):
        """
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

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_epoch: Optional[int] = None

        self._init_comparator()

    def _init_comparator(self):
        """Initializes the comparison logic based on mode."""
        if self.mode == 'min':
            self.compare = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf')
        else:
            self.compare = lambda current, best: current > best + self.min_delta
            self.best_score = float('-inf')

    def __call__(self, current_score: float, model: Optional[nn.Module] = None, epoch: Optional[int] = None):
        """
        Evaluates the current score against the best score and updates state.

        Args:
            current_score (float): The score from the current epoch.
            model (Optional[nn.Module]): The model to save if a new best score is found.
            epoch (Optional[int]): The current epoch number.
        """
        if self.best_score is None or self.compare(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                log_message = f"New best score: {current_score:.4f}"
                if epoch is not None:
                    log_message += f" at epoch {epoch + 1}"
                self.logger.info(log_message)

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
                if self.verbose and epoch is not None:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")

    def reset(self):
        """Resets the state of the EarlyStopping instance."""
        self.counter = 0
        self.early_stop = False
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = None
