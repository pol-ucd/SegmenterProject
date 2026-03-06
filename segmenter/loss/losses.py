"""
Loss Functions for Segmentation
Hybrid loss combining multiple objectives for optimal performance.
Based on Transection_Classifier implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, label_smoothing=0.0):
        super().__init__()
        self.smooth = smooth
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / 2

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / 2

        bce = F.binary_cross_entropy(pred.float(), target.float(), reduction='none')

        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    def __init__(self, sigma=5.0, kernel_size=3):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def _get_boundary(self, mask):
        dilated = F.max_pool2d(
            mask, kernel_size=self.kernel_size, stride=1, padding=self.pad
        )
        eroded = 1.0 - F.max_pool2d(
            1.0 - mask, kernel_size=self.kernel_size, stride=1, padding=self.pad
        )
        return dilated - eroded

    def forward(self, pred, target):
        boundary = self._get_boundary(target)
        weight = 1.0 + self.sigma * boundary
        bce = F.binary_cross_entropy(pred.float(), target.float(), reduction='none')
        return (bce * weight).mean()


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - tversky


class ComboLoss(nn.Module):
    def __init__(
        self,
        dice_weight=0.5,
        focal_weight=0.3,
        boundary_weight=0.2,
        dice_smooth=1e-6,
        focal_alpha=0.25,
        focal_gamma=2.0,
        boundary_sigma=5.0,
        label_smoothing=0.0
    ):
        super().__init__()

        # A1: validate weights sum to 1.0
        assert abs(dice_weight + focal_weight + boundary_weight - 1.0) < 1e-4, (
            f"Loss weights must sum to 1.0, got {dice_weight + focal_weight + boundary_weight:.4f}"
        )

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

        self.dice_loss = DiceLoss(smooth=dice_smooth, label_smoothing=label_smoothing)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
        self.boundary_loss = BoundaryLoss(sigma=boundary_sigma)

    def forward(self, pred_mask, target_mask):
        loss_dice = self.dice_loss(pred_mask, target_mask)
        loss_focal = self.focal_loss(pred_mask, target_mask)
        loss_boundary = self.boundary_loss(pred_mask, target_mask)

        total_loss = (
            self.dice_weight * loss_dice +
            self.focal_weight * loss_focal +
            self.boundary_weight * loss_boundary
        )

        # E5: compute metrics once here so callers don't need a second pass
        with torch.no_grad():
            pred_binary = (pred_mask.detach() > 0.5).float()
            target_flat = target_mask.detach()
            tp = (pred_binary * target_flat).sum()
            fp = (pred_binary * (1.0 - target_flat)).sum()
            fn = ((1.0 - pred_binary) * target_flat).sum()
            intersection = tp
            union = pred_binary.sum() + target_flat.sum() - intersection
            smooth = 1e-6
            iou = ((intersection + smooth) / (union + smooth)).item()
            dice_metric = ((2.0 * intersection + smooth) / (pred_binary.sum() + target_flat.sum() + smooth)).item()
            precision = (tp / (tp + fp + smooth)).item()
            recall = (tp / (tp + fn + smooth)).item()

        loss_dict = {
            'total': total_loss,
            'dice': loss_dice,
            'focal': loss_focal,
            'boundary': loss_boundary,
            'iou': iou,
            'dice_metric': dice_metric,
            'precision': precision,
            'recall': recall,
        }

        return total_loss, loss_dict


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video segmentation.
    Encourages smooth predictions between consecutive frames.
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_curr, pred_next):
        """
        Args:
            pred_curr: Current frame predictions [B, C, H, W]
            pred_next: Next frame predictions [B, C, H, W]
        """
        diff = (pred_curr - pred_next).abs()
        return diff.mean() * self.alpha


class MultiScaleLoss(nn.Module):
    """Multi-scale training loss for better boundary detection."""
    def __init__(self, base_loss, scales=[1.0, 0.5, 0.25], weights=None):
        super().__init__()
        self.base_loss = base_loss
        self.scales = scales
        self.weights = weights or [1.0, 0.5, 0.25]

    def forward(self, pred, target):
        total_loss = 0
        h, w = target.shape[2:]
        
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1.0:
                pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                pred_scaled, target_scaled = pred, target
            
            total_loss += self.base_loss(pred_scaled, target_scaled) * weight
        
        return total_loss


def get_loss_function(loss_type='combo', **kwargs):
    """
    Factory function to get loss function by name.
    """
    loss_functions = {
        'dice': DiceLoss,
        'focal': FocalLoss,
        'boundary': BoundaryLoss,
        'iou': IoULoss,
        'tversky': TverskyLoss,
        'combo': ComboLoss,
    }

    if loss_type not in loss_functions:
        raise ValueError(f"Loss type '{loss_type}' not recognized. Available: {list(loss_functions.keys())}")

    return loss_functions[loss_type](**kwargs)
