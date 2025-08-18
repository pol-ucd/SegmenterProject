# loss_seg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot_mask(mask_hw: torch.LongTensor, num_classes: int, ignore_index: int = -1):
    # mask_hw: [B,H,W] Long
    # B, H, W = mask_hw.shape

    mask = mask_hw.clone().squeeze(1)
    valid = (mask != ignore_index)
    mask[~valid] = 0

    oh = F.one_hot(mask.squeeze(1), num_classes=num_classes).permute(0,3,1,2).float()  # [B,C,H,W]
    # zero-out ignored pixels in one-hot
    oh *= valid.unsqueeze(1)
    return oh, valid

class SoftJaccardLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = -1, class_weights: torch.Tensor = None, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target_hw: torch.Tensor):
        """
        logits: [B,C,H,W] raw outputs
        target_hw: [B,H,W] class indices (Long), may contain ignore_index
        """
        probs = F.softmax(logits, dim=1) # [B,n_classes,H,W]
        target_oh, valid = one_hot_mask(target_hw, self.num_classes, self.ignore_index)

        # Mask out ignored pixels
        probs = probs * valid.unsqueeze(1)

        intersection = (probs * target_oh).sum(dim=(0,2,3))  # [C]
        union = probs.sum(dim=(0,2,3)) + target_oh.sum(dim=(0,2,3)) - intersection  # [C]
        jaccard = (intersection + self.smooth) / (union + self.smooth)               # [C]
        loss_c = 1.0 - jaccard                                                       # [C]

        if self.class_weights is not None:
            loss = (loss_c * self.class_weights).sum() / (self.class_weights.sum() + 1e-12)
        else:
            loss = loss_c.mean()
        return loss

class CEJaccardLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = -1, ce_weight: float = 0.5,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      # ignore_index=ignore_index
                                      )
        self.jacc = SoftJaccardLoss(num_classes=num_classes,
                                    ignore_index=ignore_index,
                                    class_weights=class_weights)
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits, target_hw):

        j = self.jacc(logits,
                      target_hw)
        ce_target_hw, _ = one_hot_mask(target_hw, self.num_classes, self.ignore_index)
        ce = self.ce(logits,
                     ce_target_hw)
        return self.ce_weight * ce + (1.0 - self.ce_weight) * j


class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = -1, class_weights: torch.Tensor = None, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target_hw: torch.Tensor):
        """
        Args:
            logits: [B, C, H, W] raw model outputs
            target_hw: [B, H, W] integer class labels
        Returns:
            Dice loss (float)
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        target_oh, valid = one_hot_mask(target_hw, self.num_classes, self.ignore_index)

        # Mask out ignored pixels
        probs = probs * valid.unsqueeze(1)
        target_oh = target_oh * valid.unsqueeze(1)

        intersection = (probs * target_oh).sum(dim=(0, 2, 3))  # [C]
        union = probs.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))  # [C]
        dice = (2 * intersection + self.smooth) / (union + self.smooth)  # [C]
        loss_c = 1.0 - dice  # [C]

        if self.class_weights is not None:
            loss = (loss_c * self.class_weights).sum() / (self.class_weights.sum() + 1e-12)
        else:
            loss = loss_c.mean()
        return loss


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, **kwargs):
        super().__init__()
        self.dice = MulticlassDiceLoss(**kwargs)
        self.ce = nn.CrossEntropyLoss(ignore_index=kwargs.get("ignore_index", -1))
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, target):
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)

