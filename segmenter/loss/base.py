from abc import ABC, abstractmethod

import torch
from torch import nn as nn
from torch.nn import functional as F

from segmenter.loss.utils import one_hot

EPSILON = 1e-6

class BaseLoss(nn.Module, ABC):
    """Abstract base class for all loss functions."""

    def __init__(self, **kwargs):
        super().__init__()
        self.class_scores = None
        self.targets = None
        self.probabilities = None
        self.kwargs = kwargs

    @abstractmethod
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    def setup(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """"Calculates the loss."""
        # self._check_params(predicted, target)
        self.probabilities, self.targets = self._format_params(predicted, target)
        self.scores = self._shape_scores(self.probabilities, self.targets)
        return self.class_scores.sum(dim=0)

    @staticmethod
    def _check_params(predicted: torch.Tensor, target: torch.Tensor):
        try:
            assert predicted.shape == target.shape
        except AssertionError:
            msg = f" Input mismatch, Logits and Target Mask must have the same shape. Logits shape: {predicted.shape}, target mask shape: {target.shape}"
            raise LossException(msg)

    @staticmethod
    def _format_params(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(predicted, dim=1, dtype=torch.float32)
        if len(target.shape) == 3:
        #     target = target.unsqueeze(1)
        # if target.shape[1] == 1:
            target_onehot = one_hot(target,
                                    num_classes=predicted.shape[1]).to(probs.dtype)
        else:
            target_onehot = target.to(probs.dtype)
        try:
            assert probs.shape == target_onehot.shape
        except AssertionError:
            msg = f" Input mismatch, Logits and Target Mask must have the same shape. Logits shape: {predicted.shape}, target mask shape: {target.shape}"
            raise LossException(msg)

        return probs, target_onehot

    def _shape_scores(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the per-class (tp, fn) scores for a predicted and target mask."""
        (b, c, h, w) = predicted.shape
        scores = torch.zeros(size=(c, 4), device=predicted.device)
        probs, target = self._format_params(predicted, target)
        if c == 1:
            predicted_max = (probs > 0.5)
            target_max = target.float()
        else:
            predicted_max = torch.argmax(probs, dim=1)
            target_max = torch.argmax(target, dim=1)

        denom = b * h * w
        for i_c in range(c):
            target_c = (target_max == i_c).float()
            pred_c = (predicted_max == i_c).float()

            scores[i_c, 0] = (target_c * pred_c).sum() / denom  # tp
            scores[i_c, 1] = ((1 - target_c) * pred_c).sum() / denom  # fn
            scores[i_c, 2] = (target_c * (1 - pred_c)).sum() / denom  # fp
            scores[i_c, 3] = ((1 - target_c) * (1 - pred_c)).sum() / denom  # tn
        self.class_scores = scores
        return scores.sum(dim=0)


class LossException(Exception):
    def __init__(self, message, errors=None):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        """ Custom errors that the caller can print from e.errors"""
        self.errors = errors
