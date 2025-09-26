import torch

from segmenter.loss.base import BaseLoss
from segmenter.modules import EPSILON


class DiceLoss(BaseLoss):
    """
    Calculates the multi-class Dice Loss.
    """

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = super().setup(predicted, target)

        tp = scores[0]
        fp = scores[1]
        fn = scores[2]
        dice = (2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON)

        return 1 - dice
