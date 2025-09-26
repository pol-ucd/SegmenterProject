import torch

from segmenter.loss.base import BaseLoss
from segmenter.modules import EPSILON


class IoULoss(BaseLoss):
    """Calculates the Intersection over Union (IoU) Loss."""

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = super().setup(predicted, target)

        tp = scores[0]
        fp = scores[1]
        fn = scores[2]
        iou = (tp + EPSILON) / (tp + fp + fn + EPSILON)

        return 1 - iou
