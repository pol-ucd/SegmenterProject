import torch

from segmenter.loss import BaseLoss, EPSILON


class IoULoss(BaseLoss):
    """Calculates Intersection over Union (IoU or Jaccard) Loss."""

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = super().setup(predicted, target)

        tp = scores[0]
        fp = scores[1]
        fn = scores[2]
        iou = (tp + EPSILON) / (tp + fp + fn + EPSILON)

        return 1 - iou
