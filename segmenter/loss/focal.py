import torch

from segmenter.loss.base import BaseLoss
from segmenter.modules import EPSILON


class FocalLoss(BaseLoss):
    """Calculates the Focal Loss."""

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _ = super().setup(predicted, target)

        alpha = self.kwargs.get('alpha', 0.25)
        gamma = self.kwargs.get('gamma', 2.0)
        focal = torch.zeros(1, device=predicted.device)

        n_classes = target.shape[1]

        for c in range(n_classes):
            pt = torch.where(self.targets[:,c,:,:] == c,
                             self.probabilities[:,c,:,:],
                             1 - self.probabilities[:,c,:,:])

            focal_term = alpha * (1 - pt) ** gamma
            bce = -torch.log(pt + EPSILON)
            focal += (focal_term * bce).mean()
        return focal.mean()
