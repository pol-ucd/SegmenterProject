import torch

from segmenter.loss import BaseLoss, EPSILON


class TverskyLoss(BaseLoss):
    """Calculates the Tversky Loss."""

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        scores = super().setup(predicted, target)
        alpha = self.kwargs.get('alpha', 0.5)
        beta = self.kwargs.get('beta', 0.5)

        tp = scores[0]
        fp = scores[1]
        fn = scores[2]

        tversky = (tp + EPSILON) / (tp + alpha * fp + beta * fn + EPSILON)
        return 1.0 - tversky
