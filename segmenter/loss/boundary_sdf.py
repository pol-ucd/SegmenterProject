import torch
from torch.nn import functional as F

from segmenter.loss import BaseLoss, one_hot, DistanceTransform2D


class BoundarySDFLoss(BaseLoss):
    """
    Boundary Loss based on Signed Distance Functions (SDF).
    Penalizes misalignment between predicted and ground truth boundaries.
    """

    def __init__(self, dt_backend: str = "fastgeodis", **kwargs):
        super().__init__(**kwargs)
        self.dt = DistanceTransform2D(backend=dt_backend)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = max(predicted.shape[1], 2)
        is_one_class = (predicted.shape[1] == 1)

        probs = F.softmax(predicted, dim=1)
        if target.shape[1] == 1:
            target_onehot = one_hot(target, num_classes=num_classes).to(predicted.dtype)
        else:
            target_onehot = target.to(predicted.dtype)

        loss = 0.0
        for c in range(num_classes):
            gt_c = target_onehot[:, c:c + 1]
            sdt_c = self.dt.signed_distance(gt_c)
            # Normalize SDF per-sample for scale invariance
            sdt_c = sdt_c / (sdt_c.amax(dim=(-1, -2), keepdim=True) + EPSILON)

            # Penalize the difference between prediction and ground truth,
            # weighted by the absolute signed distance. This ensures zero loss
            # for a perfect match, regardless of the SDF value.
            loss_c = (sdt_c.abs() * (probs[:, c:c + 1] - gt_c).abs()).mean()
            if torch.isnan(loss_c).any():
                loss_c = torch.zeros_like(loss_c)
            loss += loss_c

            if is_one_class:
                break

        return loss / num_classes
