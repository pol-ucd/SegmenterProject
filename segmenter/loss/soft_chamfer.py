import torch
from torch import nn as nn
from torch.nn import functional as F

from segmenter.loss import BaseLoss
from segmenter.loss import EPSILON
from segmenter.loss.boundary import make_soft_boundary
from segmenter.loss.distance import DistanceTransform2D
from segmenter.loss.utils import one_hot


class SoftChamferLoss(BaseLoss):
    """
    A symmetric soft Chamfer loss that penalizes surface misalignment.
    Uses soft predicted boundaries and GT boundaries based on morphological operations.
    """

    def __init__(self, dt_backend: str = "fastgeodis", tau: float = 0.8, band_px: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.dt = DistanceTransform2D(backend=dt_backend)
        self.tau = tau
        self.band_px = band_px
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = max(predicted.shape[1], 2)
        probs = F.softmax(predicted, dim=1)
        if target.shape[1] == 1:
            target_onehot = one_hot(target, num_classes=num_classes).to(predicted.dtype)
        else:
            target_onehot = target.to(predicted.dtype)
        loss = 0.0
        for c in range(num_classes):
            p = probs[:, c:c + 1]
            m = target_onehot[:, c:c + 1]

            # GT boundary approximation via morphological gradient
            dil = self.pool(m)
            ero = 1.0 - self.pool(1.0 - m)
            e_gt = torch.clamp(dil - ero, 0, 1)

            # Narrow band
            if self.band_px > 0:
                k = 2 * self.band_px + 1
                band = nn.MaxPool2d(k, stride=1, padding=self.band_px)(e_gt)
                e_gt_band = band
            else:
                e_gt_band = e_gt

            # Soft predicted boundary
            e_pred = make_soft_boundary(p, tau=self.tau)

            # Distances to boundaries
            d_to_gt = self.dt.edt(e_gt_band)
            d_to_pred = self.dt.edt((e_pred > 0.01).float()).detach()

            # Chamfer-style symmetric loss terms
            term_pred_to_gt = (e_pred * d_to_gt).sum(dim=(2, 3)) / (e_pred.sum(dim=(2, 3)) + EPSILON)
            term_gt_to_pred = (e_gt * d_to_pred).sum(dim=(2, 3)) / (e_gt.sum(dim=(2, 3)) + EPSILON)
            loss_c = 0.5 * (term_pred_to_gt.mean() + term_gt_to_pred.mean())
            if torch.isnan(loss_c).any():
                loss_c = torch.zeros_like(loss_c)
            loss += loss_c

        return loss / num_classes
