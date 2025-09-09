"""
Various Loss functions and related utilities.

Note. The Loss functions below have signatures that are consistent with the type and shape
of prediction and target tensors used in the processed pipeline.

Target masks are always one hot encoded to be of shape [B, C, H, W]
Predictions are logits of shape [B, C, H, W]

Where:
    B = the batch dimension.
    C = the number of classes.
    H, W = the image height and width.

Target Masks have type Torch.Long
Predictions have type Torch.Float

Loss is returned as a single mean loss per batch and class. If the calling function
needs per sample loss the returned loss should be multiplied by the batch size.

"""

import logging
from abc import abstractmethod, ABC
from typing import Dict, Tuple

try:
    import FastGeodis

    is_geo_installed = True
except ImportError:
    is_geo_installed = False

try:
    import kornia
except ImportError:
    if not is_geo_installed:
        msg = f"Install kornia or FastGeodis packages required for GPU distance transform."
        raise ImportError(msg)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Global Constants
# ---------------------------
EPSILON = 1e-6


# ---------------------------
# Utilities
# ---------------------------

def one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts a segmentation mask to a one-hot encoded tensor.
    Args:
        mask (torch.Tensor): Segmentation mask of shape (B, H, W) with class indices.
        num_classes (int): The total number of classes.
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (B, C, H, W).
    """
    if mask.shape[1] == num_classes:
        return mask.float()
    return F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()


def sobel_grad(img: torch.Tensor) -> torch.Tensor:
    """
    Computes the Sobel gradient magnitude of a single-channel image.
    Args:
        img (torch.Tensor): Image tensor of shape (B, 1, H, W).
    Returns:
        torch.Tensor: Gradient magnitude tensor of the same shape.
    """
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    # Apply 2D convolution with padding
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + EPSILON)


def make_soft_boundary(prob: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Creates a soft boundary map from a probability map using Sobel gradient.
    Args:
        prob (torch.Tensor): Probability tensor of shape (B, 1, H, W) in [0,1].
        tau (float): Tau parameter for sharpening the probability map before
                     computing the gradient.
    Returns:
        torch.Tensor: The soft boundary map.
    """
    if tau != 1.0:
        prob = torch.clamp(prob, EPSILON, 1 - EPSILON)
        prob = torch.pow(prob, 1.0 / tau)
    e = sobel_grad(prob)
    # Normalize per-map to [0,1] for stable weighting
    return e / (e.amax(dim=(-1, -2), keepdim=True) + EPSILON)


# ---------------------------
# Distance transform backends (GPU)
# ---------------------------

class DistanceTransform2D:
    """
    Wrapper for different GPU-accelerated 2D distance transform backends.
    Supports Kornia (Euclidean) and FastGeodis (Geodesic with lambda=0 for Euclidean).
    """

    def __init__(self, backend: str = "kornia", spacing: Tuple[float, float] = (1.0, 1.0)):
        """
        Args:
            backend (str): The backend to use, either "kornia" or "fastgeodis".
            spacing (Tuple[float, float]): Pixel spacing (unused by Kornia).
        """
        self.backend = backend
        self.spacing = spacing
        if backend == "kornia":
            try:
                self.dt_fn = kornia.contrib.distance_transform
            except ImportError as e:
                raise ImportError("Install kornia for GPU distance transform: pip install kornia") from e
        elif backend == "fastgeodis":
            try:
                self.geodis_fn = FastGeodis.generalised_geodesic2d
            except ImportError as e:
                raise ImportError("Install FastGeodis for GPU distance transform: pip install FastGeodis") from e
        else:
            raise ValueError("backend must be 'kornia' or 'fastgeodis'")

    @torch.no_grad()
    def edt(self, binary: torch.Tensor) -> torch.Tensor:
        """
        Computes Euclidean Distance Transform (EDT) for a binary mask.
        Args:
            binary (torch.Tensor): Binary mask of shape (B, 1, H, W) with values {0, 1}.
        Returns:
            torch.Tensor: EDT map where each pixel value is the Euclidean distance
                          to the nearest '1' pixel.
        """
        n_batch = binary.shape[0]

        if self.backend == "kornia":
            # Kornia computes distance to zeros, so we invert the input
            return self.dt_fn(1.0 - binary)
        else:
            # FastGeodis computes distance to seeds, so we use the binary mask as seeds
            I = torch.zeros_like(binary)
            S = binary.float()
            result = torch.zeros_like(binary)
            for b_i in range(n_batch):
                # Generalised geodesic with v=1e10, lambda=0 approximates Euclidean DT
                result[b_i] = self.geodis_fn(I[b_i].unsqueeze(0),
                                             S[b_i].unsqueeze(0),
                                             v=1e10,
                                             lamb=0,
                                             iter=2)
            # --- NORMALIZATION ADDED HERE ---
            # Normalizing distance maps to [0, 1] for stable loss values
            max_result = result.amax(dim=(-1, -2), keepdim=True) + EPSILON
            return result/max_result

    @torch.no_grad()
    def signed_distance(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the Signed Distance Transform (SDT) of a mask.
        Args:
            mask (torch.Tensor): Mask of shape (B, 1, H, W) with values {0, 1}.
        Returns:
            torch.Tensor: Signed distance map (positive inside, negative outside).
        """
        d_fg = self.edt(mask)
        d_bg = self.edt(1.0 - mask)
        return d_bg - d_fg


# ---------------------------
# Loss Components
# ---------------------------

class LossException(Exception):
    def __init__(self, message, errors=None):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        """ Custom errors that the caller can print from e.errors"""
        self.errors = errors


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


class IoULoss(BaseLoss):
    """Calculates the Intersection over Union (IoU) Loss."""

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = super().setup(predicted, target)

        tp = scores[0]
        fp = scores[1]
        fn = scores[2]
        iou = (tp + EPSILON) / (tp + fp + fn + EPSILON)

        return 1 - iou


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

            # --- NORMALIZATION ADDED HERE ---
            # Normalizing distance maps to [0, 1] for stable loss values
            # max_dist_to_gt = d_to_gt.amax(dim=(-1, -2), keepdim=True) + EPSILON
            # max_dist_to_pred = d_to_pred.amax(dim=(-1, -2), keepdim=True) + EPSILON
            # d_to_gt = d_to_gt / max_dist_to_gt
            # d_to_pred = d_to_pred / max_dist_to_pred

            # Chamfer-style symmetric loss terms
            term_pred_to_gt = (e_pred * d_to_gt).sum(dim=(2, 3)) / (e_pred.sum(dim=(2, 3)) + EPSILON)
            term_gt_to_pred = (e_gt * d_to_pred).sum(dim=(2, 3)) / (e_gt.sum(dim=(2, 3)) + EPSILON)
            loss_c = 0.5 * (term_pred_to_gt.mean() + term_gt_to_pred.mean())
            if torch.isnan(loss_c).any():
                loss_c = torch.zeros_like(loss_c)
            loss += loss_c

        return loss / num_classes


class LossFactory:
    """
    A factory class to create instances of loss functions by name.
    This centralizes loss function creation and configuration.
    """
    _loss_map = {
        'dice': DiceLoss,
        'iou': IoULoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'boundary_sdf': BoundarySDFLoss,
        'soft_chamfer': SoftChamferLoss,
        'ce': nn.CrossEntropyLoss,
        'bce': nn.BCEWithLogitsLoss
    }

    @staticmethod
    def create(loss_name: str, **kwargs) -> nn.Module:
        """
        Creates a loss function instance.
        Args:
            loss_name (str): The name of the loss function.
            **kwargs: Additional arguments for the loss function's constructor.
        Returns:
            nn.Module: An instance of the requested loss function.
        Raises:
            ValueError: If the loss name is not supported.
        """
        if loss_name not in LossFactory._loss_map:
            raise ValueError(f"Loss '{loss_name}' not supported. "
                             f"Supported losses are: {list(LossFactory._loss_map.keys())}")
        return LossFactory._loss_map[loss_name](**kwargs)


class HybridLoss(BaseLoss):
    """
    Combines multiple loss functions with configurable weights.
    The list of losses to use and their weights are passed in a dictionary.
    """

    def __init__(self, loss_configs: Dict[str, Dict[str, float]]):
        """
        Args:
            loss_configs (Dict[str, Dict[str, float]]):
                A dictionary where keys are loss names and values are dictionaries
                containing 'weight' and any other loss-specific arguments.
                Example: {'dice': {'weight': 0.5}, 'ce': {'weight': 0.5}}.
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict()
        self.weights = {}

        for name, config in loss_configs.items():
            weight = config.pop('weight', 1.0)
            self.weights[name] = weight
            self.loss_functions[name] = LossFactory.create(name, **config)
            logging.info(f"Initialized loss '{name}' with weight {weight} and config {config}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _ = super().setup(pred, target)
        device = pred.device

        total_loss = torch.tensor(0.0, device=device)
        for name, loss_fn in self.loss_functions.items():
            current_loss = loss_fn(pred,
                                   self.targets)
            total_loss += self.weights[name] * current_loss

        return total_loss


if __name__ == '__main__':
    # ---------------------------
    # Test Calls for all functions and classes
    # ---------------------------

    n_test_epochs = 10

    # Setup for tests
    n_batch, h, w, n_classes = 3, 128, 128, 11
    print("\n--- Creating Individual Components ---")
    dice_loss_fn = LossFactory.create('dice')
    iou_loss_fn = LossFactory.create('iou')
    focal_loss_fn = LossFactory.create('focal', alpha=0.5, gamma=3.0)
    tversky_loss_fn = LossFactory.create('tversky', alpha=0.3, beta=0.7)
    ce_loss_fn = LossFactory.create('ce')
    boundary_sdf_loss_fn = LossFactory.create('boundary_sdf', dt_backend="kornia")
    soft_chamfer_loss_fn = LossFactory.create('soft_chamfer', dt_backend="kornia")

    loss_config_binary = {
        'bce': {'weight': 0.4},
        'ce': {'weight': 0.4},
        'tversky': {'weight': 0.6},
        'dice': {'weight': 0.5},
        'boundary_sdf': {'weight': 0.1, 'dt_backend': 'fastgeodis'}
    }
    hybrid_loss_binary = HybridLoss(loss_config_binary)

    loss_config_multi = {
        'ce': {'weight': 0.5},
        'dice': {'weight': 0.4},
        'focal': {'weight': 0.5, 'alpha': 1.0, 'gamma': 3.0},
        'boundary_sdf': {'weight': 0.1, 'dt_backend': 'kornia'}
    }
    hybrid_loss_multi = HybridLoss(loss_config_multi)

    dt_kornia = DistanceTransform2D(backend="kornia")
    dt_fastgeodis = DistanceTransform2D(backend="fastgeodis")

    print(f"Running {n_test_epochs} test epochs.")
    for n in range(n_test_epochs):
        print(f"Test epoch [{n+1} / {n_test_epochs}]")
        test_logits = torch.randn(n_batch, n_classes, h, w)
        test_mask = torch.randint(0, n_classes, (n_batch, h, w))
        test_mask = one_hot(test_mask, n_classes)
        binary_logits = torch.randn(n_batch, n_classes, h, w)
        binary_mask = torch.randint(0, n_classes, (n_batch, h, w))

        print("--- Testing Utility Functions ---")
        test_one_hot = one_hot(test_mask, n_classes)
        print(f"one_hot output shape: {test_one_hot.shape}")
        test_sobel_grad = sobel_grad(torch.randn(1, 1, h, w))
        print(f"sobel_grad output shape: {test_sobel_grad.shape}")
        test_soft_boundary = make_soft_boundary(torch.rand(1, 1, h, w))
        print(f"make_soft_boundary output shape: {test_soft_boundary.shape}")

        print("\n--- Testing DistanceTransform2D ---")
        # binary_map = (test_mask[:, None, :, :] == 1).float()
        binary_map = test_mask.float()
        print(f"DistanceTransform2D (kornia) EDT: {dt_kornia.edt(binary_map).mean().item():.4f}")
        print(f"DistanceTransform2D (fastgeodis) EDT: {dt_fastgeodis.edt(binary_map).mean().item():.4f}")
        print(f"DistanceTransform2D (kornia) SDT: {dt_kornia.signed_distance(binary_map).mean().item():.4f}")
        print(f"DistanceTransform2D (fastgeodis) SDT: {dt_fastgeodis.signed_distance(binary_map).mean().item():.4f}")


        print(f"Dice Loss: {dice_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"IoU Loss: {iou_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Focal Loss: {focal_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Tversky Loss: {tversky_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Cross-Entropy Loss: {ce_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Boundary SDF Loss: {boundary_sdf_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Soft Chamfer Loss: {soft_chamfer_loss_fn(test_logits, test_mask).item():.4f}")

        print("\n--- Testing HybridLoss Class ---")
        hybrid_total_loss_multi = hybrid_loss_multi(test_logits, test_mask)
        print(f"HybridLoss (multi-class) total loss: {hybrid_total_loss_multi.item():.4f}")

        hybrid_total_loss_binary = hybrid_loss_binary(binary_logits, binary_mask)
        print(f"HybridLoss (binary) total loss: {hybrid_total_loss_binary.item():.4f}")

    # ------------------------------
    """ Test for Perfect matches """
    # ------------------------------
    perfect_pred = torch.zeros(n_batch, n_classes, h, w)
    perfect_target = torch.zeros(n_batch, n_classes, h, w).long()
    perfect_pred[:n_batch, 0, :, :] = 100.0  # Set logits for class 0 to be very high
    perfect_target[:n_batch] = 0  # Ground truth is class 0 everywhere

    print("\n--- Testing DiceLoss with a perfect prediction ---")
    perfect_loss = dice_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "DiceLoss for perfect prediction is not near zero!"

    print("\n--- Testing IoULoss with a perfect prediction ---")
    perfect_loss = iou_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "IoULoss for perfect prediction is not near zero!"

    print("\n--- Testing FocalLoss with a perfect prediction ---")
    perfect_loss = focal_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "FocalLoss for perfect prediction is not near zero!"

    print("\n--- Testing TverskyLoss with a perfect prediction ---")
    perfect_loss = tversky_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "TverskyLoss for perfect prediction is not near zero!"

    print("\n--- Testing CrossEntropyLoss with a perfect prediction ---")
    perfect_loss = ce_loss_fn(perfect_pred,
                              perfect_target.float()).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "CrossEntropyLoss for perfect prediction is not near zero!"

    # Boundary SDF Loss
    print("\n--- Testing BoundarySDFLoss with perfect prediction ---")
    perfect_sdf_loss = boundary_sdf_loss_fn(perfect_pred, perfect_target).item()
    print(f"Boundary SDF Loss for perfect prediction: {perfect_sdf_loss:.6f}")
    assert perfect_sdf_loss < 1e-5, "Boundary SDF Loss for perfect prediction is not near zero!"

    print("\n--- Testing SoftChamferLoss with a perfect prediction ---")
    perfect_loss = soft_chamfer_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "SoftChamferLoss for perfect prediction is not near zero!"

    print("\n--- Testing HybridLoss with a perfect prediction ---")
    perfect_loss = hybrid_loss_multi(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "SoftChamferLoss for perfect prediction is not near zero!"


