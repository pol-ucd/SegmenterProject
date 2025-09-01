import logging
from abc import abstractmethod, ABC
from typing import Optional, Dict, Tuple

import FastGeodis
import kornia
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
    def __init__(self, backend: str = "fastgeodis", spacing: Tuple[float, float] = (1.0, 1.0)):
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

            return result

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

class BaseLoss(nn.Module, ABC):
    """Abstract base class for all loss functions."""
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the loss."""
        pass


class DiceLoss(BaseLoss):
    """Calculates the multi-class Dice Loss."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = max(pred.shape[1], 2)

        probs = F.softmax(pred, dim=1)
        if target.shape[1] == 1:
            target_onehot = one_hot(target, num_classes=num_classes).to(pred.dtype)
        else:
            target_onehot = target.to(pred.dtype)
        # Vectorized calculation
        intersection = (probs * target_onehot).sum(dim=(2, 3))
        union = (probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)))
        dice_score = (2.0 * intersection + EPSILON) / (union + EPSILON)
        return 1 - dice_score.mean()


class IoULoss(BaseLoss):
    """Calculates the Intersection over Union (IoU) Loss."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = max(pred.shape[1], 2)
        probs = F.softmax(pred, dim=1)
        if target.shape[1] == 1:
            target_onehot = one_hot(target, num_classes=num_classes).to(pred.dtype)
        else:
            target_onehot = target.to(pred.dtype)
        # Vectorized calculation
        intersection = (probs * target_onehot).sum(dim=(2, 3))
        union = (probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))) - intersection
        iou_score = (intersection + EPSILON) / (union + EPSILON)
        return 1 - iou_score.mean()


class FocalLoss(BaseLoss):
    """Calculates the Focal Loss."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = max(pred.shape[1], 2)
        alpha = self.kwargs.get('alpha', 0.25)
        gamma = self.kwargs.get('gamma', 2.0)
        probs = F.softmax(pred, dim=1)
        if target.shape[1] == 1:
            target_onehot = one_hot(target, num_classes=num_classes).to(pred.dtype)
        else:
            target_onehot = target.to(pred.dtype)
        pt = torch.where(target_onehot == 1, probs, 1 - probs)
        focal_term = alpha * (1 - pt) ** gamma
        bce = -torch.log(pt + EPSILON)
        return (focal_term * bce).mean()


class TverskyLoss(BaseLoss):
    """Calculates the Tversky Loss."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = max(pred.shape[1], 2)

        alpha = self.kwargs.get('alpha', 0.5)
        beta = self.kwargs.get('beta', 0.5)
        probs = F.softmax(pred, dim=1)
        if target.shape[1] == 1:
            target_oh = one_hot(target, num_classes=num_classes).to(pred.dtype)
        else:
            target_oh = target.to(pred.dtype)
        # Vectorized calculation over all dimensions
        TP = (probs * target_oh).sum(dim=(0, 2, 3))
        FP = (probs * (1 - target_oh)).sum(dim=(0, 2, 3))
        FN = ((1 - probs) * target_oh).sum(dim=(0, 2, 3))
        tversky = (TP + EPSILON) / (TP + alpha * FP + beta * FN + EPSILON)
        return 1.0 - tversky.mean()


class BoundarySDFLoss(BaseLoss):
    """
    Boundary Loss based on Signed Distance Functions (SDF).
    Penalizes misalignment between predicted and ground truth boundaries.
    """
    def __init__(self, dt_backend: str = "fastgeodis", **kwargs):
        super().__init__(**kwargs)
        self.dt = DistanceTransform2D(backend=dt_backend)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = max(pred.shape[1], 2)
        is_one_class = (pred.shape[1] == 1)
        probs = F.softmax(pred, dim=1)
        if target.shape[1] == 1:
            target_onehot = one_hot(target, num_classes=num_classes).to(pred.dtype)
        else:
            target_onehot = target.to(pred.dtype)
        loss = 0.0
        for c in range(num_classes):
            gt_c = target_onehot[:, c:c + 1]
            sdt_c = self.dt.signed_distance(gt_c)
            # Normalize SDF per-sample for scale invariance
            sdt_c = sdt_c / (sdt_c.amax(dim=(-1, -2), keepdim=True) + EPSILON)

            # Penalize the difference between prediction and ground truth,
            # weighted by the absolute signed distance. This ensures zero loss
            # for a perfect match, regardless of the SDF value.
            loss += (sdt_c.abs() * (probs[:, c:c + 1] - gt_c).abs()).mean()

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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = max(pred.shape[1], 2)
        probs = F.softmax(pred, dim=1)
        if target.shape[1] == 1:
            target_onehot = one_hot(target, num_classes=num_classes).to(pred.dtype)
        else:
            target_onehot = target.to(pred.dtype)
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
            max_dist_to_gt = d_to_gt.amax(dim=(-1, -2), keepdim=True) + EPSILON
            max_dist_to_pred = d_to_pred.amax(dim=(-1, -2), keepdim=True) + EPSILON
            d_to_gt = d_to_gt / max_dist_to_gt
            d_to_pred = d_to_pred / max_dist_to_pred

            # Chamfer-style symmetric loss terms
            term_pred_to_gt = (e_pred * d_to_gt).sum(dim=(2, 3)) / (e_pred.sum(dim=(2, 3)) + EPSILON)
            term_gt_to_pred = (e_gt * d_to_pred).sum(dim=(2, 3)) / (e_gt.sum(dim=(2, 3)) + EPSILON)
            loss += 0.5 * (term_pred_to_gt.mean() + term_gt_to_pred.mean())

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


class HybridLoss(nn.Module):
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
        num_classes = max(pred.shape[1], 2)
        total_loss = torch.tensor(0.0, device=pred.device)
        # Ensure target is of the correct format for multi-class losses
        if target.dim() == pred.dim():
            target = target.squeeze(1)

        for name, loss_fn in self.loss_functions.items():
            if name == 'bce':
                # BCE expects pred with shape (B, 1, H, W) and target (B, 1, H, W)
                current_loss = loss_fn(pred, target.unsqueeze(1).type(pred.dtype))
            elif name == 'ce':
                if num_classes == 2:
                    current_loss = loss_fn(pred, target.type(pred.dtype))
                else:
                    current_loss = loss_fn(pred,
                                           one_hot(target, num_classes))
            else:
                current_loss = loss_fn(pred,
                                       target)
            total_loss += self.weights[name] * current_loss

        return total_loss


class EarlyStopping:
    """
    Early Stopping Implementation to halt training when a metric stops improving.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 mode: str = 'min', verbose: bool = False,
                 save_path: Optional[str] = None):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' for loss, 'max' for accuracy or IoU.
            verbose (bool): If True, logs updates.
            save_path (str): If set, saves best model to this path.
        """
        assert mode in ['min', 'max'], "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path
        self.logger = logging.getLogger(self.__class__.__name__)

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_epoch: Optional[int] = None

        self._init_comparator()

    def _init_comparator(self):
        """Initializes the comparison logic based on mode."""
        if self.mode == 'min':
            self.compare = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf')
        else:
            self.compare = lambda current, best: current > best + self.min_delta
            self.best_score = float('-inf')

    def __call__(self, current_score: float, model: Optional[nn.Module] = None, epoch: Optional[int] = None):
        """
        Evaluates the current score against the best score and updates state.
        Args:
            current_score (float): The score from the current epoch.
            model (Optional[nn.Module]): The model to save if a new best score is found.
            epoch (Optional[int]): The current epoch number.
        """
        if self.best_score is None or self.compare(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                log_message = f"New best score: {current_score:.4f}"
                if epoch is not None:
                    log_message += f" at epoch {epoch + 1}"
                self.logger.info(log_message)

            if self.save_path and model is not None:
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    self.logger.info(f"Model saved to {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f"No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose and epoch is not None:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")

    def reset(self):
        """Resets the state of the EarlyStopping instance."""
        self.counter = 0
        self.early_stop = False
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = None


if __name__ == '__main__':
    # ---------------------------
    # Test Calls for all functions and classes
    # ---------------------------

    # Setup for tests
    n_batch, h, w, n_classes = 4, 128, 128, 3
    test_logits = torch.randn(n_batch, n_classes, h, w)
    test_mask = torch.randint(0, n_classes, (n_batch, h, w))
    binary_logits = torch.randn(n_batch, 1, h, w)
    binary_mask = torch.randint(0, 2, (n_batch, h, w))

    print("--- Testing Utility Functions ---")
    test_one_hot = one_hot(test_mask, n_classes)
    print(f"one_hot output shape: {test_one_hot.shape}")
    test_sobel_grad = sobel_grad(torch.randn(1, 1, h, w))
    print(f"sobel_grad output shape: {test_sobel_grad.shape}")
    test_soft_boundary = make_soft_boundary(torch.rand(1, 1, h, w))
    print(f"make_soft_boundary output shape: {test_soft_boundary.shape}")

    print("\n--- Testing DistanceTransform2D ---")
    dt_kornia = DistanceTransform2D(backend="kornia")
    dt_fastgeodis = DistanceTransform2D(backend="fastgeodis")
    binary_map = (test_mask[:, None, :, :] == 1).float()
    print(f"DistanceTransform2D (kornia) EDT: {dt_kornia.edt(binary_map).mean().item():.4f}")
    print(f"DistanceTransform2D (fastgeodis) EDT: {dt_fastgeodis.edt(binary_map).mean().item():.4f}")
    print(f"DistanceTransform2D (kornia) SDT: {dt_kornia.signed_distance(binary_map).mean().item():.4f}")
    print(f"DistanceTransform2D (fastgeodis) SDT: {dt_fastgeodis.signed_distance(binary_map).mean().item():.4f}")

    print("\n--- Testing Individual Loss Components (via LossFactory) ---")
    dice_loss_fn = LossFactory.create('dice')
    iou_loss_fn = LossFactory.create('iou')
    focal_loss_fn = LossFactory.create('focal', alpha=0.5, gamma=3.0)
    tversky_loss_fn = LossFactory.create('tversky', alpha=0.3, beta=0.7)
    ce_loss_fn = LossFactory.create('ce')
    boundary_sdf_loss_fn = LossFactory.create('boundary_sdf', dt_backend="fastgeodis")
    soft_chamfer_loss_fn = LossFactory.create('soft_chamfer', dt_backend="fastgeodis")

    print(f"Dice Loss: {dice_loss_fn(test_logits, test_mask).item():.4f}")
    print(f"IoU Loss: {iou_loss_fn(test_logits, test_mask).item():.4f}")
    print(f"Focal Loss: {focal_loss_fn(test_logits, test_mask).item():.4f}")
    print(f"Tversky Loss: {tversky_loss_fn(test_logits, test_mask).item():.4f}")
    print(f"Cross-Entropy Loss: {ce_loss_fn(test_logits, test_mask).item():.4f}")
    print(f"Boundary SDF Loss: {boundary_sdf_loss_fn(test_logits, test_mask).item():.4f}")
    print(f"Soft Chamfer Loss: {soft_chamfer_loss_fn(test_logits, test_mask).item():.4f}")

    print("\n--- Testing DiceLoss with a perfect prediction ---")
    perfect_pred = torch.zeros(1, n_classes, h, w)
    perfect_target = torch.zeros(1, h, w).long()
    perfect_pred[0, 0, :, :] = 100.0  # Set logits for class 0 to be very high
    perfect_target[0, :, :] = 0  # Ground truth is class 0 everywhere
    perfect_loss = dice_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "DiceLoss for perfect prediction is not near zero!"

    print("\n--- Testing IoULoss with a perfect prediction ---")
    perfect_pred = torch.zeros(1, n_classes, h, w)
    perfect_target = torch.zeros(1, h, w).long()
    perfect_pred[0, 0, :, :] = 100.0  # Set logits for class 0 to be very high
    perfect_target[0, :, :] = 0  # Ground truth is class 0 everywhere
    perfect_loss = iou_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "IoULoss for perfect prediction is not near zero!"

    print("\n--- Testing FocalLoss with a perfect prediction ---")
    perfect_pred = torch.zeros(1, n_classes, h, w)
    perfect_target = torch.zeros(1, h, w).long()
    perfect_pred[0, 0, :, :] = 100.0  # Set logits for class 0 to be very high
    perfect_target[0, :, :] = 0  # Ground truth is class 0 everywhere
    perfect_loss = focal_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "FocalLoss for perfect prediction is not near zero!"


    print("\n--- Testing TverskyLoss with a perfect prediction ---")
    perfect_pred = torch.zeros(1, n_classes, h, w)
    perfect_target = torch.zeros(1, h, w).long()
    perfect_pred[0, 0, :, :] = 100.0  # Set logits for class 0 to be very high
    perfect_target[0, :, :] = 0  # Ground truth is class 0 everywhere
    perfect_loss = tversky_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "TverskyLoss for perfect prediction is not near zero!"

    print("\n--- Testing CrossEntropyLoss with a perfect prediction ---")
    perfect_pred = torch.zeros(1, n_classes, h, w)
    perfect_target = torch.zeros(1, h, w).long()
    perfect_pred[0, 0, :, :] = 100.0  # Set logits for class 0 to be very high
    perfect_target[0, :, :] = 0  # Ground truth is class 0 everywhere
    perfect_loss = ce_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "CrossEntropyLoss for perfect prediction is not near zero!"

    # Boundary SDF Loss
    print("\n--- Testing BoundarySDFLoss with perfect prediction ---")
    perfect_pred = torch.zeros(1, n_classes, h, w)
    perfect_target = torch.zeros(1, h, w).long()
    # Set logits for class 0 to be very high, predicting the correct class
    perfect_pred[0, 0, :, :] = 100.0
    perfect_target[0, :, :] = 0
    perfect_sdf_loss = boundary_sdf_loss_fn(perfect_pred, perfect_target).item()
    print(f"Boundary SDF Loss for perfect prediction: {perfect_sdf_loss:.6f}")
    assert perfect_sdf_loss < 1e-5, "Boundary SDF Loss for perfect prediction is not near zero!"

    print("\n--- Testing SoftChamferLoss with a perfect prediction ---")
    perfect_pred = torch.zeros(1, n_classes, h, w)
    perfect_target = torch.zeros(1, h, w).long()
    perfect_pred[0, 0, :, :] = 100.0  # Set logits for class 0 to be very high
    perfect_target[0, :, :] = 0  # Ground truth is class 0 everywhere
    perfect_loss = soft_chamfer_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "SoftChamferLoss for perfect prediction is not near zero!"

    print("\n--- Testing HybridLoss Class ---")
    loss_config_multi = {
        'ce': {'weight': 0.5},
        'dice': {'weight': 0.5},
        'boundary_sdf': {'weight': 0.1, 'dt_backend': 'fastgeodis'}
    }
    hybrid_loss_multi = HybridLoss(loss_config_multi)
    hybrid_total_loss_multi = hybrid_loss_multi(test_logits, test_mask)
    print(f"HybridLoss (multi-class) total loss: {hybrid_total_loss_multi.item():.4f}")

    loss_config_binary = {
        'bce': {'weight': 0.4},
        'ce': {'weight': 0.4},
        'tversky': {'weight': 0.6},
        'dice': {'weight': 0.5},
        'boundary_sdf': {'weight': 0.1, 'dt_backend': 'fastgeodis'}
    }
    hybrid_loss_binary = HybridLoss(loss_config_binary)
    hybrid_total_loss_binary = hybrid_loss_binary(binary_logits, binary_mask)
    print(f"HybridLoss (binary) total loss: {hybrid_total_loss_binary.item():.4f}")

    print("\n--- Testing EarlyStopping Class ---")
    es = EarlyStopping(patience=3, min_delta=0.01, verbose=True, mode='min')
    print("Initial state:")
    es(0.5)
    es(0.48)
    es(0.47)
    es(0.46)
    print(f"Early stop flag: {es.early_stop}")
    es(0.47) # No improvement
    es(0.48) # No improvement
    es(0.49) # Trigger early stop
    print(f"Early stop flag: {es.early_stop}")
    es.reset()
    print(f"After reset, early stop flag: {es.early_stop}")
