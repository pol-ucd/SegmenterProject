import logging
from abc import abstractmethod, ABC
from typing import Optional

import FastGeodis
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------

def one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    # mask: (B, H, W) int64 in [0..C-1] -> (B, C, H, W) float
    return F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()


def sobel_grad(img: torch.Tensor) -> torch.Tensor:
    # img: (B, 1, H, W)
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def make_soft_boundary(prob: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    # prob: (B, 1, H, W) in [0,1]; sharpen to emphasize edges, then Sobel
    if tau != 1.0:
        prob = torch.clamp(prob, 1e-6, 1 - 1e-6)
        prob = torch.pow(prob, 1.0 / tau)
    e = sobel_grad(prob)
    # normalize per-map to [0,1] for stable weighting
    e = e / (e.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return e


# ---------------------------
# Distance transform backends (GPU)
# Choose either Kornia or FastGeodis; both are CUDA-friendly.
# ---------------------------

class DistanceTransform2D:
    def __init__(self, backend: str = "kornia", spacing=(1.0, 1.0)):
        self.backend = backend
        self.spacing = spacing
        if backend == "kornia":
            try:
                import kornia
                self.kornia = kornia
            except Exception as e:
                raise ImportError("Install kornia for GPU distance transform: pip install kornia") from e
        elif backend == "fastgeodis":
            try:
                import FastGeodis  # noqa
                self.FastGeodis = FastGeodis
            except Exception as e:
                raise ImportError("Install FastGeodis: pip install FastGeodis") from e
        else:
            raise ValueError("backend must be 'kornia' or 'fastgeodis'")

    @torch.no_grad()
    def edt(self, binary: torch.Tensor) -> torch.Tensor:
        # binary: (B,1,H,W) in {0,1}; returns Euclidean distance to nearest 1-pixel.
        if self.backend == "kornia":
            # Kornia expects float; computes distance to zeros or ones depending on function.
            # We compute distance to foreground by inverting once and once more for background as needed outside.
            # kornia.contrib.distance_transform expects 1 for foreground; it returns distance to zero-valued background.
            # To get distance to foreground: invert.
            from kornia.contrib import distance_transform
            # distance to foreground (1s): invert so foreground becomes 0, then DT to zeros:
            inv = 1.0 - binary
            dt = distance_transform(inv)
            return dt
        else:
            # FastGeodis generalized geodesic with lambda=0 behaves like Euclidean DT over 4/8-connectivity
            # Seeds are foreground; costs uniform.
            import FastGeodis
            # FastGeodis expects (B,1,H,W) float32
            I = torch.zeros_like(binary)  # uniform cost
            S = (binary > 0.5).float()
            # geodesic2d(img, seeds, spacing, v, l, iter)
            dt = FastGeodis.generalised_geodesic2d(I, S, v=1e10, lamb=0, iter=2)
            return dt

    @torch.no_grad()
    def signed_distance(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (B,1,H,W) in {0,1}; returns D_bg - D_fg (positive inside, negative outside)
        d_fg = self.edt(mask)
        d_bg = self.edt(1.0 - mask)
        return d_bg - d_fg


# ---------------------------
# Loss terms
# ---------------------------

def boundary_sdf_loss(pred_probs: torch.Tensor, gt_one_hot: torch.Tensor, dt: DistanceTransform2D) -> torch.Tensor:
    # pred_probs: (B,C,H,W), gt_one_hot: (B,C,H,W)
    B, C, H, W = pred_probs.shape
    loss = 0.0
    for c in range(C):
        gt_c = gt_one_hot[:, c:c + 1]
        sdt_c = dt.signed_distance(gt_c)  # (B,1,H,W), detached by @no_grad
        # normalize SDF per-sample for scale invariance
        sdt_c = sdt_c / (sdt_c.amax(dim=(-1, -2), keepdim=True) + 1e-6)
        loss += (pred_probs[:, c:c + 1] * sdt_c).mean()
    return loss / C


def soft_chamfer_surface_loss(pred_probs: torch.Tensor, gt_one_hot: torch.Tensor, dt: DistanceTransform2D,
                              tau: float = 0.8, band_px: int = 3) -> torch.Tensor:
    # Build GT boundary maps (binary). Use morphological gradient via max-pool.
    B, C, H, W = pred_probs.shape
    pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    loss = 0.0
    eps = 1e-8

    for c in range(C):
        p = pred_probs[:, c:c + 1]  # (B,1,H,W)
        m = gt_one_hot[:, c:c + 1]  # (B,1,H,W)

        # GT boundary: dilation - erosion (approx via maxpool of mask and of inverted mask)
        dil = pool(m)
        ero = 1.0 - pool(1.0 - m)
        e_gt = torch.clamp(dil - ero, 0, 1)  # binary-ish boundary

        # Narrow band (optional): widen boundary to band_px to relax supervision
        if band_px > 0:
            k = 2 * band_px + 1
            band = torch.nn.MaxPool2d(k, stride=1, padding=band_px)(e_gt)
            e_gt_band = band
        else:
            e_gt_band = e_gt

        # Soft predicted boundary
        e_pred = make_soft_boundary(p, tau=tau)  # [0,1]

        # Distances: GT-edge distances used by pred-edge and vice-versa.
        # Note: detach distances computed from pred for stability (optional).
        d_to_gt = dt.edt(e_gt_band)  # distance to GT edge
        d_to_pred = dt.edt((e_pred > 0.01).float())  # binarize soft edge for DT
        d_to_pred = d_to_pred.detach()  # prevent exploding grads via DT backprop

        # Chamfer-style symmetric surface loss
        term_pred_to_gt = (e_pred * d_to_gt).sum(dim=(2, 3)) / (e_pred.sum(dim=(2, 3)) + eps)
        term_gt_to_pred = (e_gt * d_to_pred).sum(dim=(2, 3)) / (e_gt.sum(dim=(2, 3)) + eps)
        loss += 0.5 * (term_pred_to_gt.mean() + term_gt_to_pred.mean())

    return loss / C


# ---------------------------
# Full hybrid loss
# ---------------------------

class HybridLesionLoss(torch.nn.Module):
    def __init__(self, num_classes: int, alpha: float = 1.0, beta: float = 1.0, delta: float = 0.5,
                 dt_backend: str = "kornia", spacing=(1.0, 1.0), tau: float = 0.8, band_px: int = 3):
        super().__init__()
        self.C = num_classes
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.dt = DistanceTransform2D(backend=dt_backend, spacing=spacing)
        self.tau = tau
        self.band_px = band_px

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        # logits: (B,C,H,W), gt_mask: (B,H,W)
        probs = F.softmax(logits, dim=1)
        gt = one_hot(gt_mask, self.C).to(probs.dtype)

        Lb = boundary_sdf_loss(probs, gt, self.dt)
        Ls = soft_chamfer_surface_loss(probs, gt, self.dt, tau=self.tau, band_px=self.band_px)
        Lo = dice_loss(probs, gt)

        return self.alpha * Lb + self.beta * Ls + self.delta * Lo


def boundary_loss(pred, mask):
    """
    Boundary Loss
    Penalizes misalignment between predicted and ground truth boundaries using distance maps.
    Boundary loss rewards pixel-level alignment.

    :param pred:
    :param mask:
    :param num_classes:
    :return:
    """
    num_batch = pred.shape[0]
    num_classes = pred.shape[1]
    one_hot_mask = one_hot(mask, num_classes)
    dist_maps = torch.zeros_like(one_hot_mask)

    for b in range(num_batch):
        for c in range(num_classes):
            mask_c = one_hot_mask[b, c, :, :].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

            dist_map = FastGeodis.generalised_geodesic2d(mask_c, mask_c,
                                                         v=1e10,
                                                         lamb=0.0,
                                                         iter=2)

            dist_maps[b, c, :, :] = dist_map

    loss = (pred * dist_maps).abs().sum(dim=(2, 3)).mean()
    return loss


def fourier_loss(pred_mask, true_mask):
    """
    Shape-Aware Loss
    Penalizes shape discrepancies using Fourier descriptors or signed distance fields (SDFs).
    extract contours and compute Fourier descriptors for both prediction and ground truth,
    then penalize their difference. Fourier loss enforces global shape integrity.
    Assume binary masks for each class.

    :param pred_mask:
    :param true_mask:
    :return:
    """

    def compute_fd(mask):
        coords = torch.nonzero(mask)
        x, y = coords[:, 0].float(), coords[:, 1].float()
        complex_coords = x + 1j * y
        fd = torch.fft.fft(complex_coords)
        return fd

    loss = 0.0
    for b in range(pred_mask.shape[0]):
        fd_pred = compute_fd(pred_mask[b])
        fd_true = compute_fd(true_mask[b])
        loss += torch.norm(fd_pred - fd_true)
    return loss / pred_mask.shape[0]


def hybrid_loss(pred, mask, alpha=1.0, beta=1.0):
    pred_mask = torch.argmax(pred, dim=1)  # (B, H, W)
    boundary_element = boundary_loss(pred, mask)
    shape_element = fourier_loss(pred_mask, mask)
    return alpha * boundary_element + beta * shape_element


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
               smooth: float = 1e-6) -> torch.Tensor:
    """
    Computes the Focal Loss.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
        target (torch.Tensor): The ground truth mask tensor with class indices.
        alpha (float, optional): Alpha parameter for the focal loss.
        gamma (float, optional): Gamma parameter for the focal loss.
        smooth (float, optional): Small value to avoid log(0) issues.

    Returns:
        torch.Tensor: The calculated focal loss.
    """
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    pt = torch.where(target_onehot == 1, pred, 1 - pred)
    focal_term = alpha * (1 - pt) ** gamma
    bce = -torch.log(pt + smooth)
    return (focal_term * bce).mean()


def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, beta: float = 0.5,
                 smooth: float = 1e-6) -> torch.Tensor:
    """
    Computes the Tversky Loss.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
        target (torch.Tensor): The ground truth mask tensor with class indices.
        alpha (float, optional): False positive weight.
        beta (float, optional): False negative weight.
        smooth (float, optional): Small value to avoid division by zero.

    Returns:
        torch.Tensor: The calculated tversky loss.
    """
    probs = torch.softmax(pred, dim=1)
    target_oh = F.one_hot(target, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()

    # Vectorized calculation over all dimensions
    TP = (probs * target_oh).sum(dim=(0, 2, 3))
    FP = (probs * (1 - target_oh)).sum(dim=(0, 2, 3))
    FN = ((1 - probs) * target_oh).sum(dim=(0, 2, 3))

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss_tversky = 1.0 - tversky
    return loss_tversky.mean()


def dice_loss(pred: torch.Tensor,
              target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculates the multi-class Dice Loss.

    This implementation is fully vectorized for efficiency and follows the
    standard Dice Loss formula.

    Args:
        pred (torch.Tensor): The model's raw output prediction tensor.
                             Expected shape: (N, C, H, W).
        target (torch.Tensor): The ground truth mask tensor with class indices.
                                  Expected shape: (N, H, W).
        epsilon (float, optional): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The calculated multi-class Dice Loss.
    """
    num_classes = pred.shape[1]
    pred_prob = F.softmax(pred, dim=1)
    target_one_hot = one_hot(target, num_classes=num_classes)

    intersection = (pred_prob * target_one_hot).sum(dim=(2, 3))
    denom = pred_prob + target_one_hot
    denom = denom.sum(dim=(2, 3))

    dice_score = (2. * intersection + epsilon) / (denom + epsilon)
    return 1 - dice_score.mean()

def iou_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Computes the IoU Loss.

    Args:
        pred_probs (torch.Tensor): The model's raw output prediction tensor.
        target (torch.Tensor): The ground truth mask tensor with class indices.
        epsilon (float, optional): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The calculated IoU loss.
    """
    pred_prob = torch.softmax(pred, dim=1)
    target_onehot = one_hot(target, num_classes=pred_prob.shape[1])

    intersection = (pred_prob * target_onehot).sum(dim=(2, 3))
    union = pred_prob + target_onehot - (pred_prob * target_onehot)
    union = union.sum(dim=(2, 3))

    iou = (intersection + epsilon) / (union + epsilon)
    return 1 - iou.mean()


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all loss functions to ensure a consistent interface.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            pred (torch.Tensor): The model's output.
            target (torch.Tensor): The ground truth.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        pass


class BoundaryLoss(BaseLoss):
    """
    Loss based on the signed distance map from boundaries.

    Args:
        scale_factor (float): A factor to scale the final loss by, to
                              prevent it from dominating other losses.
    """

    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = boundary_loss(pred, target)
        return loss * self.scale_factor


class DiceLoss(BaseLoss):
    """Calculates the Dice Loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(pred, target, self.epsilon)


class FocalLoss(BaseLoss):
    """
    Calculates the Focal Loss with adjustable alpha and gamma parameters.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(pred, target, alpha=self.alpha, gamma=self.gamma, smooth=self.epsilon)


class TverskyLoss(BaseLoss):
    """
    Calculates the Tversky Loss with adjustable alpha and beta parameters.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tversky_loss(pred, target, alpha=self.alpha, beta=self.beta, smooth=self.epsilon)


class IoULoss(BaseLoss):
    """Calculates the Intersection over Union (IoU) Loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return iou_loss(pred, target, self.epsilon)


class HybridLoss(nn.Module):
    """
    Combines multiple loss functions with configurable weights.

    This class correctly initializes all loss components and applies them
    to the input tensors before combining.
    """

    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0,
                 weight_focal: float = 1.0, weight_tversky: float = 1.0,
                 weight_iou: float = 1.0, weight_boundary: float = 1.0):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_tversky = weight_tversky
        self.weight_iou = weight_iou
        self.weight_boundary = weight_boundary

        # Initialize all loss components
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss(alpha=0.2, beta=0.4)
        self.iou_loss = IoULoss()
        # Initializing BoundaryLoss with a scale_factor to prevent it from dominating
        self.boundary_loss = BoundaryLoss(scale_factor=1e-3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Assuming target has shape (N, H, W) and pred is logits (N, C, H, W)
        # Squeeze dim 1 if it exists, as CrossEntropyLoss expects (N, C, ...) and (N, ...)
        target_squeezed = target.squeeze(1) if target.dim() == pred.dim() else target

        loss_ce = self.ce_loss(pred, target_squeezed)
        loss_dice = self.dice_loss(pred, target_squeezed)
        loss_focal = self.focal_loss(pred, target_squeezed)
        loss_tversky = self.tversky_loss(pred, target_squeezed)
        loss_iou = self.iou_loss(pred, target_squeezed)
        loss_boundary = self.boundary_loss(pred, target_squeezed)

        total_loss = (
                self.weight_ce * loss_ce +
                self.weight_dice * loss_dice +
                self.weight_focal * loss_focal +
                self.weight_tversky * loss_tversky +
                self.weight_iou * loss_iou +
                self.weight_boundary * loss_boundary
        )
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions for binary image classification.

    This version correctly handles the binary case by using BCEWithLogitsLoss
    and passing the appropriate targets to other multi-class losses.
    """

    def __init__(self, weights: Optional[dict] = None):
        super().__init__()
        if weights is None:
            weights = {'bce': 0.2, 'tversky': 0.4, 'focal': 0.4, 'dice': 0.6, 'jaccard': 0.6}
        self.weights = weights

        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=0.2, beta=0.4)
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # For binary classification, pred has shape (N, 1, H, W) or (N, H, W)
        # target has shape (N, H, W) with values 0 or 1

        # BCEWithLogitsLoss expects float target
        bce = self.bce(pred, target.unsqueeze(1).float())

        # For the other losses, we treat the binary case as a multi-class
        # problem with 2 classes (0 and 1) to use the existing multi-class
        # implementations.
        tversky = self.tversky(pred, target.long())
        focal = self.focal(pred, target.long())
        dice = self.dice(pred, target.long())
        jaccard = self.iou(pred, target.long())

        total_loss = (self.weights['bce'] * bce +
                      self.weights['tversky'] * tversky +
                      self.weights['focal'] * focal +
                      self.weights['dice'] * dice +
                      self.weights['jaccard'] * jaccard)
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
