import torch
from torch.nn import functional as F

EPSILON = 1e-06


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
