import torch

from segmenter.loss import EPSILON, sobel_grad


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
