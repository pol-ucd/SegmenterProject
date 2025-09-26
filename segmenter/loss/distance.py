from typing import Tuple

import kornia
import torch

from segmenter.loss import EPSILON


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
        # elif backend == "fastgeodis":
        #     try:
        #         self.geodis_fn = FastGeodis.generalised_geodesic2d
        #     except ImportError as e:
        #         raise ImportError("Install FastGeodis for GPU distance transform: pip install FastGeodis") from e
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
