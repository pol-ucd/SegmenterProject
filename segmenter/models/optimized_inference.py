import collections
from typing import Any, Optional, Tuple, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_autocast_context(device_type: str, enabled: bool = True):
    if device_type == 'cpu' or not enabled:
        return torch.no_grad()
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


class FastGaussianBlur(nn.Module):
    """Optimized Gaussian blur using separable convolutions for faster inference."""
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._build_kernel()

    def _build_kernel(self):
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(x.device)

        if self.kernel_size == 3:
            weight = kernel.expand(x.shape[1], 1, -1, -1)
        else:
            weight = kernel.unsqueeze(0).unsqueeze(0).expand(x.shape[1], 1, -1, -1)

        return F.conv2d(x, weight, padding=self.kernel_size//2, groups=x.shape[1])


class TemporalSmoother:
    """Exponential moving average for temporal consistency across video frames."""
    def __init__(self, decay: float = 0.3, adaptive: bool = False):
        self.decay = decay
        self.adaptive = adaptive
        self.prev_mask: Optional[torch.Tensor] = None
        self.frame_count = 0

    def update(self, mask: torch.Tensor) -> torch.Tensor:
        self.frame_count += 1

        if self.prev_mask is None:
            self.prev_mask = mask.clone()
            return mask

        # A3: soft confidence weighting — probability IS the confidence for binary seg
        blend_weight = torch.clamp(mask, 0.0, 1.0)
        stable_mask = blend_weight * mask + (1.0 - blend_weight) * self.prev_mask

        # T2: adaptive decay based on frame-to-frame change magnitude
        if self.adaptive:
            change = (mask - self.prev_mask).abs().mean().item()
            effective_decay = max(0.1, min(0.7, self.decay / (1.0 + 3.0 * change)))
        else:
            effective_decay = self.decay

        smoothed = effective_decay * self.prev_mask + (1.0 - effective_decay) * stable_mask

        self.prev_mask = smoothed.clone()
        return smoothed

    def update_with_flow(
        self,
        mask: torch.Tensor,
        prev_frame_rgb: Any,
        curr_frame_rgb: Any,
    ) -> torch.Tensor:
        """T3: Optical-flow-warped temporal update. Falls back to plain update if cv2 unavailable."""
        try:
            import cv2
            import numpy as np

            if self.prev_mask is None:
                self.prev_mask = mask.clone()
                return mask

            prev_gray = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame_rgb, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            h, w = flow.shape[:2]
            map_x = flow[:, :, 0] + np.arange(w).reshape(1, w).astype(np.float32)
            map_y = flow[:, :, 1] + np.arange(h).reshape(h, 1).astype(np.float32)

            prev_np = self.prev_mask.squeeze().cpu().numpy()
            warped_prev = cv2.remap(prev_np, map_x, map_y, cv2.INTER_LINEAR)
            warped_prev_t = torch.from_numpy(warped_prev).unsqueeze(0).unsqueeze(0).to(mask.device)

            self.prev_mask = warped_prev_t
            return self.update(mask)
        except ImportError:
            return self.update(mask)

    def reset(self):
        self.prev_mask = None
        self.frame_count = 0


class AdaptiveThresholding:
    """Adaptive threshold with Otsu's method approximation for robust binarization."""
    def __init__(self, initial_threshold: float = 0.5):
        self.threshold = initial_threshold

    def compute(self, mask: torch.Tensor) -> float:
        hist = torch.histc(mask.flatten(), bins=256, min=0., max=1.)
        hist = hist / hist.sum()

        cumsum = torch.cumsum(hist, dim=0)
        cummean = torch.cumsum(torch.arange(256, device=mask.device) * hist, dim=0)

        total_mean = cummean[-1]

        variance = (total_mean * cumsum - cummean) ** 2 / (cumsum + 1e-10) / (1 - cumsum + 1e-10)

        max_variance, optimal_idx = variance.max(dim=0)
        self.threshold = (optimal_idx.float() / 255).item()

        return self.threshold

    def __call__(self, mask: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        t = threshold if threshold is not None else self.threshold
        return (mask > t).float()


@dataclass
class InferenceConfig:
    image_size: int = 512
    use_amp: bool = True
    use_temporal_smoothing: bool = True
    temporal_alpha: float = 0.3
    adaptive_temporal: bool = False
    use_adaptive_threshold: bool = False
    use_tta: bool = False
    tta_scales: List[float] = field(default_factory=lambda: [1.0])
    blur_kernel: int = 3
    blur_sigma: float = 1.0
    median_filter_size: int = 0
    device: Optional[str] = None
    use_optical_flow: bool = False


class OptimizedSegmenter:
    """
    Optimized segmentation model wrapper with:
    - Fast Gaussian smoothing (replaces expensive MedianPool2d)
    - Temporal smoothing for video consistency
    - Test-time augmentation (TTA)
    - Adaptive thresholding
    - Cross-platform MPS/CUDA/CPU support
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[InferenceConfig] = None,
    ):
        self.model = base_model
        self.config = config or InferenceConfig()

        self.device = next(base_model.parameters()).device

        self._detect_and_validate_device()

        # E1: allocate mean/std once, move with .to(device)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.blur = FastGaussianBlur(
            kernel_size=self.config.blur_kernel,
            sigma=self.config.blur_sigma
        ) if self.config.blur_kernel > 0 else None

        self.temporal_smoother = TemporalSmoother(
            decay=self.config.temporal_alpha,
            adaptive=self.config.adaptive_temporal,
        ) if self.config.use_temporal_smoothing else None

        self.threshold = AdaptiveThresholding() if self.config.use_adaptive_threshold else None

    def _detect_and_validate_device(self):
        """Detect device type and validate configuration."""
        device_type = self.device.type

        if self.config.device is not None:
            device_type = self.config.device

        if device_type == 'mps':
            if not torch.backends.mps.is_available():
                import warnings
                warnings.warn("MPS not available, falling back to CPU")
                self.device = torch.device('cpu')
                device_type = 'cpu'

        elif device_type == 'cuda':
            if not torch.cuda.is_available():
                import warnings
                warnings.warn("CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
                device_type = 'cpu'

        self.config.use_amp = self.config.use_amp and (device_type in ('cuda', 'mps'))

    def _get_autocast_enabled(self) -> bool:
        return self.config.use_amp and self.device.type in ('cuda', 'mps')

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image: optional scale-to-[0,1], resize to model input size, normalise."""
        if image.max() > 1.0:
            image = image / 255.0

        # B2: always resize to training resolution using direct resize (no letterbox)
        target = self.config.image_size
        if image.shape[2] != target or image.shape[3] != target:
            image = F.interpolate(
                image, size=(target, target), mode='bilinear', align_corners=False
            )

        # E1: reuse pre-allocated buffers
        mean = self._mean.to(image.device)
        std = self._std.to(image.device)
        return (image - mean) / std

    def _apply_tta(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Apply test-time augmentation by scaling."""
        results = []
        for scale in self.config.tta_scales:
            if scale == 1.0:
                results.append(image)
            else:
                h, w = image.shape[2:]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
                scaled = F.interpolate(scaled, size=(h, w), mode='bilinear', align_corners=False)
                results.append(scaled)
        return results

    def predict(
        self,
        image: torch.Tensor,
        apply_smoothing: bool = True,
        apply_temporal: bool = True,
        prev_frame_rgb: Optional[Any] = None,
        curr_frame_rgb: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a single image.

        Args:
            image: Input tensor [B, C, H, W] in range [0, 1] or [0, 255]
            apply_smoothing: Apply Gaussian smoothing
            apply_temporal: Apply temporal smoothing (for video)
            prev_frame_rgb: Previous RGB frame numpy array (for optical flow, optional)
            curr_frame_rgb: Current RGB frame numpy array (for optical flow, optional)

        Returns:
            Tuple of (probability_map, binary_mask)
        """
        self.model.eval()

        original_size = image.shape[2:]

        image = image.to(self.device)

        preprocessed = self.preprocess(image)

        use_amp = self._get_autocast_enabled()
        device_type = self.device.type if self.device.type != 'mps' else 'cpu'

        # B4: wrap entire inference (including TTA) in no_grad
        with torch.no_grad():
            if self.config.use_tta and len(self.config.tta_scales) > 1:
                augmented = self._apply_tta(preprocessed)
                outputs = []
                for aug_img in augmented:
                    with torch.amp.autocast(device_type, enabled=use_amp):
                        out = self.model(aug_img)
                        out = torch.softmax(out, dim=1)[:, 1:2, :, :]
                        outputs.append(out)
                prob = torch.stack(outputs).mean(dim=0)
            else:
                with torch.amp.autocast(device_type, enabled=use_amp):
                    output = self.model(preprocessed)
                    prob = torch.softmax(output, dim=1)[:, 1:2, :, :]

        prob = F.interpolate(prob, size=original_size, mode='bilinear', align_corners=False)

        if apply_smoothing and self.blur is not None:
            prob = self.blur(prob)

        if apply_temporal and self.temporal_smoother is not None:
            if (
                self.config.use_optical_flow
                and prev_frame_rgb is not None
                and curr_frame_rgb is not None
            ):
                prob = self.temporal_smoother.update_with_flow(prob, prev_frame_rgb, curr_frame_rgb)
            else:
                prob = self.temporal_smoother.update(prob)

        binary = self._compute_binary(prob)

        return prob, binary

    def _compute_binary(self, prob: torch.Tensor) -> torch.Tensor:
        if self.threshold is not None:
            t = self.threshold.compute(prob)
            return (prob > t).float()
        return (prob > 0.5).float()

    def predict_batch(
        self,
        images: torch.Tensor,
        apply_smoothing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a batch of images."""
        self.model.eval()

        original_size = images.shape[2:]
        preprocessed = self.preprocess(images)

        with torch.no_grad():
            output = self.model(preprocessed)
            probs = torch.softmax(output, dim=1)[:, 1:2, :, :]

        probs = F.interpolate(probs, size=original_size, mode='bilinear', align_corners=False)

        if apply_smoothing and self.blur is not None:
            probs = self.blur(probs)

        binaries = self._compute_binary(probs)

        return probs, binaries

    def reset_temporal(self):
        """Reset temporal smoothing state (call for new video)."""
        if self.temporal_smoother is not None:
            self.temporal_smoother.reset()

    def set_temporal_alpha(self, alpha: float):
        """Update temporal smoothing strength (decay)."""
        if self.temporal_smoother is not None:
            self.temporal_smoother.decay = alpha

    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        if self.blur is not None:
            self.blur = self.blur.to(device)
        return self


class FrameCache:
    """Simple frame cache for keyframe-based video processing."""

    def __init__(self, max_size: int = 30, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        # E3: use deque for O(1) eviction
        self.cache: collections.deque = collections.deque(maxlen=max_size)

    def should_process(self, frame_idx: int, frame: torch.Tensor) -> bool:
        """Determine if frame should be processed or interpolated."""
        if len(self.cache) == 0:
            return True

        last_idx, last_frame = self.cache[-1]

        similarity = self._compute_similarity(last_frame, frame)

        if similarity < self.similarity_threshold:
            return True

        return frame_idx - last_idx > 10

    def _compute_similarity(self, frame1: torch.Tensor, frame2: torch.Tensor) -> float:
        """Compute structural similarity between frames."""
        diff = (frame1 - frame2).abs()
        return 1.0 - diff.mean().item()

    def add(self, frame_idx: int, frame: torch.Tensor):
        """Add frame to cache."""
        self.cache.append((frame_idx, frame.clone()))

    def get(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Get cached frame by index."""
        for idx, frame in self.cache:
            if idx == frame_idx:
                return frame
        return None

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
