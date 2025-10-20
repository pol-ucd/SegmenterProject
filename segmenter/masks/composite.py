import numpy as np
import torch
import torchvision

from segmenter.masks import (FluidMask, InstrumentMask, RandomShapeMask, FoldMask)


class CompositeMask:
    def __init__(self,
                 mask_ratio: float = 0.1):
        self.mask_ratio = mask_ratio
        self.masks = [InstrumentMask(num_shapes=1),
                      FluidMask(num_shapes=1),
                      RandomShapeMask(num_shapes=1),
                      FoldMask(num_shapes=1)]

    def _mask_percentage(self, mask: np.ndarray) -> float:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        return (mask > 0).sum() / mask.size

    def generate_pixel_mask(self, x) -> torch.Tensor:
        b, c, h, w = x.shape

        pixel_mask = np.zeros(shape=(b, h, w), dtype=float)

        for idx_image in range(b):
            _mask_ratio = 0.0
            _masks = np.zeros(shape=(h, w), dtype=float)
            while _mask_ratio < self.mask_ratio:
                # Add mask at random
                mask_idx = np.random.randint(low=0, high=len(self.masks))
                _masks = _masks + self.masks[mask_idx].mask2d(x)
                _mask_ratio = self._mask_percentage(mask=_masks)

            pixel_mask[idx_image:, ...] = _masks

        pixel_mask = torch.tensor(pixel_mask, dtype=torch.float).unsqueeze(0).permute(1, 0, 2, 3) # .repeat(c, 1, 1, 1)
        return (pixel_mask > 0).float()


if __name__ == '__main__':
    n_channels = 1
    b, c, h, w = 8, 3, 240, 320
    mask_ratio = 0.7

    x_image = torch.randn(b, c, h, w)
    mask_generator = CompositeMask(mask_ratio=mask_ratio)
    mask = mask_generator.generate_pixel_mask(x=x_image)
    print(f"Percentage of image that was masked: {mask_generator._mask_percentage(mask=mask): .2f}%")
    trans = torchvision.transforms.ToPILImage()
    out = trans(mask[1])
    out.show()
