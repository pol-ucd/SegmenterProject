from typing import List, Optional

import numpy as np
import torch

from segmenter.masks import (FluidMask, InstrumentMask, RandomShapeMask, FoldMask, BaseMask)


class CompositeMask:
    def __init__(self, mask_ratio: float = 0.7, shapes_per_image: int = 20):
        """
        Orchestrator class for generating batches of composite masks.

        :param mask_ratio: (Deprecated for performance) Target coverage ratio.
                           Used only for tracking/logging, fixed number of shapes is used instead.
        :param shapes_per_image: The fixed number of shapes to draw on *each* image in the batch.
                                 This guarantees deterministic and fast execution time.
        """
        self.mask_ratio = mask_ratio
        self.shapes_per_image = shapes_per_image
        self.masks: List[BaseMask] = [
            InstrumentMask(thickness=20),
            FluidMask(max_radius=100, saturation=0.5),
            RandomShapeMask(max_vertices=10, max_radius=120),
            FoldMask(max_length=180, max_width=20)
        ]
        self.mask_names = [m.__class__.__name__ for m in self.masks]

    def _mask_percentage(self, mask: np.ndarray) -> float:
        """ Calculates mask coverage percentage on a BxHxW or 1xHxW array. """
        if mask.ndim == 4:
            # Handle BxCxHxW from torch.Tensor input (only look at one channel)
            mask = mask[:, 0, ...]

            # Check if mask is a tensor, if so, detach and convert
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # Calculate mean coverage across the entire batch (or single image)
        return (mask > 0).sum() / mask.size

    def generate_pixel_mask(self, x: torch.Tensor, multi_channel: Optional[bool] = False) -> torch.Tensor:
        """
        Generates a batch of masks by adding a fixed number of random shapes
        to each image in a highly vectorized way.
        """
        if x.ndim == 2:
            # Handle HxW input (single image)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # Handle CxHxW input (single image)
            x = x.unsqueeze(0)

        # Minimal Tensor to NumPy conversion (just to get shape)
        b, c, h, w = x.shape

        # Total number of 2D masks to generate
        total_shapes = b * self.shapes_per_image

        # Prepare a list of all 2D masks needed for the entire batch
        all_2d_masks = []
        # percentage_covered = 0.0
        # Generate all 2D masks (fast part)
        for _ in range(total_shapes):
            # Pick a random mask generator
            mask_idx = np.random.randint(low=0, high=len(self.masks))
            mask_gen = self.masks[mask_idx]

            # Generate the single 2D mask (H, W)
            mask_2d = mask_gen._mask2D(h, w)
            # percentage_covered += self._mask_percentage(mask_2d)
            # if percentage_covered > self.mask_ratio:
            #     break
            all_2d_masks.append(mask_2d)

        # Stack all masks into a single 3D array: (Total_Shapes, H, W)
        if not all_2d_masks:
            # Handle case where total_shapes is 0
            composite_mask_np = np.zeros(shape=(b, h, w), dtype=np.uint8)
        else:
            all_masks_np = np.stack(all_2d_masks, axis=0)

            # Reshape into (B, Shapes_Per_Image, H, W)
            all_masks_np = all_masks_np.reshape(b, self.shapes_per_image, h, w)

            # Vectorized composition: take the maximum across the 'Shapes_Per_Image' dimension
            # np.max(axis=1) is fast and efficient for combining masks (OR operation)
            composite_mask_np = np.max(all_masks_np, axis=1).astype(np.uint8)

        # Final conversion back to Torch Tensor
        # Resulting shape: (B, H, W) -> (B, 1, H, W)
        pixel_mask_torch = torch.from_numpy(composite_mask_np).float().unsqueeze(1)

        # Expand channels (optional: if multi-channel mask is needed)
        if multi_channel & c > 1:
            pixel_mask_torch = pixel_mask_torch.repeat(1, c, 1, 1)

        # Ensure mask is binary (0.0 or 1.0)
        return (pixel_mask_torch > 0).float()



if __name__ == '__main__':
    # Using torchvision only for the visualization utility
    try:
        import torchvision.transforms

        trans = torchvision.transforms.ToPILImage()
        show_img = True
    except ImportError:
        print("Note: torchvision not found. Image display is disabled.")
        trans = None
        show_img = False

    b, c, h, w = 16, 3, 512, 512  # Larger batch size to test performance
    shapes_per_image = 10

    x_image = torch.randn(b, c, h, w)

    # Instantiate the composite mask generator
    mask_generator = CompositeMask(shapes_per_image=shapes_per_image)

    # --------------------------------
    # TIME CRITICAL SECTION START
    # --------------------------------
    import time

    start_time = time.time()
    mask = mask_generator.generate_pixel_mask(x=x_image)
    end_time = time.time()
    # --------------------------------
    # TIME CRITICAL SECTION END
    # --------------------------------

    # Verification and timing
    print("-" * 50)
    print(f"Input batch shape: {x_image.shape}")
    print(f"Output mask shape: {mask.shape}")
    print(f"Shapes generated per image: {shapes_per_image}")
    print(f"Total time taken for batch: {end_time - start_time:.4f} seconds")
    print(f"Percentage of batch that was masked: {mask_generator._mask_percentage(mask=mask): .2f}")
    print("-" * 50)

    # Display the first image mask in the batch
    if show_img:
        print("Displaying the mask for the first image in the batch.")
        # Only show the first channel of the mask for visualization
        out = trans(mask[0, 0].cpu())
        out.show()
        # Note: If running in a headless environment, out.show() might raise an error
        # or not display anything. It works fine in standard environments.
        print(f"PIL Image of mask shape: {out.size}")

