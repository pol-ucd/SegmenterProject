import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_dummy_data(base_dir: str, num_samples: int = 10):
    base_path = Path(base_dir)
    images_dir = base_path / 'images'
    masks_dir = base_path / 'masks'

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_samples} dummy image-mask pairs in {base_dir}")

    for i in range(num_samples):
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(images_dir / f"sample_{i:03d}.png")

        mask = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        Image.fromarray(mask).save(masks_dir / f"sample_{i:03d}.png")

    print(f"Created dummy data: {images_dir} and {masks_dir}")
    return str(base_path)


def print_batch_statistics(batch: dict, batch_idx: int):
    images = batch['images']
    masks = batch['masks']

    print(f"\n{'='*60}")
    print(f"Batch {batch_idx} Statistics")
    print(f"{'='*60}")

    print(f"\nImages:")
    print(f"  Shape: {images.shape}")
    print(f"  Dtype: {images.dtype}")
    print(f"  Min: {images.min().item():.4f}")
    print(f"  Max: {images.max().item():.4f}")
    print(f"  Mean: {images.mean().item():.4f}")
    print(f"  Std: {images.std().item():.4f}")

    print(f"\nMasks:")
    print(f"  Shape: {masks.shape}")
    print(f"  Dtype: {masks.dtype}")
    print(f"  Unique values: {torch.unique(masks).tolist()}")
    print(f"  Min: {masks.min().item():.4f}")
    print(f"  Max: {masks.max().item():.4f}")
    print(f"  Mean (foreground ratio): {masks.mean().item():.4f}")

    print(f"{'='*60}\n")


def test_dataloader():
    from segmenter.utils.directory_dataset import (
        DirectoryImageMaskDataset,
        create_dataloader,
        find_image_mask_pairs,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Using temporary directory: {tmpdir}")

        create_dummy_data(tmpdir, num_samples=10)

        print("\n--- Testing find_image_mask_pairs ---")
        pairs = find_image_mask_pairs([tmpdir])
        print(f"Found {len(pairs)} image-mask pairs")
        for img, mask in pairs[:3]:
            print(f"  {img.name} <-> {mask.name}")

        print("\n--- Testing DirectoryImageMaskDataset ---")
        dataset = DirectoryImageMaskDataset(
            directories=[tmpdir],
            image_size=(512, 512),
            is_train=True,
        )
        print(f"Dataset length: {len(dataset)}")

        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Sample image shape: {sample['images'].shape}")
        print(f"Sample mask shape: {sample['masks'].shape}")

        print("\n--- Testing create_dataloader ---")
        dataloader = create_dataloader(
            directories=[tmpdir],
            image_size=(512, 512),
            is_train=True,
            batch_size=4,
            num_workers=0,
            shuffle=True,
        )
        print(f"Dataloader batches: {len(dataloader)}")

        for batch_idx, batch in enumerate(dataloader):
            print_batch_statistics(batch, batch_idx)
            if batch_idx >= 2:
                break

        print("\n--- Testing validation mode (no augmentations) ---")
        val_dataloader = create_dataloader(
            directories=[tmpdir],
            image_size=(512, 512),
            is_train=False,
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )

        for batch_idx, batch in enumerate(val_dataloader):
            print_batch_statistics(batch, batch_idx)
            break

        print("\n--- All tests passed! ---")


def main():
    print("Testing DirectoryImageMaskDataset and Surgical Augmentation Pipeline")
    print("=" * 70)

    try:
        test_dataloader()
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("Please install albumentations: pip install albumentations")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
