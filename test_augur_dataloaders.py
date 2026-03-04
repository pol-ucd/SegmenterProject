import logging
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_dataset_statistics(loader, name, max_batches=50):
    print(f"\n{'='*70}")
    print(f"Computing statistics for {name.upper()} dataset (sampling up to {max_batches} batches)...")
    print(f"{'='*70}")

    image_sum = 0.0
    image_sum_sq = 0.0
    mask_sum = 0.0
    mask_sum_sq = 0.0
    mask_positive = 0
    total_pixels = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        images = batch['images']
        masks = batch['masks']

        image_sum += images.sum().item()
        image_sum_sq += (images ** 2).sum().item()

        mask_sum += masks.sum().item()
        mask_sum_sq += (masks ** 2).sum().item()

        mask_positive += (masks > 0.5).sum().item()
        total_pixels += masks.numel()

        num_batches += 1

    num_samples = num_batches * images.shape[0]
    num_pixels = num_samples * images.shape[2] * images.shape[3]

    image_mean = image_sum / num_pixels
    image_std = np.sqrt(image_sum_sq / num_pixels - image_mean ** 2)

    mask_mean = mask_sum / num_pixels
    mask_std = np.sqrt(mask_sum_sq / num_pixels - mask_mean ** 2)
    foreground_ratio = mask_positive / total_pixels if total_pixels > 0 else 0

    print(f"\n{name.upper()} Dataset Statistics (sampled {num_batches} batches, ~{num_samples} samples):")
    print(f"  Batch shape: {images.shape}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Image mean: {image_mean:.4f}")
    print(f"  Image std: {image_std:.4f}")
    print(f"  Image min: {images.min().item():.4f}")
    print(f"  Image max: {images.max().item():.4f}")
    print(f"  Mask dtype: {masks.dtype}")
    print(f"  Mask mean (foreground ratio): {foreground_ratio:.4f}")
    print(f"  Mask unique values: {torch.unique(masks).tolist()}")

    return {
        'num_samples': num_samples,
        'num_batches': num_batches,
        'image_mean': image_mean,
        'image_std': image_std,
        'image_min': images.min().item(),
        'image_max': images.max().item(),
        'foreground_ratio': foreground_ratio,
    }


def test_augur_dataloaders():
    from segmenter.utils.directory_dataset import (
        SurgicalDataLoader,
        find_image_mask_pairs,
    )

    data_root = 'data/augur/data'

    print(f"\n{'#'*70}")
    print(f"# Testing SurgicalDataLoader with data from: {data_root}")
    print(f"{'#'*70}")

    data_loader = SurgicalDataLoader(
        data_root=data_root,
        image_size=(512, 512),
        batch_size=8,
        num_workers=0,
        normalize=True,
        train_augment=True,
    )

    print("\n--- Verifying image-mask matching ---")
    issues = data_loader.verify_matching()
    has_issues = False
    for split, issue_list in issues.items():
        if issue_list:
            has_issues = True
            print(f"  {split}: {len(issue_list)} issues")
            for issue in issue_list[:5]:
                print(f"    - {issue}")
        else:
            print(f"  {split}: All matched ✓")

    if not has_issues:
        print("\n✓ All image-mask pairs matched successfully!")

    data_loader.print_dataset_info()

    train_stats = compute_dataset_statistics(data_loader.train_loader, 'train')
    val_stats = compute_dataset_statistics(data_loader.val_loader, 'val')
    test_stats = compute_dataset_statistics(data_loader.test_loader, 'test')

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    print(f"  Train: {train_stats['num_samples']} samples, foreground: {train_stats['foreground_ratio']:.2%}")
    print(f"  Val:   {val_stats['num_samples']} samples, foreground: {val_stats['foreground_ratio']:.2%}")
    print(f"  Test:  {test_stats['num_samples']} samples, foreground: {test_stats['foreground_ratio']:.2%}")
    print(f"{'='*70}\n")


def main():
    print("Testing Augur Data Loaders")
    print("=" * 70)

    try:
        test_augur_dataloaders()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure the data directory exists at 'data/augur/data'")
        sys.exit(1)
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
