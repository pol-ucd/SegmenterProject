import json
import logging
import os
from glob import glob
from os import PathLike
from typing import List, Tuple

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def _compute_signed_distance_map(one_hot_mask: np.ndarray) -> np.ndarray:
    """
    Computes a signed distance map for a single one-hot encoded mask.

    This function calculates the distance from each pixel to the nearest boundary.
    It's a CPU-bound operation and should be used with caution inside a training
    loop. For best performance, consider pre-calculating these maps offline
    and saving them with your dataset.

    Args:
        one_hot_mask (np.ndarray): A 2D NumPy array representing a single
                                   class mask (1s for the class, 0s otherwise).

    Returns:
        np.ndarray: A 2D NumPy array with the signed distance map.
    """
    # Calculate positive and negative distance maps
    dist_map_positive = distance_transform_edt(one_hot_mask)
    dist_map_negative = distance_transform_edt(1 - one_hot_mask)

    # Combine them to get the signed distance map
    signed_dist_map = dist_map_positive - dist_map_negative
    return signed_dist_map


def create_hdf5_dataset(image_dirs: List, mask_dirs: List, file_types: List,
                        num_classes: int, hdf5_image_size: Tuple, hdf5_path: PathLike):
    """
    Scans image and mask directories, computes unsigned distance maps,
    and saves all data into a single HDF5 file.

    Args:
        image_dirs (List): Paths to the directory containing input images.
        mask_dirs (List): Paths to the directory containing ground truth masks.
        file_types (List): A list of file types to use.
        num_classes (int): Number of classes in the masks
        hdf5_path (str): The output path for the HDF5 file.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting HDF5 dataset creation ---")

    if hdf5_image_size is None:
        hdf5_image_size = (512, 512)

    all_images, all_masks = [], []

    # Iterate through specified paths to find all image and mask files.
    for img_path, mask_path in zip(image_dirs, mask_dirs):
        if not img_path or not mask_path:
            logger.warning(f"Skipping empty image or mask directory for {img_path} and {mask_path}")
            continue

        for file_type in file_types:
            images_found = sorted(glob(os.path.join(img_path, file_type)))
            masks_found = sorted(glob(os.path.join(mask_path, file_type)))

            all_images.extend(images_found)
            all_masks.extend(masks_found)

        if len(all_images) != len(all_masks):
            logger.error(
                f"The number of images and masks do not match. Found {len(all_images)} images and {len(all_masks)} masks.")
            raise ValueError("The number of images and masks do not match.")

    string_dtype = h5py.string_dtype('utf-8', 512)

    # Create the HDF5 file
    with h5py.File(hdf5_path, 'w') as f:
        # Create datasets to store the images, masks, and distance maps.
        # Set `chunks` for better performance and `compression` to save space.
        images_ds = f.create_dataset('images', (len(all_images),) + hdf5_image_size + (3,),
                                     dtype='uint8', chunks=True, compression="gzip")
        masks_ds = f.create_dataset('masks', (len(all_masks),) + hdf5_image_size,
                                    dtype='uint8', chunks=True, compression="gzip")
        dist_maps_ds = f.create_dataset('dist_maps', (len(all_masks), num_classes) + hdf5_image_size,
                                        dtype='float32', chunks=True, compression="gzip")

        image_paths_ds = f.create_dataset('image_paths', (len(all_images),), dtype=string_dtype, compression="gzip")
        mask_paths_ds = f.create_dataset('mask_paths', (len(all_masks),), dtype=string_dtype, compression="gzip")
        image_sizes_ds = f.create_dataset('image_sizes', (len(all_images), 2), dtype='int32', compression="gzip")

        logger.info(f"Processing {len(all_images)} images-mask pairs")
        for i, (image_path, mask_path) in enumerate(zip(all_images, all_masks)):

            image_size = Image.open(image_path).size  # Returns (width, height)
            mask_size = Image.open(mask_path).size

            assert image_size == mask_size

            # Load and convert to numpy array
            image = Image.open(image_path).convert("RGB").resize(hdf5_image_size,
                                                                 Image.Resampling.BICUBIC)
            image = np.array(image, dtype='uint8')

            mask = Image.open(mask_path).convert("L").resize(hdf5_image_size,
                                                             Image.Resampling.NEAREST)
            mask = np.array(mask, dtype='uint8')

            dist_map = []

            for c in range(num_classes):
                mask_c = (mask == c).astype(np.uint8)
                dist_map += [distance_transform_edt(mask_c)]
            dist_map = np.array(dist_map)

            # Store the data in the HDF5 datasets
            images_ds[i] = image
            masks_ds[i] = mask
            dist_maps_ds[i] = dist_map

            image_paths_ds[i] = image_path
            mask_paths_ds[i] = mask_path
            image_sizes_ds[i] = image_size

    logger.info(f"Saved {len(all_images)} samples to {hdf5_path}")


def main():
    logger = logging.getLogger()

    # --- Load parameters from JSON file ---
    try:
        with open('params.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        logger.error("Error: 'params.json' file not found. Please ensure it is in the same directory.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from 'params.json': {e}")
        return

    num_classes = params['num_classes']
    hdf5_image_size = tuple(params['image_size'])
    hdf5_file = params['datasets']['hdf5']
    image_paths = params["datasets"]["image_paths"]
    mask_paths = params["datasets"]["mask_paths"]
    file_types = params["datasets"]["file_types"]

    logger.info(f"Loaded parameters: {params}")

    try:
        # 2. Create the HDF5 dataset (this is the offline step)

        create_hdf5_dataset(image_dirs=image_paths,
                            mask_dirs=mask_paths,
                            file_types=file_types,
                            num_classes=num_classes,
                            hdf5_image_size=hdf5_image_size,
                            hdf5_path=hdf5_file)
    except TypeError as e:
        logger.error(f"Error during HDF5 dataset creation: {e}")
        raise e


if __name__ == '__main__':
    main()
