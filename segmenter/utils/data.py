import numbers
import os
import platform

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, random_split, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F


def sharpening_kernel(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    if kernel is None:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
    sharpened_cv2_kernel = cv2.filter2D(img, -1, kernel)
    return sharpened_cv2_kernel.astype(np.uint8)


def gray_world(img: np.ndarray) -> np.ndarray:
    """Simple Gray World color constancy."""
    # Compute average per channel
    avg_b, avg_g, avg_r = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    # Scale     each channel
    img[:, :, 0] = np.clip(img[:, :, 0] * (avg_gray / avg_b), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (avg_gray / avg_g), 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return img.astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Apply CLAHE on the L-channel of LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    lab = cv2.merge([cl, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image_pipeline(image: np.ndarray):
    img_sharp = sharpening_kernel(image)
    img_gw = gray_world(img_sharp)

    # final = apply_clahe(img_gw, clip_limit=5.0, tile_grid_size=(8, 8))

    return img_gw


def get_num_samples_from_hdf5(hdf5_path):
    """
    Dynamically gets the number of samples from an HDF5 file.
    A helper function.

    Args:
        hdf5_path (str): The path to the HDF5 data file.

    Returns:
        int: The number of samples in the 'images' dataset.
    """
    with h5py.File(hdf5_path, 'r', swmr=True) as hf:
        return len(hf['images'])


class GaussianSmoothing(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.data = None
        self._open_hdf5_file()

    def _open_hdf5_file(self):
        """
        Opens the HDF5 file and assigns the dataset references.
        This is called by each worker process on first access.
        """
        self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        self.data = {k:v[:] for k,v in self.hdf5_file.items()}

    def __len__(self):
        if self.data is None:
            self._open_hdf5_file()
        return len(self.data['images'])

    def __getitem__(self, idx):
        if self.data is None:
            self._open_hdf5_file()
        return {'image': self.data['images'][idx],
                'mask': self.data['masks'][idx],
                'size': self.data['image_sizes'][idx],
                'image_path': self.data['image_paths'][idx],
                'mask_path': self.data['mask_paths'][idx]}


class HDF5ImageDataset(HDF5Dataset):
    """
    A PyTorch Dataset subclass for loading the preprocessed Dresden Surgical Anatomy
    Dataset from a single HDF5 file.

    This class is designed to handle a pre-filtered list of indices, allowing for
    dynamic training/test splits. It handles augmentations for the training set.
    """

    def __init__(self, hdf5_path, indices, is_train_split,
                 image_size=(512, 512), n_augment=0):
        """
        Initializes the dataset.

        Args:
            hdf5_path (str): The path to the HDF5 data file.
            indices (list): A list of integer indices to be included in this split.
            is_train_split (bool): A flag to indicate if this is a training split.
            image_size (tuple): The target size (height, width) for resizing/cropping.
            n_augment (int): The number of augmented versions to create for each
                             training sample.

        """
        super(HDF5ImageDataset, self).__init__(hdf5_path)
        self.split_indices = indices
        self.is_train_split = is_train_split
        self.image_size = image_size
        self.n_augment = n_augment
        self.sigma = 30.0


        # Initialize h5py file and dataset references to None
        self.hdf5_file = None
        self.images = None
        self.masks = None
        self.original_names = None

    # def _open_hdf5_file(self):
    #     """
    #     Opens the HDF5 file and assigns the dataset references.
    #     This is called by each worker process on first access.
    #     """
    #     self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
    #     self.images = self.hdf5_file['images']
    #     self.masks = self.hdf5_file['masks']

    def __len__(self):
        """Returns the number of samples in the current split."""
        if self.is_train_split:
            # Total size is (original samples * (1 original + n_augment augmented))
            return len(self.split_indices) * (self.n_augment + 1)
        else:
            return len(self.split_indices)

    @staticmethod
    def _normalise_image(image_pil):
        # Apply CLAHE to image only
        img_np = np.array(image_pil)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_l = clahe_obj.apply(l)

        l_processed = cv2.merge((cl_l, a, b))
        processed_np = cv2.cvtColor(l_processed, cv2.COLOR_LAB2RGB)
        return Image.fromarray(processed_np)

    def _apply_augmentations(self, image_pil, mask_tensor):
        """Applies a single set of random augmentations to an image and mask pair."""
        # Random Horizontal Flip
        if torch.rand(1) < 0.5:
            image_pil = F.hflip(image_pil)
            mask_tensor = F.hflip(mask_tensor)

        # Random Rotation
        angle = transforms.RandomRotation.get_params(degrees=[-45, 45])
        image_pil = F.rotate(image_pil, angle, interpolation=Image.BILINEAR)
        mask_tensor = F.rotate(mask_tensor, angle, interpolation=Image.NEAREST)

        # Random Crop and Resize
        i, j, h, w = transforms.RandomResizedCrop.get_params(image_pil, scale=(0.08, 1.0), ratio=(0.75, 1.33))
        image_pil = F.resized_crop(image_pil, i, j, h, w, self.image_size, interpolation=Image.BILINEAR)
        mask_tensor = F.resized_crop(mask_tensor, i, j, h, w, self.image_size, interpolation=Image.NEAREST)

        # Color Augmentations (applied only to image)
        if torch.rand(1) < 0.5:
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image_pil = color_jitter(image_pil)

        return image_pil, mask_tensor

    def __getitem__(self, idx):
        """
        Loads and pre-processes a single sample.

        Args:
            idx (int): The index of the sample to load. For a training split, this
                       can be an augmented sample.

        Returns:
            tuple: A tuple containing the pre-processed image and mask tensors.
        """
        # Open the file and get dataset references if they are not already open.
        if self.hdf5_file is None:
            self._open_hdf5_file()

        if self.is_train_split:
            # Map the linear index to the original sample and augmentation step
            original_idx = self.split_indices[idx // (self.n_augment + 1)]
            augment_step = idx % (self.n_augment + 1)
        else:
            original_idx = self.split_indices[idx]
            augment_step = 0  # No augmentation for val/test splits

        # Load image and mask as NumPy arrays
        image_np = self.images[original_idx]
        mask_np = self.masks[original_idx]

        # Convert NumPy arrays to PIL Images for easy transformation
        image_pil = Image.fromarray(image_np).convert("RGB")
        # Convert masks to float32 for geometric transformations
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32))

        # Normalise the image light intensity using CLAHE
        image_pil = self._normalise_image(image_pil)

        if augment_step > 0:
            # Apply augmentations if this is an augmented sample
            image_pil, mask_tensor = self._apply_augmentations(image_pil, mask_tensor)
        else:
            # For original samples or val/test splits, only resize
            image_pil = F.resize(image_pil, self.image_size, interpolation=Image.BILINEAR)
            mask_tensor = F.resize(mask_tensor, self.image_size, interpolation=Image.NEAREST)

        # Convert image to Tensor (C, H, W) and normalize
        image_tensor = F.to_tensor(image_pil)
        # image_tensor = shading_correction_pure_torch(image_tensor, sigma=self.sigma)

        # Final conversion of mask to Long Tensor for loss functions
        mask_tensor = mask_tensor.to(torch.long)

        return image_tensor, mask_tensor



# The rest of the script remains the same, but with the modified usage in the main block.
if __name__ == "__main__":
    NUM_CLASSES = 11
    # --- Create a dummy HDF5 file for demonstration purposes ---
    print("Creating dummy HDF5 files for demonstration...")

    # Create the first dummy HDF5 file
    dummy_hdf5_path_1 = "dummy_data_1.h5"
    if os.path.exists(dummy_hdf5_path_1):
        os.remove(dummy_hdf5_path_1)

    H, W = 1024, 1024
    num_samples_1 = 3

    dummy_images_1 = np.random.randint(0, 255, size=(num_samples_1, H, W, 3), dtype=np.uint8)
    dummy_masks_1 = np.random.randint(0, NUM_CLASSES, size=(num_samples_1, H, W), dtype=np.uint8)
    dummy_masks_one_hot_1 = np.zeros((num_samples_1, NUM_CLASSES, H, W), dtype=np.uint8)
    for i in range(num_samples_1):
        for c in range(NUM_CLASSES):
            dummy_masks_one_hot_1[i, c, :, :] = (dummy_masks_1[i, :, :] == c).astype(np.uint8)
    dummy_splits_1 = ['train'] * num_samples_1

    with h5py.File(dummy_hdf5_path_1, 'w') as hf:
        hf.create_dataset('images', data=dummy_images_1, compression='gzip')
        hf.create_dataset('masks', data=dummy_masks_one_hot_1, compression='gzip')
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('splits', data=np.array(dummy_splits_1, dtype=dt))

    print(f"Dummy HDF5 file 1 created at '{dummy_hdf5_path_1}'.")

    # Create the second dummy HDF5 file
    dummy_hdf5_path_2 = "dummy_data_2.h5"
    if os.path.exists(dummy_hdf5_path_2):
        os.remove(dummy_hdf5_path_2)

    num_samples_2 = 4

    dummy_images_2 = np.random.randint(0, 255, size=(num_samples_2, H, W, 3), dtype=np.uint8)
    dummy_masks_2 = np.random.randint(0, NUM_CLASSES, size=(num_samples_2, H, W), dtype=np.uint8)
    dummy_masks_one_hot_2 = np.zeros((num_samples_2, NUM_CLASSES, H, W), dtype=np.uint8)
    for i in range(num_samples_2):
        for c in range(NUM_CLASSES):
            dummy_masks_one_hot_2[i, c, :, :] = (dummy_masks_2[i, :, :] == c).astype(np.uint8)
    dummy_splits_2 = ['train'] * num_samples_2

    with h5py.File(dummy_hdf5_path_2, 'w') as hf:
        hf.create_dataset('images', data=dummy_images_2, compression='gzip')
        hf.create_dataset('masks', data=dummy_masks_one_hot_2, compression='gzip')
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('splits', data=np.array(dummy_splits_2, dtype=dt))

    print(f"Dummy HDF5 file 2 created at '{dummy_hdf5_path_2}'.\n")

    # --- Demonstrate dynamic data splitting and DataLoader usage ---
    image_target_size = (256, 256)
    BATCH_SIZE = 2

    # Check the operating system to determine num_workers
    if platform.system() == 'Darwin':
        num_workers = 0
    else:
        num_workers = os.cpu_count() or 1

    # 1. Load the full dataset from both HDF5 files and combine them
    print("Loading all data from both HDF5 files...")
    # These datasets are only used to get a combined list of indices.
    full_dataset_1 = HDF5ImageDataset(
        hdf5_path=dummy_hdf5_path_1,
        indices=list(range(num_samples_1)),
        is_train_split=True,
        image_size=image_target_size,
        n_augment=0
    )
    full_dataset_2 = HDF5ImageDataset(
        hdf5_path=dummy_hdf5_path_2,
        indices=list(range(num_samples_2)),
        is_train_split=True,
        image_size=image_target_size,
        n_augment=0
    )

    combined_full_dataset = full_dataset_1 + full_dataset_2
    total_samples = len(combined_full_dataset)
    print(f"Total number of combined samples: {total_samples}")

    # 2. Create the dynamic split (e.g., 90% train, 10% test)
    print("\nCreating a dynamic 90/10 train/test split...")
    train_size = int(0.9 * total_samples)
    test_size = total_samples - train_size

    # Use random_split to create the indices for each split
    train_split, test_split = random_split(combined_full_dataset, [train_size, test_size])

    print(f"Training split size: {len(train_split)}")
    print(f"Test split size: {len(test_split)}")

    # 3. Create the DataLoaders for the new splits
    # Use the Subset objects directly, which correctly handle the index mapping to the underlying datasets.
    n_augmentations = 2

    # We create two new ConcatDataset objects, one for each split.
    # The Subset objects automatically handle the logic of which file to load.

    # Create the training dataset with augmentations
    # Note: We can't apply augmentations directly to the Subset object.
    # Instead, we define a custom class or pass the augmentation logic through.
    # The current design is flawed because the Subset object doesn't carry augmentation info.
    #
    # To fix the immediate bug and maintain the existing structure, we will
    # revert to creating the splits before concatenating.

    all_indices_1 = list(range(num_samples_1))
    all_indices_2 = list(range(num_samples_2))

    # Combine and shuffle all indices from both files
    all_combined_indices = all_indices_1 + all_indices_2
    # Ensure reproducibility with a fixed seed
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(all_combined_indices)).tolist()

    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    # Now we correctly split the indices and create new datasets
    train_datasets = []
    test_datasets = []

    # Split indices for file 1
    train_indices_1 = [i for i in train_indices if i < num_samples_1]
    test_indices_1 = [i for i in test_indices if i < num_samples_1]

    if train_indices_1:
        train_datasets.append(HDF5ImageDataset(
            hdf5_path=dummy_hdf5_path_1,
            indices=train_indices_1,
            is_train_split=True,
            image_size=image_target_size,
            n_augment=n_augmentations
        ))
    if test_indices_1:
        test_datasets.append(HDF5ImageDataset(
            hdf5_path=dummy_hdf5_path_1,
            indices=test_indices_1,
            is_train_split=False,
            image_size=image_target_size,
            n_augment=0
        ))

    # Split indices for file 2
    # We must map the combined index back to the file-specific index.
    train_indices_2 = [i - num_samples_1 for i in train_indices if i >= num_samples_1]
    test_indices_2 = [i - num_samples_1 for i in test_indices if i >= num_samples_1]

    if train_indices_2:
        train_datasets.append(HDF5ImageDataset(
            hdf5_path=dummy_hdf5_path_2,
            indices=train_indices_2,
            is_train_split=True,
            image_size=image_target_size,
            n_augment=n_augmentations
        ))
    if test_indices_2:
        test_datasets.append(HDF5ImageDataset(
            hdf5_path=dummy_hdf5_path_2,
            indices=test_indices_2,
            is_train_split=False,
            image_size=image_target_size,
            n_augment=0
        ))

    final_train_dataset = ConcatDataset(train_datasets)
    final_test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        final_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        final_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"Number of batches in the training DataLoader: {len(train_loader)}")
    print(f"Number of batches in the test DataLoader: {len(test_loader)}")

    # 4. Iterate over a few batches from each DataLoader to show it works
    print("\nIterating through the first batch of the training DataLoader:")
    for i, (images, masks) in enumerate(train_loader):
        print(f"  Batch {i + 1}:")
        print(f"    Image batch shape: {images.shape}")
        print(f"    Mask batch shape: {masks.shape}")
        break

    print("\nIterating through the first batch of the test DataLoader:")
    for i, (images, masks) in enumerate(test_loader):
        print(f"  Batch {i + 1}:")
        print(f"    Image batch shape: {images.shape}")
        print(f"    Mask batch shape: {masks.shape}")
        break

    # Clean up the dummy files
    os.remove(dummy_hdf5_path_1)
    os.remove(dummy_hdf5_path_2)
    print("\nDummy HDF5 files cleaned up.")
