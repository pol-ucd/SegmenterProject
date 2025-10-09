import numbers
import os
import platform
from typing import Tuple, Iterator

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split, ConcatDataset, DataLoader, Sampler
from torchvision import transforms, transforms as T
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms import v2 as v2
from torchvision.tv_tensors import Image, Mask


def sharpening_kernel(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    if kernel is None:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
    sharpened_cv2_kernel = cv2.filter2D(img, -1, kernel)
    return sharpened_cv2_kernel.astype(np.uint8)

#
# def gray_world(img: np.ndarray) -> np.ndarray:
#     """Simple Gray World color constancy."""
#     # Compute average per channel
#     avg_b, avg_g, avg_r = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
#     avg_gray = (avg_b + avg_g + avg_r) / 3
#     # Scale     each channel
#     img[:, :, 0] = np.clip(img[:, :, 0] * (avg_gray / avg_b), 0, 255)
#     img[:, :, 1] = np.clip(img[:, :, 1] * (avg_gray / avg_g), 0, 255)
#     img[:, :, 2] = np.clip(img[:, :, 2] * (avg_gray / avg_r), 0, 255)
#     return img.astype(np.uint8)
#
#
# def apply_clahe(img: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
#     """Apply CLAHE on the L-channel of LAB."""
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     cl = clahe.apply(l)
#     lab = cv2.merge([cl, a, b])
#     return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#
#
# def preprocess_image_pipeline(image: np.ndarray):
#     img_sharp = sharpening_kernel(image)
#     img_gw = gray_world(img_sharp)
#
#     # final = apply_clahe(img_gw, clip_limit=5.0, tile_grid_size=(8, 8))
#
#     return img_gw
#

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


def pretrain_transform(batch: dict) -> dict:
    """
    Base image pretraining transform intended as preprocessing for
    SegFormer backbone model input. Standardise the pixel values
    to align with the ImageNet mean and std. This means the images
    not in the range [0.0, 1.0] anymore and will have pixel
    values in the range [-3.0, 3.0] (approximately).

    To convert back to a 'regular' image you have to reverse
    the ImageNet normalisation step.

    :param batch:
    :return:
    """
    # Standard ImageNet mean and standard deviation
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    TARGET_SIZE = (512, 512)  # Size for SegFormer

    pipeline = v2.Compose([v2.ToImage(),
                           v2.ToDtype(torch.float32, scale=True),
                           v2.Resize(TARGET_SIZE, interpolation=InterpolationMode.BICUBIC),
                           v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                           ])

    images = [pipeline(img) for img in batch['images']]
    result = {'images': torch.stack(images)}
    return result


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



class HDF5BatchSampler(Sampler[int]):
    """
    A custom Sampler that yields contiguous index chunks (batches) to maximize
    HDF5 read efficiency. This implements "weak shuffling" by shuffling the
    order of the batch chunks but keeping indices within a chunk contiguous.
    """

    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool = True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate the starting index of each batch chunk
        self.start_indices = list(range(0, dataset_size, batch_size))

        # If the last chunk is smaller than batch_size, its starting index is still included.

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yields a list of contiguous indices for the DataLoader to pass to __getitem__.
        """
        indices = self.start_indices
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in indices:
            # Determine the end of the current batch (exclusive)
            end_idx = min(start_idx + self.batch_size, self.dataset_size)

            # Yield the contiguous range of indices
            yield list(range(start_idx, end_idx))

    def __len__(self) -> int:
        """Returns the number of batches."""
        return len(self.start_indices)


class HDF5DatasetOptimized(Dataset):
    """
    HDF5 Dataset designed for batch reading using a custom Sampler.
    It opens the file in the worker process via the worker_init_fn
    (best practice for h5py multiprocessing).

    The 'transform' function MUST support the dictionary {key, value}
    format and return transformed values in the same format.
    """

    def __init__(self, hdf5_path, data_keys=None, transform=None):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.data_keys = data_keys if data_keys is not None else ['images']
        self.data_key = self.data_keys[0]

        # File handle is initialized to None and will be opened by worker_init_fn
        self.f = None

        # Get dataset length once
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                self.dataset_len = len(f[self.data_key])
        except Exception as e:
            print(f"Error reading dataset length: {e}")
            self.dataset_len = 0

    # --- FIX for TypeError: h5py objects cannot be pickled ---
    def __getstate__(self):
        """Called when serializing (pickling) the object for workers."""
        # Create a copy of the instance's dictionary
        state = self.__dict__.copy()
        # Explicitly remove the unpickleable file handle 'f' before serialization
        state['f'] = None
        return state

    def __setstate__(self, state):
        """Called when deserializing (unpickling) the object in a worker."""
        # Restore the state (all attributes except 'f', which is None)
        self.__dict__.update(state)
        # 'f' remains None, and will be correctly opened by worker_init_fn

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        """
        Reads a single item (scalar idx) or a batch (list idx) and applies
        transforms on a per-sample basis if necessary.
        """
        # Ensure file handle is open (safety check)
        if self.f is None:
            try:
                self.f = h5py.File(self.hdf5_path, 'r', swmr=True, libver='latest')
            except Exception as e:
                raise RuntimeError(f"Could not open HDF5 file: {e}")

        is_batch = isinstance(idx, (list, np.ndarray))
        print(f"is_batch: {is_batch}, idx: {idx}")
        # Efficient Batch Read (by a DataLoader)
        # HDF5 supports batch indices so retrieve everything as batches for simplicity
        # items will have shape (len(idx), H, W, C)
        if not is_batch:
            idx = [idx]
        results = {k: self.f[k][idx] for k in self.data_keys if k in self.f} # (B, H, W, C)

        if self.transform:
            results = self.transform(results)

        return results


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, data_keys=None, transform=None):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.data = None
        self.transform = transform
        self.data_keys = data_keys if data_keys is not None else ['images', 'masks', 'image_sizes',
                                                                  'image_paths', 'mask_paths']
        self.data_key = self.data_keys[0]
        # Open the file connection, but DO NOT load the data.
        self.f = None

        # Get the total number of items (length) for __len__.
        # This requires opening the file briefly or assuming it's structured.
        with h5py.File(self.hdf5_path, 'r') as f:
            self.dataset_len = len(f[self.data_key])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # Check if the file handle is open. If not, open it.
        # Need same handle for all multi-process DataLoader workers.
        if self.f is None:
            # Re-open the file handle for this specific worker/process
            self.f = h5py.File(self.hdf5_path, 'r')
        self.data = {k: self.f[k] for k in self.data_keys if k in self.f}

        return {k: v[idx] for k, v in self.data.items()}


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


def hdf5_worker_init_fn(worker_id):
    """
    Initializes the HDF5 file handle for each DataLoader worker process.
    This prevents h5py's multiprocessing issues.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset object

    # Open the HDF5 file handle for this worker
    if dataset.f is None:
        # Use swmr=True (Single Writer Multiple Reader) for read safety
        dataset.f = h5py.File(dataset.hdf5_path, 'r', swmr=True, libver='latest')


class MSNPretrainDatasetHDF5(HDF5DatasetOptimized):
    """ A dataset containing raw medical images (for pre-training)
       and annotated images/masks (for fine-tuning).
    """

    def __init__(self, hdf5_path,
                 size: Tuple[int, int] = (512, 512)):
        super().__init__(hdf5_path)
        self.image_size = size

    def __getitem__(self, idx):
        _data = super().__getitem__(idx)

        image = _data['images']
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        print(image.shape)

        image_augment = T.Compose([T.ToTensor(),
                                   T.Resize(self.image_size,
                                            T.InterpolationMode.BICUBIC),
                                   T.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                                   ])

        image = image_augment(image)

        return image.float()


class MSNFinetuneDatasetHDF5(HDF5DatasetOptimized):
    """ A dataset containing raw medical images (for pre-training)
       and annotated images/masks (for fine-tuning).
    """

    def __init__(self, hdf5_path,
                 indices,
                 size: Tuple[int, int] = (512, 512)):
        super().__init__(hdf5_path)
        self.indices = indices
        self.image_size = size
        self.len = len(indices)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.f is None:
            # Re-open the file handle for this specific worker/process
            self.f = h5py.File(self.hdf5_path, 'r')

        # Load image and convert to tensor
        image = self.f['images'][self.indices[idx]]
        mask = self.f['masks'][self.indices[idx]]

        image_augment = T.Compose([T.ToTensor(),
                                   T.Resize(self.image_size,
                                            T.InterpolationMode.BICUBIC),
                                   T.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                                   ])

        mask_augment = T.Compose([T.ToTensor(),
                                  T.Resize(self.image_size,
                                           T.InterpolationMode.BICUBIC)
                                  ])
        # Apply augmentations
        image = image_augment(image)
        mask = mask_augment(mask)
        return image, mask.long()
#
# def pretrain_transform(batch: dict) -> dict:
#     """
#     Base image pretraining transform intended as preprocessing for
#     SegFormer backbone model input. Standardise the pixel values
#     to align with the ImageNet mean and std. This means the images
#     not in the range [0.0, 1.0] anymore and will have pixel
#     values in the range [-3.0, 3.0] (approximately).
#
#     To convert back to a 'regular' image you have to reverse
#     the ImageNet normalisation step.
#
#     :param batch:
#     :return:
#     """
#     # Standard ImageNet mean and standard deviation
#     IMAGENET_MEAN = [0.485, 0.456, 0.406]
#     IMAGENET_STD = [0.229, 0.224, 0.225]
#     TARGET_SIZE = (512, 512)  # Size for SegFormer
#
#     pipeline = v2.Compose([v2.ToImage(),
#                            v2.ToDtype(torch.float32, scale=True),
#                            v2.Resize(TARGET_SIZE, interpolation=InterpolationMode.BICUBIC),
#                            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#                            ])
#
#     images = [ pipeline(img) for img in batch['images']]
#     result = {'images': torch.stack(images)}
#     return result
#


if __name__ == '__main__':
    import PIL.Image as Image

    source = "/Users/polmacaonghusa/Documents/Projects/segmenter/data/pretrain_images.h5"

    # image_augment = T.Compose([v2.Resize((1024, 1024),
    #                                     T.InterpolationMode.BICUBIC),
    #                            # v2.Normalize(mean=[0.485, 0.456, 0.406],
    #                            #             std=[0.229, 0.224, 0.225])
    #                            ])

    pretrain_dataset = HDF5DatasetOptimized(hdf5_path=source,
                                            data_keys=['images'],
                                            # transform=None)
                                            transform=pretrain_transform)
    indices = list(range(8))
    batch = pretrain_dataset[indices]
    print(f"Batch read of {len(indices)} items, result shape is {batch['images'].shape}, type {type(batch['images'])}")
    print(batch['images'][0].max(), batch['images'][0].min(), batch['images'][0].dtype)
    image = batch['images'][0]
    plt.imshow(image.permute(1,2,0))
    plt.show()

    single = pretrain_dataset[10]
    print(f"Single read of item, result shape is {single['images'].shape}, type {type(single['images'])}")
    image = single['images']
    plt.imshow(image.squeeze(0).permute(1,2,0))
    plt.show()
