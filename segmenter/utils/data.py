import numbers
import os
import platform
from typing import Tuple, Iterator, Optional, List, Union, Any, Iterable, Dict

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, random_split, ConcatDataset, DataLoader, Sampler, get_worker_info, BatchSampler
import torchvision.transforms.v2 as v2
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
    values in the range [-2.12, 2.7] (approximately).

    To convert back to a 'regular' image you have to reverse
    the ImageNet normalisation step.

    :param batch: 4D stack fo images
    :return: 4D stack with transforms applied by image
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


def finetune_transform(batch: dict) -> dict:
    """
    Finetuning transform intended as preprocessing for
    SegFormer backbone model input. Standardise the pixel values
    to align with the ImageNet mean and std. This means the images
    not in the range [0.0, 1.0] anymore and will have pixel
    values in the range [-2.12, 2.7] (approximately).

    To convert back to a 'regular' image you have to reverse
    the ImageNet normalisation step.

    :param batch: 4D stack fo images
    :return: 4D stack with transforms applied by image
    """
    # Standard ImageNet mean and standard deviation
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    TARGET_SIZE = (512, 512)  # Size for SegFormer

    pipeline_img = v2.Compose([v2.ToImage(),
                               v2.ToDtype(torch.float32, scale=True),
                               v2.Resize(TARGET_SIZE, interpolation=InterpolationMode.BICUBIC),
                               v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                               ])

    pipeline_mask = v2.Compose([v2.ToImage(),
                                v2.Resize(TARGET_SIZE, interpolation=InterpolationMode.NEAREST),
                                v2.ToDtype(torch.long, scale=False),
                                ])

    masks = np.argmax(batch['masks'], axis=1)
    images = batch['images']

    b = images.shape[0]
    images = [pipeline_img(images[idx]) for idx in range(b)]
    masks = [pipeline_mask(masks[idx]) for idx in range(b)]
    images = torch.stack(images)
    masks = torch.stack(masks)

    return {'images': images, 'masks': masks}

class HDF5Exception(Exception):
    pass

class SSLTransformException(Exception):
    pass

class TransformCheckNan(torch.nn.Module):
    def __init__(self, step_name: str = ""):
        """
        Helper class to debug torchvision transform pipelines
        :param step_name: A unique string to identify the pipeline step
        """
        super(TransformCheckNan, self).__init__()
        self.step_name = step_name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            print(f"NaN generated after step: {self.step_name}")
        return x

class TransformClamp(torch.nn.Module):
    def __init__(self):
        """
        Helper class to debug torchvision transform pipelines
        :param step_name: A unique string to identify the pipeline step
        """
        super(TransformCheckNan, self).__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0, max=1)

class SSLTransformPipeline:
    """
    Creates the dual-view augmentation pipeline for the Siamese Network.

    This function applies two independent sets of transformations (one for the
    Anchor view and one for the Target view) to the same image. The pipeline is
    designed based on best practices from self-supervised learning (DINO, MSN),
    prioritizing strong photometric and geometric differences between views.
    """

    def __init__(self,
                 size: Tuple[int, int] = (512, 512),
                 global_crop_scale: Tuple[float, float] = (0.4, 1.0),
                 # Parameters for local crops, following DINO's common settings
                 num_local_crops: int = 8,
                 local_crop_size: Tuple[int, int] = (96, 96),
                 local_crop_scale: Tuple[float, float] = (0.05, 0.4)):
        """
        Initializes the SSL augmentation pipeline.

        Args:
            size: The output resolution (H, W) of the main anchor and target views (Global Crops).
            global_crop_scale: The minimum and maximum scale for the global random crops.
            num_local_crops: The number of small local crops to generate per image.
            local_crop_size: The output resolution (H, W) of the local crops.
            local_crop_scale: The minimum and maximum scale for the local random crops.
        """
        # Store local crop configuration
        self.num_local_crops = num_local_crops

        # Define Common Strong Photometric Augmentations

        # Defined: brightness, contrast, saturation are chosen uniformly from [1-0.4, 1+0.4] = [0.6, 1.4]
        # Defined: hue is chosen uniformly from [-0.1, 0.1]
        color_jitter = v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)

        # Solarization is a key non-linear augmentation often applied to the Target view
        solarize_transform = v2.RandomSolarize(threshold=0.5, p=0.2)

        # Base Image Transform (for visualization or non-augmented input)
        self.Image_Transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=size, interpolation=InterpolationMode.BICUBIC),
        ])

        # --- Anchor View Transform (Global Crop): Strong Augmentation + Gaussian Blur + Random Crop ---
        # The Anchor (Student) network is typically trained with stronger noise/regularization.
        self.Anchor_Transform = v2.Compose([
            v2.ToImage(),
            TransformCheckNan("After V2.ToImage()"),
            v2.ToDtype(torch.float32, scale=True),
            # 1. Geometric: Global RandomResizedCrop
            TransformCheckNan("After V2.ToDtype()"),
            v2.RandomResizedCrop(size=size, scale=global_crop_scale,
                                 interpolation=InterpolationMode.BICUBIC),
            TransformCheckNan("After v2.RandomResizedCrop()"),
            v2.RandomHorizontalFlip(p=0.5),  # Standard geometric augmentation
            TransformCheckNan("After v2.RandomHorizontalFlip()"),
            # 2. Photometric: Strong Color Jitter
            v2.RandomApply([color_jitter], p=0.5),
            TransformClamp(),
            TransformCheckNan("After color_jitter"),
            v2.RandomGrayscale(p=0.2),
            TransformCheckNan("v2.RandomGrayscale"),
            # 3. Regularization: Gaussian Blur (p=0.5 is common for Anchor)
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.5),
            TransformCheckNan("After v2.GaussianBlur()"),
            # 4. Final: Normalization
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            TransformCheckNan("After v2.Normalize()"),
        ])

        # --- Target View Transform (Global Crop): Strong Augmentation + Solarization + Random Crop ---
        self.Target_Transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # 1. Geometric: Independent Global Random Cropping
            v2.RandomResizedCrop(size=size, scale=global_crop_scale,
                                 interpolation=InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            # 2. Photometric: Strong Color Jitter
            v2.RandomApply([color_jitter], p=0.8),
            TransformClamp(),
            v2.RandomGrayscale(p=0.2),
            # 3. Non-linear Augmentation: Solarization (key for non-collapse)
            v2.RandomApply([solarize_transform], p=0.2),
            # 4. Regularization: Reduced Gaussian Blur probability
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.1),
            # 5. Final: Normalization
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # --- Local Anchor View Transform (Local Crop): Small Crop + Augmentation ---
        if num_local_crops > 0:
            self.Local_Transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                # 1. Geometric: Small, Local RandomResizedCrop
                v2.RandomResizedCrop(size=local_crop_size, scale=local_crop_scale,
                                     interpolation=InterpolationMode.BICUBIC),
                v2.RandomHorizontalFlip(p=0.5),
                # 2. Photometric: Strong Color Jitter
                v2.RandomApply([color_jitter], p=0.8),
                TransformClamp(),
                v2.RandomGrayscale(p=0.2),
                # 3. Regularization: Reduced Gaussian Blur probability for local views
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.1),
                # 4. Final: Normalization
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.Local_Transform = None



    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns the augmented Global Anchor, Global Target, and multiple Local Anchor views.
        """
        # Note: The transformations are applied independently to the same image.
        results = {'images': Any, 'anchors': Any, 'targets': Any, 'local_anchors': Any}
        try:
            input_images = x['images']
            try:
                if torch.isnan(input_images).any():
                    print(">>>>> NaN in input data: x['images'] (original image)")
            except TypeError:
                if np.isnan(input_images).any():
                    print(">>>>> NaN in input data: x['images'] (original image)")


            # 1. Global Views (Anchor & Target)
            results['images'] = torch.stack([self.Image_Transform(image) for image in input_images], dim=0)
            results['anchors'] = torch.stack([self.Anchor_Transform(image) for image in input_images], dim=0)
            results['targets'] = torch.stack([self.Target_Transform(image) for image in input_images], dim=0)

            # 2. Local Views (Anchors only - N crops per image)
            local_crops = []
            if self.Local_Transform is not None:
                for image in input_images:
                    # Generate N local crops for the current image
                    for _ in range(self.num_local_crops):
                        local_crops.append(self.Local_Transform(image))

                # Stack all local crops into a single batch tensor: (Batch_Size * N_Crops, C, h, w)
                results['local_anchors'] = torch.stack(local_crops, dim=0)
            else:
                # Provide an empty tensor if no local crops are configured
                results['local_anchors'] = torch.tensor([])

        except KeyError:
            raise SSLTransformException("No images found in input. Include images using the 'images' key.")
        if torch.isnan(results['images']).any():
            print("NaN in input data: results['images']")
        if torch.isnan(results['anchors']).any():
            print("NaN in input data: results['anchors']")
        if torch.isnan(results['targets']).any():
            print("NaN in input data: results['targets']")
        if torch.isnan(results['local_anchors']).any():
            print("NaN in input data: results['local_anchors']")


        return results


#
# class GaussianSmoothing(object):
#     def __init__(self, radius):
#         if isinstance(radius, numbers.Number):
#             self.min_radius = radius
#             self.max_radius = radius
#         elif isinstance(radius, list):
#             if len(radius) != 2:
#                 raise Exception(
#                     "`radius` should be a number or a list of two numbers")
#             if radius[1] < radius[0]:
#                 raise Exception(
#                     "radius[0] should be <= radius[1]")
#             self.min_radius = radius[0]
#             self.max_radius = radius[1]
#         else:
#             raise Exception(
#                 "`radius` should be a number or a list of two numbers")
#
#     def __call__(self, image):
#         radius = np.random.uniform(self.min_radius, self.max_radius)
#         return image.filter(ImageFilter.GaussianBlur(radius))
#

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


class HDF5BatchSubsetSampler(BatchSampler):
    """
    Refactored Sampler that yields contiguous index chunks (batches) from a defined
    subset, supporting multiprocessing workers. Inherits from BatchSampler
    to correctly signal that __iter__ yields lists of indices (batches).
    """

    # Note: We must implement a custom __init__ since BatchSampler's default
    # __init__ requires a dataset. We only need the logic.
    def __init__(self,
                 dataset_size: int,
                 batch_size: int,
                 indices: Optional[List[int]] = None,
                 shuffle: bool = True):

        # We don't call the base BatchSampler __init__ as it requires a dataset size
        # and has its own parameter handling. We only use its __iter__ contract.

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        if indices is None:
            self.subset_indices = list(range(dataset_size))
        else:
            self.subset_indices = sorted(indices)

        self.sample_size = len(self.subset_indices)
        self.start_positions = list(range(0, self.sample_size, batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yields a list of contiguous indices (a batch), partitioned for DataLoader workers.
        """
        worker_info = get_worker_info()
        positions_to_iterate = self.start_positions

        # 1. Apply shuffling
        if self.shuffle:
            # Use NumPy's permutation for shuffling stability across workers/epochs
            rng = np.random.default_rng()
            positions_to_iterate = rng.permutation(positions_to_iterate).tolist()

        # 2. Partition the batches based on worker ID
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            positions_to_iterate = positions_to_iterate[worker_id::num_workers]

        # 3. Iterate over the worker's assigned batches and yield the lists of global indices
        for start_pos in positions_to_iterate:
            # Determine the end *position*
            end_pos = min(start_pos + self.batch_size, self.sample_size)

            # Get the slice of *global indices* from the subset list
            batch_indices = self.subset_indices[start_pos:end_pos]

            # Yields a list of indices (a batch)
            yield batch_indices

    def __len__(self) -> int:
        """Returns the total number of batches."""
        return len(self.start_positions)


class HDF5DatasetOptimized(Dataset):
    """
    HDF5 Dataset designed for batch reading using a custom Sampler.
    It opens the file in the worker process via the worker_init_fn
    (best practice for h5py multiprocessing).

    The 'transform' function MUST support the dictionary {key, value}
    format and return transformed values in the same format.
    """

    def __init__(self, hdf5_path: str,
                 data_keys: Optional[List[str]] = None,
                 transform=None):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.data_keys = data_keys
        self.batch_len = None
        self.dataset_len = None

        # File handle is initialized to None and will be opened by worker_init_fn
        self.f = None
        self.key_dtype = {}

        # Get dataset length once
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                for k in f.keys():
                    self.key_dtype[k] = f[k].dtype
                    if self.dataset_len is None:
                        self.dataset_len = len(f[k])
        except Exception as e:
            raise HDF5Exception(f"Error reading dataset length: {e}")
        if self.data_keys is None:
            self.data_keys = list(self.key_dtype.keys())


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

        if is_batch and self.batch_len is None:
            self.batch_len = len(idx)

        # Efficient Batch Read (by a DataLoader)
        # HDF5 supports batch indices so retrieve everything as batches for simplicity
        # items will have shape (len(idx), H, W, C)
        if not is_batch:
            idx = [idx]

        results = {k: self.f[k][idx] for k in self.data_keys if k in self.f.keys()}  # (B, H, W, C)

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
        angle = v2.RandomRotation.get_params(degrees=[-45, 45])
        image_pil = F.rotate(image_pil, angle, interpolation=Image.BILINEAR)
        mask_tensor = F.rotate(mask_tensor, angle, interpolation=Image.NEAREST)

        # Random Crop and Resize
        i, j, h, w = v2.RandomResizedCrop.get_params(image_pil, scale=(0.08, 1.0), ratio=(0.75, 1.33))
        image_pil = F.resized_crop(image_pil, i, j, h, w, self.image_size, interpolation=Image.BILINEAR)
        mask_tensor = F.resized_crop(mask_tensor, i, j, h, w, self.image_size, interpolation=Image.NEAREST)

        # Color Augmentations (applied only to image)
        if torch.rand(1) < 0.5:
            color_jitter = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
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

        image_augment = v2.Compose([
            v2.Resize(self.image_size,
                      v2.InterpolationMode.BICUBIC),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        image = image_augment(image)

        return image.float()


class MSNFinetuneDatasetHDF5(HDF5DatasetOptimized):
    """ A dataset containing raw medical images (for pre-training)
       and annotated images/masks (for fine-tuning).
    """

    def __init__(self, hdf5_path: str,
                 indices: Union[Iterable[int], None] = None,
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

        image_augment = v2.Compose([
            v2.Resize(self.image_size,
                      v2.InterpolationMode.BICUBIC),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        mask_augment = v2.Compose([
            v2.Resize(self.image_size,
                      v2.InterpolationMode.NEAREST),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        # Apply augmentations
        mask = torch.stack([mask_augment(mask[i]) for i in range(mask.shape[0])]).squeeze(1)

        image = image_augment(image)
        return {'images': image, 'masks': mask.long()}


if __name__ == '__main__':
    test_size = 0.1
    # source = "/Users/polmacaonghusa/Documents/Projects/segmenter/data/Classica.h5"
    source = "/Users/polmacaonghusa/Documents/Projects/segmenter/data/pretrain_images.h5"

    h5_len = get_num_samples_from_hdf5(hdf5_path=source)

    indices = list(range(h5_len))[:int(h5_len * test_size)]

    ds = HDF5DatasetOptimized(hdf5_path=source,
                              transform=SSLTransformPipeline())

    print(len(ds))
    print([ds[0][k].shape for k in ds[0].keys()])
