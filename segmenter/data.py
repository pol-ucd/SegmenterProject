import os
import platform

import h5py
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import conv2d
from torch.utils.data import Dataset, random_split, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F


# # --- Custom Paired Augmentation Classes ---
# # These classes ensure the same random transformation is applied to both the image and the mask.
#
# class PairedRandomHorizontalFlip:
#     """
#     Applies a random horizontal flip to both the image and the mask.
#     """
#
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, img, mask):
#         if random.random() < self.p:
#             img = F.hflip(img)
#             mask = F.hflip(mask)
#         return img, mask
#
#
# class PairedRandomRotation:
#     """
#     Applies a random rotation to both the image and the mask.
#     """
#
#     def __init__(self, degrees, p=0.5):
#         self.degrees = degrees
#         self.p = p
#
#     def __call__(self, img, mask):
#         if random.random() < self.p:
#             angle = transforms.RandomRotation.get_params(self.degrees)
#             # Use interpolation=Image.NEAREST for the mask to prevent new pixel values.
#             img = F.rotate(img, angle, interpolation=Image.BICUBIC)
#             mask = F.rotate(mask, angle, interpolation=Image.NEAREST)
#         return img, mask
#
#
# class PairedColorJitter:
#     """
#     Applies a color jitter transformation to the image only.
#     The mask remains unchanged as it contains segmentation labels.
#     """
#
#     def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
#         self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
#         self.p = p
#
#     def __call__(self, img, mask):
#         if random.random() < self.p:
#             img = self.transform(img)
#         return img, mask
#
#
# class PairedRandomCropAndResize:
#     """
#     Applies a random crop to both the image and the mask, then resizes
#     the cropped regions back to the original size.
#     """
#
#     def __init__(self, size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5):
#         self.size = size
#         self.scale = scale
#         self.ratio = ratio
#         self.p = p
#
#     def __call__(self, img, mask):
#         if random.random() < self.p:
#             i, j, h, w = transforms.RandomResizedCrop.get_params(
#                 img, scale=self.scale, ratio=self.ratio
#             )
#             img = F.crop(img, i, j, h, w)
#             mask = F.crop(mask, i, j, h, w)
#             img = F.resize(img, self.size, interpolation=Image.BICUBIC)
#             mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
#         return img, mask
#
#
# # --- Custom Dataset Class for Semantic Segmentation Augmentation ---
#
# class SemanticSegmentationDatasetAugmentor(Dataset):
#     """
#     A custom PyTorch Dataset subclass for semantic segmentation
#     that loads image-mask pairs and applies N random augmentations
#     for each original pair, effectively boosting the dataset size.
#     """
#
#     def __init__(self, image_paths, mask_paths, n_augments, image_size=(512, 512),
#                  mean: Tuple = (0.485, 0.456, 0.406),
#                  std: Tuple = (0.229, 0.224, 0.225)):
#         """
#         Initializes the dataset.
#
#         Args:
#             image_paths (list): A list of paths to the images.
#             mask_paths (list): A list of paths to the masks.
#             n_augments (int): The number of augmented pairs to generate for each original pair.
#             image_size (tuple): The size to which images and masks will be resized.
#             mean (Tuple): Mean for image normalization.
#             std (Tuple): Standard deviation for image normalization.
#         """
#         assert len(image_paths) == len(mask_paths)
#         self.image_files = image_paths
#         self.mask_files = mask_paths
#         self.N = n_augments
#         self.img_mean = mean
#         self.img_std = std
#
#         assert len(self.image_files) == len(self.mask_files), "Image and mask lists must have the same number of files."
#
#         # Define base transformations that are always applied
#         self.base_transforms = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#
#         # Define the series of random augmentations
#         self.augmentations = [
#             PairedRandomRotation(degrees=(-45, 45), p=0.5),
#             PairedRandomHorizontalFlip(p=0.5),
#             PairedColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
#             PairedRandomCropAndResize(size=image_size, p=0.5)
#         ]
#
#         # Define a single resize transformation
#         self.resize_transform = transforms.Compose([
#             transforms.Resize(image_size, interpolation=Image.BICUBIC),
#         ])
#
#         self.transform_norm = transforms.Compose([
#             transforms.Normalize(self.img_mean, self.img_std)
#         ])
#
#     def __len__(self):
#         """
#         Returns the total number of samples in the dataset, including augmented ones.
#         Each original pair contributes (N + 1) samples (N augmented + 1 original).
#         """
#         return len(self.image_files) * (self.N + 1)
#
#     def __getitem__(self, idx):
#         """
#         Retrieves a single image-mask pair from the dataset.
#         The index determines if an original or an augmented pair is returned.
#         """
#         original_idx = idx // (self.N + 1)
#         version_idx = idx % (self.N + 1)
#
#         image_path = self.image_files[original_idx]
#         mask_path = self.mask_files[original_idx]
#
#         image = Image.open(image_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")
#
#         image = self.resize_transform(image)
#         mask = self.resize_transform(mask)
#
#         if version_idx > 0:
#             for transform in self.augmentations:
#                 image, mask = transform(image, mask)
#
#         image = self.base_transforms(image)
#         mask = self.base_transforms(mask)
#
#         # Ensure mask contains only 0 and 1 values. Masks with >127 are considered 1.
#         if mask.max() > 1:
#             mask = (mask > 127).long()
#
#         image = self.transform_norm(image)
#         return image, mask.long()
#
# #
# class SemanticSegmentationDatasetBasic(Dataset):
#     def __init__(self, image_paths, mask_paths, image_size=(512, 512),
#                  mean: Tuple = (0.485, 0.456, 0.406),
#                  std: Tuple = (0.229, 0.224, 0.225)):
#         self.image_files = image_paths
#         self.mask_files = mask_paths
#         self.size = image_size
#         self.img_mean = mean
#         self.img_std = std
#
#         assert len(image_paths) == len(mask_paths)
#
#         self.image_transform = transforms.Compose([
#             transforms.Resize(self.size, interpolation=Image.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
#         ])
#
#         self.mask_transform = transforms.Compose([
#             transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
#             transforms.ToTensor(),
#         ])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         image_path = self.image_files[idx]
#         mask_path = self.mask_files[idx]
#
#         image = Image.open(image_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")
#
#         image = self.image_transform(image)
#         mask = self.mask_transform(mask)
#
#         # Ensure mask contains only 0 and 1 values.
#         if mask.max() > 1:
#             mask = (mask > 127).long()
#
#         return image, mask.long()
#

# def split_images_and_masks(image_paths: list = None,
#                            mask_paths: list = None,
#                            file_types: list = None,
#                            split: float = None) -> tuple[Any, Any, Any, Any]:
#     """
#     Splits image and mask file paths into training and testing sets.
#
#     Args:
#         image_paths (list): A list of directories containing images.
#         mask_paths (list): A list of directories containing masks.
#         file_types (list): A list of file extensions to search for (e.g., "*.jpg").
#         split (float): The proportion of the dataset to include in the test split.
#
#     Returns:
#         tuple[Any, Any, Any, Any]: A tuple containing four lists:
#             (train_images, train_masks, test_images, test_masks)
#     """
#     split_size = split if split is not None else 0.1
#     image_paths = image_paths
#     mask_paths = mask_paths
#     file_types = file_types
#
#     logger = logging.getLogger(__name__)
#     if file_types is None or image_paths is None or mask_paths is None:
#         logger.error(f"Error loading data: {file_types}, {image_paths} and {mask_paths} cannot be None")
#         raise ValueError(f"Either image_paths or mask_paths must be provided.")
#
#     all_images, all_masks = [], []
#     train_idx, test_idx = [], []
#
#     # Iterate through specified paths to find all image and mask files.
#     for img_path, mask_path in zip(image_paths, mask_paths):
#         for file_type in file_types:
#             images_found = sorted(glob(os.path.join(img_path, file_type)))
#             masks_found = sorted(glob(os.path.join(mask_path, file_type)))
#
#             start_idx = len(all_images)
#             all_images.extend(images_found)
#             all_masks.extend(masks_found)
#
#             if len(images_found) > 0:
#                 indices = np.arange(len(images_found)) + start_idx
#                 _, _, _train, _test = train_test_split(indices[:, np.newaxis], indices,
#                                                        test_size=split_size,
#                                                        shuffle=False)
#                 train_idx.extend(_train)
#                 test_idx.extend(_test)
#
#     # Sanity check: ensure image and mask basenames match.
#     for idx, image in enumerate(all_images):
#         base_name = os.path.splitext(os.path.basename(image))[0]
#         # Remove any suffix like '_mask' for proper comparison
#         base_name = base_name.split("_")[0]
#         try:
#             assert os.path.splitext(os.path.basename(all_masks[idx]))[0].startswith(base_name)
#         except AssertionError:
#             logger.error(
#                 f"Assertion error: image with {base_name} does not match with mask: {os.path.basename(all_masks[idx])}")
#
#     all_images, all_masks = np.array(all_images), np.array(all_masks)
#     logger.info(f"Found {len(all_images)} total image-mask pairs.")
#
#     return (all_images[train_idx], all_masks[train_idx],
#             all_images[test_idx], all_masks[test_idx])
#
#
# class HDF5Dataset(Dataset):
#     """
#     A custom PyTorch Dataset class to efficiently load data from an HDF5 file.
#
#     This class enables fast training by reading pre-computed data, avoiding
#     expensive on-the-fly calculations and CPU-GPU data transfers.
#     """
#
#     def __init__(self, hdf5_path, transform=None):
#         """
#         Initializes the dataset by opening the HDF5 file and creating a
#         lookup map for image paths to indices.
#
#         Args:
#             hdf5_path (str): Path to the HDF5 data file.
#             transform (callable, optional): Optional transform to be applied
#                                             on a sample.
#         """
#         super().__init__()
#         self.hdf5_path = hdf5_path
#         self.transform = transform
#
#         # Open the file once during initialization to get paths and length
#         with h5py.File(self.hdf5_path, 'r') as f:
#             # Load all image paths into a list for quick access
#             self.image_paths_list = [path.decode('utf-8') for path in f['image_paths']]
#             self.dataset_length = len(self.image_paths_list)
#
#         # Create a dictionary for efficient lookup from path to index
#         self.path_to_index = {path: i for i, path in enumerate(self.image_paths_list)}
#
#     def __len__(self):
#         """Returns the total number of samples in the dataset."""
#         return self.dataset_length
#
#     def __getitem__(self, idx):
#         """
#         Retrieves one sample from the dataset.
#
#         Args:
#             idx (int): The index of the sample to retrieve.
#
#         Returns:
#             tuple: A tuple containing (image, mask, distance_map).
#         """
#         # Open the file inside __getitem__ to handle multiple workers
#         with h5py.File(self.hdf5_path, 'r') as f:
#             image = f['images'][idx]
#             mask = f['masks'][idx]
#             dist_map = f['dist_maps'][idx]
#
#             # --- Retrieve metadata ---
#             image_path = self.image_paths_list[idx]  # Use the pre-loaded list
#             mask_path = f['mask_paths'][idx].decode('utf-8')
#             image_name = f['image_names'][idx].decode('utf-8')
#             image_size = f['image_sizes'][idx]
#
#         # Convert numpy arrays to PyTorch tensors and permute dimensions
#         # to match PyTorch's (C, H, W) format for images
#         image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
#         mask = torch.from_numpy(mask).long()
#         dist_map = torch.from_numpy(dist_map).float()
#
#         # You might need to add a channel dimension to the mask and dist_map
#         # depending on your model's input expectations.
#         # mask = mask.unsqueeze(0)
#         # dist_map = dist_map.unsqueeze(0)
#
#         # --- Apply transform if provided ---
#         if self.transform:
#             image, mask, dist_map = self.transform(image, mask, dist_map)
#
#         # --- Return the metadata with the data tensors ---
#         return image, mask, dist_map, image_path, mask_path, image_name, image_size
#
#     def get_indices_from_paths(self, paths_list):
#         """
#         Finds the indices of a list of image paths within the dataset.
#
#         Args:
#             paths_list (list): A list of image file paths.
#
#         Returns:
#             list: A list of corresponding indices. If a path is not found,
#                   its index will be None.
#         """
#         indices = [self.path_to_index.get(path, None) for path in paths_list]
#         return indices
#
#     def get_all_image_paths(self):
#         """
#         Returns a list of all image paths in the dataset.
#
#         Returns:
#             list: A list of all image paths.
#         """
#         return self.image_paths_list
#
#     def create_subset_from_indices(self, indices, transform=None):
#         """
#         Creates a new Subset dataset from a given list of indices.
#
#         This is an efficient way to create a smaller dataset without
#         copying the HDF5 data.
#
#         Args:
#             indices (list or tuple): A list or tuple of integer indices.
#             transform (callable, optional): Optional transform to be applied
#                                             to the subset.
#
#         Returns:
#             torch.utils.data.Subset: A new dataset containing only the
#                                      samples at the specified indices.
#         """
#         return TransformedSubset(self, indices, transform)
#
#
# # Helper class to apply transforms to a Subset ---
# class TransformedSubset(Dataset):
#     """
#     A wrapper class to apply transforms to a PyTorch Subset dataset.
#     This is necessary because the built-in Subset class does not
#     natively support transforms.
#     """
#
#     def __init__(self, dataset, indices, transform=None):
#         self.dataset = Subset(dataset, indices)
#         self.transform = transform
#
#     def __getitem__(self, idx):
#         # Get the item from the underlying subset
#         image, mask, dist_map, image_path, mask_path, image_name, image_size = self.dataset[idx]
#
#         # Apply the transform if it exists
#         if self.transform:
#             image = self.transform(image)
#
#         return image, mask, dist_map, image_path, mask_path, image_name, image_size
#
#     def __len__(self):
#         return len(self.dataset)
#

#
# class CheckpointHandler:
#     """
#     A class to handle loading and saving pandas DataFrames from CSV files.
#
#     This class provides methods to load a CSV file, save the DataFrame to a
#     CSV file, and includes error handling using a custom exception class.
#     """
#
#     def __init__(self, save_path: str = None, load_path: str = None,
#                  suffix: str = None, prefix: str = None, stopper_patience: int = 7):
#         """
#         Initializes the CheckpointHandler with an option
#         al file path.
#
#         Args:
#             save_path (str, optional): The path to the checkpoint file to be loaded.
#                                        If provided, it will attempt to load the data.
#             load_path (str, optional): The path to the checkpoint file to be loaded.
#             suffix (str, optional): The optional datetime suffix appended to the filename
#                                       If provided it is appended at save time and used as
#                                       a suffix to search for a checkpoint at load time.
#                                       Defaults to None in which case a system generated
#                                       datetime suffix is used for load and save.
#         """
#
#         self.prefix = prefix if prefix is not None else "model_checkpoint"
#         self.suffix = suffix
#         if suffix is None:
#             # Get the current date and time
#             now = datetime.now()
#             # YYYY-MM-DD_HH-MM-SS
#             self.suffix = now.strftime("%Y-%m-%d_%H-%M-%S")
#
#         self.df = pd.DataFrame()  # Initialize an empty DataFrame
#         self.pt = None
#         self.train_images, self.train_masks, self.test_images, self.test_masks = None, None, None, None
#
#         self.save_file_path = save_path
#         self.load_file_path = load_path
#         file_name = self.prefix + "_" + self.suffix
#         if load_path is not None:
#             self.load_pt_name = os.path.join(self.load_file_path, file_name + ".pt")
#             self.load_csv_name = os.path.join(self.load_file_path, file_name + ".csv")
#         else:
#             self.load_pt_name = None
#             self.load_csv_name = None
#         if save_path is not None:
#             self.save_pt_name = os.path.join(self.save_file_path, file_name + ".pt")
#             self.save_csv_name = os.path.join(self.save_file_path, file_name + ".csv")
#         else:
#             self.save_pt_name = None
#             self.save_csv_name = None
#
#         self.criterion = EarlyStopping(patience=stopper_patience, min_delta=0.0001,
#                                        mode='min', verbose=True,
#                                        save_path=self.save_pt_name)
#
#         if load_path:
#             self.load()
#
#     def load(self):
#         """
#         Loads a CSV file into the class's DataFrame attribute.
#
#         Args:
#             file_path (str, optional): The path to the CSV file. If not provided,
#                                        it uses the file_path from initialization.
#         """
#
#         try:
#             self.df = pd.read_csv(self.load_csv_name)
#             self.pt = torch.load(self.load_pt_name)
#         except pd.errors.EmptyDataError:
#             self.df = pd.DataFrame()  # Reset DataFrame
#             raise CheckpointError(f"The file {self.load_file_path} is empty.")
#         except pd.errors.ParserError as e:
#             self.df = pd.DataFrame()
#             raise CheckpointError(
#                 f"Unable to parse the file {self.load_file_path}. Check its format. Original error: {e}")
#         except Exception as e:
#             self.df = pd.DataFrame()
#             raise CheckpointError(f"An unexpected error occurred during loading: {e}")
#
#         return self.df, self.pt
#
#     def save(self, df: pd.DataFrame,
#              obj: Any, index: bool = False):
#         """
#         Saves the current DataFrame to a CSV file.
#
#         Args:
#             df (pd.DataFrame): The DataFrame to be saved.
#             obj (Any): The object to be saved (usually model weights)
#             index (bool): Whether to write the DataFrame index to the CSV.
#                           Defaults to False.
#         """
#         # Ensure the directory exists before saving
#         directory = os.path.dirname(self.save_file_path)
#         if directory and not os.path.exists(directory):
#             try:
#                 os.makedirs(directory)
#             except OSError as e:
#                 raise CheckpointError(f"Error creating directory {directory}: {e}")
#
#         try:
#             df.to_csv(self.save_csv_name, index=index)
#             torch.save(obj, self.save_pt_name)
#         except Exception as e:
#             raise CheckpointError(f"An error occurred while saving the Checkpoint: {e}")

def get_gaussian_kernel2d(kernel_size: int, sigma: float, device):
    """Generate a 2D Gaussian kernel."""
    # Create 1D Gaussian
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    g = torch.exp(- (coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    # Outer product to get 2D kernel
    kernel2d = g[:, None] @ g[None, :]
    return kernel2d

def shading_correction_pure_torch(
    img: torch.Tensor,
    sigma: float = 50.0,
    eps: float = 1e-6) -> torch.Tensor:
    """
    Flat-field shading correction using pure PyTorch conv2d.

    Args:
        img: Tensor (C, H, W) or (B, C, H, W), float32 in [0, 255].
        sigma: blur σ in pixels.
        eps: small offset.

    Returns:
        Corrected tensor, same shape/dtype as input.
    """
    # 1. Ensure 4D batch
    batched = img.unsqueeze(0) if img.ndim == 3 else img
    B, C, H, W = batched.shape

    # 2. Create Gaussian kernel
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    kernel2d = get_gaussian_kernel2d(k, sigma, device=batched.device)
    # reshape for depthwise conv: (C, 1, k, k)
    kernel4d = kernel2d.expand(C, 1, k, k)

    # 3. Add eps
    float_img = batched + eps

    # 4. Estimate illumination via depthwise convolution
    illum = conv2d(float_img, weight=kernel4d, groups=C, padding=k//2)

    # 5. Mean over spatial dims
    mean_illum = illum.mean(dim=(-2, -1), keepdim=True)

    # 6. Correct shading
    corrected = float_img / illum * mean_illum

    # 7. Clamp and restore dims
    corrected = torch.clamp(corrected, 0.0, 255.0)
    return corrected.squeeze(0) if img.dim() == 3 else corrected


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

class HDF5ImageDataset(Dataset):
    """
    A PyTorch Dataset subclass for loading the preprocessed Dresden Surgical Anatomy
    Dataset from a single HDF5 file.

    This class is designed to handle a pre-filtered list of indices, allowing for
    dynamic training/test splits. It handles augmentations for the training set.
    """

    def __init__(self, hdf5_path, indices, is_train_split,
                 image_size=(512, 512), n_augment=0,
                 light_control: torch.nn.Module = None):
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
        self.hdf5_path = hdf5_path
        self.split_indices = indices
        self.is_train_split = is_train_split
        self.image_size = image_size
        self.n_augment = n_augment
        self.light_control = light_control
        self.sigma = 30.0

        # Initialize h5py file and dataset references to None
        self.hdf5_file = None
        self.images = None
        self.masks = None

    def _open_hdf5_file(self):
        """
        Opens the HDF5 file and assigns the dataset references.
        This is called by each worker process on first access.
        """
        self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        self.images = self.hdf5_file['images']
        self.masks = self.hdf5_file['masks']
        # self.original_shapes = self.hdf5_file['original_shapes']

    def __len__(self):
        """Returns the number of samples in the current split."""
        if self.is_train_split:
            # Total size is (original samples * (1 original + n_augment augmented))
            return len(self.split_indices) * (self.n_augment + 1)
        else:
            return len(self.split_indices)

    def _apply_augmentations(self, image_pil, mask_tensor):
        """Applies a single set of random augmentations to an image and mask pair."""
        # if self.is_train_split and self.light_control is not None:
        image_pil = self.light_control(image_pil)

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
        # original_shape = tuple(self.original_shapes[original_idx])

        # Convert NumPy arrays to PIL Images for easy transformation
        image_pil = Image.fromarray(image_np).convert("RGB")
        # Convert masks to float32 for geometric transformations
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32))

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

        return image_tensor, mask_tensor  #, original_shape


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
