import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def find_image_mask_pairs(
    directories: List[str],
    supported_extensions: set = None,
) -> List[Tuple[Path, Path]]:
    if supported_extensions is None:
        supported_extensions = SUPPORTED_EXTENSIONS

    pairs = []
    seen_images = set()

    for directory in directories:
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        for entry in root.rglob('*'):
            if not entry.is_dir():
                continue

            entry_name = entry.name.lower()
            if not (entry_name.endswith('images') or entry_name.endswith('masks')):
                continue

            parent = entry.parent
            parent_name = parent.name.lower()

            if parent_name.endswith('images') or parent_name.endswith('masks'):
                continue

            sibling_name = 'masks' if entry_name.endswith('images') else 'images'
            sibling = parent / sibling_name

            if not sibling.exists():
                continue

            images_dir = entry if entry_name.endswith('images') else sibling
            masks_dir = sibling if entry_name.endswith('images') else entry

            if not images_dir.exists() or not masks_dir.exists():
                continue

            image_files = {
                f.stem.lower(): f
                for f in images_dir.iterdir()
                if f.suffix.lower() in supported_extensions
            }
            mask_files = {
                f.stem.lower(): f
                for f in masks_dir.iterdir()
                if f.suffix.lower() in supported_extensions
            }

            for img_name, img_path in image_files.items():
                if img_name in mask_files:
                    pair_key = (str(img_path), str(mask_files[img_name]))
                    if pair_key not in seen_images:
                        seen_images.add(pair_key)
                        pairs.append((img_path, mask_files[img_name]))

    return pairs


class SurgicalAugmentationPipeline:
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        is_train: bool = True,
        enable_endoscopy_augmentations: bool = True,
        augmentation_config: dict = None,
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.enable_endoscopy_augmentations = enable_endoscopy_augmentations
        self._cfg = augmentation_config or {}

        if is_train:
            self.transform = self._build_train_transform()
        else:
            self.transform = self._build_val_transform()

    def _g(self, section: str, key: str, default):
        """Retrieve a value from a config sub-section, falling back to default."""
        return self._cfg.get(section, {}).get(key, default)

    def _build_train_transform(self) -> A.Compose:
        g = self._g
        tgs = g('texture', 'clahe_tile_grid_size', [8, 8])
        return A.Compose([
            A.HorizontalFlip(p=g('geometric', 'h_flip_p', 0.5)),
            A.VerticalFlip(p=g('geometric', 'v_flip_p', 0.3)),
            A.RandomRotate90(p=g('geometric', 'rot90_p', 0.5)),
            A.Affine(
                translate_percent={
                    "x": (-g('geometric', 'shift_limit', 0.1), g('geometric', 'shift_limit', 0.1)),
                    "y": (-g('geometric', 'shift_limit', 0.1), g('geometric', 'shift_limit', 0.1)),
                },
                scale=(
                    1 - g('geometric', 'scale_limit', 0.2),
                    1 + g('geometric', 'scale_limit', 0.2),
                ),
                rotate=(-g('geometric', 'rotate_limit', 45), g('geometric', 'rotate_limit', 45)),
                border_mode=cv2.BORDER_REFLECT,
                p=g('geometric', 'shift_scale_rotate_p', 0.5),
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=g('color', 'brightness_limit', 0.3),
                    contrast_limit=g('color', 'contrast_limit', 0.3),
                    p=g('color', 'brightness_contrast_p', 0.5),
                ),
                A.RandomGamma(
                    gamma_limit=(g('color', 'gamma_min', 50), g('color', 'gamma_max', 150)),
                    p=g('color', 'gamma_p', 0.5),
                ),
                A.HueSaturationValue(
                    hue_shift_limit=g('color', 'hue_shift_limit', 20),
                    sat_shift_limit=g('color', 'sat_shift_limit', 30),
                    val_shift_limit=g('color', 'val_shift_limit', 20),
                    p=g('color', 'hsv_p', 0.5),
                ),
            ], p=g('color', 'group_p', 0.5)),
            A.OneOf([
                A.CLAHE(
                    clip_limit=g('texture', 'clahe_clip_limit', 4.0),
                    tile_grid_size=(tgs[0], tgs[1]),
                    p=g('texture', 'clahe_p', 0.5),
                ),
                A.ColorJitter(
                    brightness=g('texture', 'color_jitter_brightness', 0.2),
                    contrast=g('texture', 'color_jitter_contrast', 0.2),
                    saturation=g('texture', 'color_jitter_saturation', 0.2),
                    hue=g('texture', 'color_jitter_hue', 0.1),
                    p=g('texture', 'color_jitter_p', 0.5),
                ),
            ], p=g('texture', 'group_p', 0.3)),
            A.OneOf([
                A.GaussNoise(
                    std_range=(g('noise', 'gauss_std_min', 0.01), g('noise', 'gauss_std_max', 0.05)),
                    p=g('noise', 'gauss_p', 0.5),
                ),
                A.ISONoise(
                    color_shift=(g('noise', 'iso_color_shift_min', 0.01), g('noise', 'iso_color_shift_max', 0.05)),
                    intensity=(g('noise', 'iso_intensity_min', 0.1), g('noise', 'iso_intensity_max', 0.5)),
                    p=g('noise', 'iso_p', 0.5),
                ),
            ], p=g('noise', 'group_p', 0.2)),
            A.OneOf([
                A.RGBShift(
                    r_shift_limit=g('channel', 'rgb_shift_limit', 20),
                    g_shift_limit=g('channel', 'rgb_shift_limit', 20),
                    b_shift_limit=g('channel', 'rgb_shift_limit', 20),
                    p=g('channel', 'rgb_p', 0.5),
                ),
                A.ChannelShuffle(p=g('channel', 'channel_shuffle_p', 0.5)),
            ], p=g('channel', 'group_p', 0.2)),
            A.OneOf([
                A.MotionBlur(blur_limit=g('blur', 'blur_limit', 7), p=g('blur', 'motion_p', 0.5)),
                A.MedianBlur(blur_limit=g('blur', 'blur_limit', 7), p=g('blur', 'median_p', 0.5)),
                A.GaussianBlur(blur_limit=g('blur', 'blur_limit', 7), p=g('blur', 'gaussian_p', 0.5)),
                A.Blur(blur_limit=g('blur', 'blur_limit', 7), p=g('blur', 'blur_p', 0.5)),
            ], p=g('blur', 'group_p', 0.3)),
            A.OneOf([
                A.CoarseDropout(
                    num_holes_range=(g('dropout', 'coarse_num_holes_min', 4), g('dropout', 'coarse_num_holes_max', 8)),
                    hole_height_range=(g('dropout', 'coarse_hole_height_min', 30), g('dropout', 'coarse_hole_height_max', 50)),
                    hole_width_range=(g('dropout', 'coarse_hole_width_min', 30), g('dropout', 'coarse_hole_width_max', 50)),
                    fill=0,
                    p=g('dropout', 'coarse_p', 0.5),
                ),
                A.GridDropout(
                    ratio=g('dropout', 'grid_ratio', 0.3),
                    holes_number_xy=(g('dropout', 'grid_holes_x', 4), g('dropout', 'grid_holes_y', 6)),
                    fill=0,
                    p=g('dropout', 'grid_p', 0.5),
                ),
            ], p=g('dropout', 'group_p', 0.2)),
            A.Resize(self.image_size[0], self.image_size[1]),
        ], p=1.0)

    def _build_val_transform(self) -> A.Compose:
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
        ], p=1.0)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']


class EndoscopyAugmentor:
    def __init__(
        self,
        blood_artifact_prob: float = 0.2,
        blood_artifact_alpha_range: Tuple[float, float] = (0.1, 0.4),
        tool_occlusion_prob: float = 0.15,
        nbi_simulation_prob: float = 0.15,
        blood_pooling_prob: float = 0.25,
        blood_pooling_alpha_range: Tuple[float, float] = (0.15, 0.45),
        specular_highlight_prob: float = 0.3,
        specular_num_spots: Tuple[int, int] = (3, 12),
        dark_field_prob: float = 0.25,
        red_shift_prob: float = 0.3,
        close_zoom_prob: float = 0.3,
        close_zoom_scale_range: Tuple[float, float] = (0.4, 0.8),
    ):
        self.blood_artifact_prob = blood_artifact_prob
        self.blood_artifact_alpha_range = blood_artifact_alpha_range
        self.tool_occlusion_prob = tool_occlusion_prob
        self.nbi_simulation_prob = nbi_simulation_prob
        self.blood_pooling_prob = blood_pooling_prob
        self.blood_pooling_alpha_range = blood_pooling_alpha_range
        self.specular_highlight_prob = specular_highlight_prob
        self.specular_num_spots = specular_num_spots
        self.dark_field_prob = dark_field_prob
        self.red_shift_prob = red_shift_prob
        self.close_zoom_prob = close_zoom_prob
        self.close_zoom_scale_range = close_zoom_scale_range
        self._augmentations = [
            self.simulate_close_zoom,
            self.add_blood_artifact,
            self.add_tool_occlusion,
            self.simulate_blood_pooling,
            self.simulate_specular_highlights,
            self.simulate_dark_field,
            self.simulate_red_shift,
        ]

    def add_blood_artifact(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.blood_artifact_prob:
            h, w = image.shape[:2]
            x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
            x2, y2 = random.randint(x1, w), random.randint(y1, h)

            blood = np.zeros_like(image)
            blood[y1:y2, x1:x2] = [180, 30, 30]

            alpha = random.uniform(*self.blood_artifact_alpha_range)
            image = cv2.addWeighted(image, 1 - alpha, blood, alpha, 0)

        return image, mask

    def add_tool_occlusion(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.tool_occlusion_prob:
            h, w = image.shape[:2]
            tool_w = random.randint(w // 10, w // 4)
            tool_h = random.randint(h // 20, h // 8)
            x = random.randint(0, w - tool_w)
            y = random.randint(0, h - tool_h)
            image[y:y + tool_h, x:x + tool_w] = [50, 50, 50]

        return image, mask

    def simulate_nbi(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.nbi_simulation_prob:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            image[:, :, 1] -= 20
            image[:, :, 2] += 20
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        return image

    def simulate_blood_pooling(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.blood_pooling_prob:
            tint = np.full_like(image, [140, 25, 25], dtype=np.uint8)
            alpha = random.uniform(*self.blood_pooling_alpha_range)
            image = cv2.addWeighted(image, 1 - alpha, tint, alpha, 0)

            darken = random.uniform(0.5, 0.85)
            image = np.clip(image.astype(np.float32) * darken, 0, 255).astype(np.uint8)

        return image, mask

    def simulate_specular_highlights(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.specular_highlight_prob:
            h, w = image.shape[:2]
            num_spots = random.randint(*self.specular_num_spots)
            for _ in range(num_spots):
                cx = random.randint(0, w - 1)
                cy = random.randint(0, h - 1)
                radius = random.randint(2, max(3, min(h, w) // 20))
                intensity = random.randint(220, 255)
                cv2.circle(image, (cx, cy), radius, (intensity, intensity, intensity), -1)

            if random.random() < 0.5:
                image = cv2.GaussianBlur(image, (3, 3), 0)

        return image, mask

    def simulate_dark_field(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.dark_field_prob:
            img_f = image.astype(np.float32)
            brightness_scale = random.uniform(0.35, 0.7)
            img_f *= brightness_scale

            contrast_factor = random.uniform(0.5, 0.85)
            mean_val = img_f.mean()
            img_f = mean_val + contrast_factor * (img_f - mean_val)

            image = np.clip(img_f, 0, 255).astype(np.uint8)

        return image, mask

    def simulate_red_shift(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.red_shift_prob:
            img_f = image.astype(np.float32)

            r_boost = random.uniform(1.1, 1.5)
            g_scale = random.uniform(0.65, 0.9)
            b_scale = random.uniform(0.5, 0.8)

            img_f[:, :, 0] = np.clip(img_f[:, :, 0] * r_boost, 0, 255)
            img_f[:, :, 1] = np.clip(img_f[:, :, 1] * g_scale, 0, 255)
            img_f[:, :, 2] = np.clip(img_f[:, :, 2] * b_scale, 0, 255)

            image = img_f.astype(np.uint8)

        return image, mask

    def simulate_close_zoom(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.close_zoom_prob:
            h, w = image.shape[:2]
            crop_scale = random.uniform(*self.close_zoom_scale_range)
            crop_h = int(h * crop_scale)
            crop_w = int(w * crop_scale)

            max_y = h - crop_h
            max_x = w - crop_w
            y0 = random.randint(max_y // 4, max(max_y // 4, max_y * 3 // 4))
            x0 = random.randint(max_x // 4, max(max_x // 4, max_x * 3 // 4))

            image = image[y0:y0 + crop_h, x0:x0 + crop_w]
            mask = mask[y0:y0 + crop_h, x0:x0 + crop_w]

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return image, mask

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_to_apply = random.randint(1, 2)
        selected = random.sample(self._augmentations, min(num_to_apply, len(self._augmentations)))
        for aug_fn in selected:
            image, mask = aug_fn(image, mask)

        image = self.simulate_nbi(image)

        return image, mask


class DirectoryImageMaskDataset(Dataset):
    def __init__(
        self,
        directories: List[str],
        image_size: Tuple[int, int] = (512, 512),
        is_train: bool = True,
        normalize: bool = True,
        enable_endoscopy_augmentations: bool = True,
        augmentation_config: dict = None,
        endoscopy_config: dict = None,
        mask_threshold: int = None,
        supported_extensions: set = None,
    ):
        self.directories = directories
        self.image_size = image_size
        self.is_train = is_train
        self.normalize = normalize
        self.enable_endoscopy_augmentations = enable_endoscopy_augmentations and is_train
        self._mask_threshold = mask_threshold if mask_threshold is not None else 127
        self._supported_extensions = supported_extensions

        self.logger = logging.getLogger(self.__class__.__name__)

        self.image_mask_pairs, self.domain_indices, self.dir_counts = self._find_pairs_with_domains(directories)
        if not self.image_mask_pairs:
            raise ValueError(f"No image-mask pairs found in directories: {directories}")

        self.logger.info(f"Found {len(self.image_mask_pairs)} image-mask pairs")

        self.augmentation = SurgicalAugmentationPipeline(
            image_size=image_size,
            is_train=is_train,
            augmentation_config=augmentation_config,
        )

        if self.enable_endoscopy_augmentations:
            if endoscopy_config:
                ec = endoscopy_config
                self.endoscopy_augmentor = EndoscopyAugmentor(
                    blood_artifact_prob=ec.get('blood_artifact_prob', 0.2),
                    blood_artifact_alpha_range=(ec.get('blood_artifact_alpha_min', 0.1), ec.get('blood_artifact_alpha_max', 0.4)),
                    tool_occlusion_prob=ec.get('tool_occlusion_prob', 0.15),
                    nbi_simulation_prob=ec.get('nbi_simulation_prob', 0.15),
                    blood_pooling_prob=ec.get('blood_pooling_prob', 0.25),
                    blood_pooling_alpha_range=(ec.get('blood_pooling_alpha_min', 0.15), ec.get('blood_pooling_alpha_max', 0.45)),
                    specular_highlight_prob=ec.get('specular_highlight_prob', 0.3),
                    specular_num_spots=(ec.get('specular_num_spots_min', 3), ec.get('specular_num_spots_max', 12)),
                    dark_field_prob=ec.get('dark_field_prob', 0.25),
                    red_shift_prob=ec.get('red_shift_prob', 0.3),
                    close_zoom_prob=ec.get('close_zoom_prob', 0.3),
                    close_zoom_scale_range=(ec.get('close_zoom_scale_min', 0.4), ec.get('close_zoom_scale_max', 0.8)),
                )
            else:
                self.endoscopy_augmentor = EndoscopyAugmentor()
        else:
            self.endoscopy_augmentor = None

        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]

    def _find_pairs_with_domains(self, directories: List[str]) -> Tuple[List[Tuple[Path, Path]], List[int], List[int]]:
        pairs = []
        domain_indices = []
        dir_counts = []

        for dir_idx, directory in enumerate(directories):
            dir_pairs = find_image_mask_pairs([directory], supported_extensions=self._supported_extensions)
            pairs.extend(dir_pairs)
            domain_indices.extend([dir_idx] * len(dir_pairs))
            dir_counts.append(len(dir_pairs))

        return pairs, domain_indices, dir_counts

    def get_sample_weights(self, target_domain_weight: float = 0.4) -> List[float]:
        n_dirs = len(self.dir_counts)
        if n_dirs <= 1:
            return [1.0] * len(self)

        target_idx = n_dirs - 1
        n_target = self.dir_counts[target_idx]
        n_source = len(self) - n_target

        if n_target == 0 or n_source == 0:
            return [1.0] * len(self)

        w_target = target_domain_weight / n_target
        w_source = (1.0 - target_domain_weight) / n_source

        return [
            w_target if di == target_idx else w_source
            for di in self.domain_indices
        ]

    def __len__(self) -> int:
        return len(self.image_mask_pairs)

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = Image.open(path)
        if mask.mode != 'L':
            mask = mask.convert('L')
        mask_array = np.array(mask)
        mask_array = (mask_array > self._mask_threshold).astype(np.uint8)
        return mask_array

    def _apply_augmentations(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = self.augmentation(image, mask)

        if self.endoscopy_augmentor is not None:
            image, mask = self.endoscopy_augmentor(image, mask)

        return image, mask

    def _to_tensor(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - self.IMAGENET_MEAN) / self.IMAGENET_STD

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, mask_path = self.image_mask_pairs[idx]

        try:
            image = self._load_image(image_path)
        except Exception as e:
            self.logger.warning(f"Failed to load image {image_path}: {e}")
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)

        try:
            mask = self._load_mask(mask_path)
        except Exception as e:
            self.logger.warning(f"Failed to load mask {mask_path}: {e}")
            mask = np.zeros(self.image_size, dtype=np.uint8)

        image, mask = self._apply_augmentations(image, mask)

        image_tensor, mask_tensor = self._to_tensor(image, mask)

        return {
            'images': image_tensor,
            'masks': mask_tensor,
            'image_path': str(image_path),
        }


class MixUpDataset(Dataset):
    def __init__(self, dataset: Dataset, alpha: float = 0.4, prob: float = 0.5):
        self.dataset = dataset
        self.alpha = alpha
        self.prob = prob

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample1 = self.dataset[idx]

        if random.random() < self.prob:
            idx2 = random.randint(0, len(self.dataset) - 1)
            sample2 = self.dataset[idx2]

            lam = np.random.beta(self.alpha, self.alpha)

            sample1['images'] = lam * sample1['images'] + (1 - lam) * sample2['images']
            sample1['masks'] = lam * sample1['masks'] + (1 - lam) * sample2['masks']

        return sample1


def create_dataloader(
    directories: List[str],
    image_size: Tuple[int, int] = (512, 512),
    is_train: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    normalize: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = DirectoryImageMaskDataset(
        directories=directories,
        image_size=image_size,
        is_train=is_train,
        normalize=normalize,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader


class SurgicalDataLoader:
    def __init__(
        self,
        data_root: str,
        image_size: Tuple[int, int] = (512, 512),
        batch_size: int = 8,
        num_workers: int = 4,
        normalize: bool = True,
        train_augment: bool = True,
        enable_endoscopy_augmentations: bool = True,
        use_weighted_sampling: bool = False,
        target_domain_weight: float = 0.4,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        augmentation_config: dict = None,
        endoscopy_config: dict = None,
        mask_threshold: int = 127,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.train_augment = train_augment
        self.enable_endoscopy_augmentations = enable_endoscopy_augmentations
        self.use_weighted_sampling = use_weighted_sampling
        self.target_domain_weight = target_domain_weight
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.augmentation_config = augmentation_config
        self.endoscopy_config = endoscopy_config
        self.mask_threshold = mask_threshold

        self.logger = logging.getLogger(self.__class__.__name__)

        self.train_dir = self.data_root / 'train'
        self.val_dir = self.data_root / 'val'
        self.test_dir = self.data_root / 'test'

        self._validate_directories()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._load_all()

    def _validate_directories(self):
        missing = []
        if not self.train_dir.exists():
            missing.append(str(self.train_dir))
        if not self.val_dir.exists():
            missing.append(str(self.val_dir))
        if not self.test_dir.exists():
            missing.append(str(self.test_dir))

        if missing:
            raise FileNotFoundError(f"Missing directories: {missing}")

    def _load_all(self):
        pin_memory = torch.cuda.is_available()

        train_dataset = DirectoryImageMaskDataset(
            directories=[str(self.train_dir)],
            image_size=self.image_size,
            is_train=self.train_augment,
            normalize=self.normalize,
            enable_endoscopy_augmentations=self.enable_endoscopy_augmentations,
            augmentation_config=self.augmentation_config,
            endoscopy_config=self.endoscopy_config,
            mask_threshold=self.mask_threshold,
        )

        if self.use_weighted_sampling and len(train_dataset.dir_counts) > 1:
            sample_weights = train_dataset.get_sample_weights(self.target_domain_weight)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = self.train_augment

        if self.use_mixup and self.train_augment:
            train_dataset = MixUpDataset(
                train_dataset,
                alpha=self.mixup_alpha,
                prob=self.mixup_prob
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle and sampler is None,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

        val_dataset = DirectoryImageMaskDataset(
            directories=[str(self.val_dir)],
            image_size=self.image_size,
            is_train=False,
            normalize=self.normalize,
            enable_endoscopy_augmentations=False,
            mask_threshold=self.mask_threshold,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        test_dataset = DirectoryImageMaskDataset(
            directories=[str(self.test_dir)],
            image_size=self.image_size,
            is_train=False,
            normalize=self.normalize,
            enable_endoscopy_augmentations=False,
            mask_threshold=self.mask_threshold,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        self.logger.info(f"Loaded dataloaders: train={len(self.train_loader)}, val={len(self.val_loader)}, test={len(self.test_loader)}")

    def get_dataloaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        return self.train_loader, self.val_loader, self.test_loader

    def get_dataset_stats(self) -> Dict[str, Dict]:
        stats = {}
        for name, loader in [('train', self.train_loader), ('val', self.val_loader), ('test', self.test_loader)]:
            if loader is None:
                continue

            dataset = loader.dataset
            num_samples = len(dataset)
            num_batches = len(loader)

            image_files = [pair[0] for pair in dataset.image_mask_pairs]
            mask_files = [pair[1] for pair in dataset.image_mask_pairs]

            extensions = {}
            for f in image_files:
                ext = f.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1

            stats[name] = {
                'num_samples': num_samples,
                'num_batches': num_batches,
                'image_extensions': extensions,
            }

        return stats

    def print_dataset_info(self):
        print(f"\n{'='*70}")
        print(f"Surgical DataLoader - Data Root: {self.data_root}")
        print(f"{'='*70}")

        for name, loader in [('train', self.train_loader), ('val', self.val_loader), ('test', self.test_loader)]:
            if loader is None:
                continue

            dataset = loader.dataset
            print(f"\n{name.upper()} Dataset:")
            print(f"  Samples: {len(dataset)}")
            print(f"  Batches: {len(loader)}")
            print(f"  Image size: {dataset.image_size}")
            print(f"  Augmentations: {'enabled' if dataset.is_train else 'disabled'}")

            stats = self.get_dataset_stats()
            if name in stats:
                print(f"  Image extensions: {stats[name]['image_extensions']}")

        print(f"{'='*70}\n")

    def verify_matching(self) -> Dict[str, List[str]]:
        issues = {'train': [], 'val': [], 'test': []}

        for name, data_dir in [('train', self.train_dir), ('val', self.val_dir), ('test', self.test_dir)]:
            pairs = find_image_mask_pairs([str(data_dir)])
            if not pairs:
                issues[name].append(f"No image-mask pairs found in {data_dir}")
                continue

            unmatched_images = []
            for img_path, mask_path in pairs:
                if not img_path.exists():
                    unmatched_images.append(str(img_path))
                if not mask_path.exists():
                    unmatched_images.append(str(mask_path))

            if unmatched_images:
                issues[name].extend(unmatched_images[:10])

        return issues
