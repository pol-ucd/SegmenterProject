import random
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from segmenter.masks import InstrumentMask, FluidMask, FoldMask, RandomShapeMask
from segmenter.utils.data import HDF5Dataset


class SurgicalMaskComposer:
    def __init__(self, shape, channels=3):
        self.H, self.W = shape
        self.C = channels
        self.instrument = InstrumentMask(shape=(self.H, self.W), channels=1)
        self.fluid = FluidMask(shape=(self.H, self.W), channels=1)
        self.fold = FoldMask(shape=(self.H, self.W), channels=1)
        self.shape = RandomShapeMask(shape=(self.H, self.W), channels=1)

    def generate_batch(self, batch_size):
        masks = []
        metadata = []

        for _ in range(batch_size):
            mask, info = self._generate_single()
            masks.append(mask)
            metadata.append(info)

        return torch.stack(masks), metadata

    def _generate_single(self):
        composite = torch.zeros(self.H, self.W)
        info = {}

        # Randomize mask types
        use_instrument = random.random() < 0.5
        use_fluid = random.random() < 0.5
        use_fold = random.random() < 0.5
        use_shape = random.random() < 0.3

        if use_instrument:
            m = self.instrument().squeeze()
            composite += m
            # info['instrument'] = (m == 0).nonzero(as_tuple=False)
            info['instrument'] = (m > 0).float()

        if use_fluid:
            m = self.fluid().squeeze()
            composite += m
            # info['fluid'] = (m < 0.95).nonzero(as_tuple=False)
            info['fluid'] = (m > 0).float()

        if use_fold:
            m = self.fold().squeeze()
            composite += m
            # info['fold'] = (m == 0).nonzero(as_tuple=False)
            info['fold'] = (m > 0).float()

        if use_shape:
            m = self.shape().squeeze()
            composite += m
            # info['shape'] = (m == 0).nonzero(as_tuple=False)
            info['shape'] = (m > 0).float()

        # Multi-scale: downsample and upsample to blur edges
        composite = torch.nn.functional.avg_pool2d(composite.unsqueeze(0).unsqueeze(0),
                                                   kernel_size=3, stride=1, padding=1)[0, 0]
        composite = composite.clamp(0, 1)

        # Expand to channels
        final_mask = composite.unsqueeze(0).repeat(self.C, 1, 1)
        # info['final_mask'] = (composite < 0.95).nonzero(as_tuple=False)
        info['final_mask'] = (composite > 0).float()

        return final_mask, info


class SurgicalAugmentor:
    def __init__(self, size: Tuple[int, int]=(256, 256)):
        self.augment = T.Compose([
            T.RandomResizedCrop(size=size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        return self.augment(image)


class SurgicalSiameseDataset(Dataset):
    def __init__(self, image_paths,
                 mask_composer = None,
                 augmentor=None):
        self.image_paths = image_paths
        self.mask_composer = mask_composer
        self.augmentor = augmentor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and convert to tensor
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = T.ToTensor()(image)

        # Apply augmentations
        if self.augmentor:
            image = self.augmentor(image)

        # Generate two masked views and metadata
        mask1, meta1 = self.mask_composer._generate_single()
        mask2, meta2 = self.mask_composer._generate_single()

        view1 = image * mask1
        view2 = image * mask2

        return {
            'view1': view1,
            'view2': view2,
            'original': image,
            'mask1': mask1,
            'mask2': mask2,
            'meta1': meta1,
            'meta2': meta2
        }


class SurgicalSiameseDatasetHDF5(HDF5Dataset):
    def __init__(self, hdf5_path,
                 mask_composer = None,
                 augmentor=None):
        super().__init__(hdf5_path)
        self.mask_composer = mask_composer
        self.augmentor = augmentor


    def __getitem__(self, idx):
        # Load image and convert to tensor
        image = self.data['images'][idx]
        image = T.ToTensor()(image)

        # Apply augmentations
        if self.augmentor:
            image = self.augmentor(image)

        # Generate two masked views and metadata
        mask1, meta1 = self.mask_composer._generate_single()
        mask2, meta2 = self.mask_composer._generate_single()

        view1 = image * mask1
        view2 = image * mask2

        return {
            'view1': view1,
            'view2': view2,
            'original': image,
            'mask1': mask1,
            'mask2': mask2,
            # 'meta1': meta1,
            # 'meta2': meta2
        }



class MaskScheduler:
    def __init__(self, start_ratio=0.3, end_ratio=0.05, total_epochs=100, decay='linear'):
        self.start = start_ratio
        self.end = end_ratio
        self.total = total_epochs
        self.decay = decay

    def get_ratio(self, epoch):
        progress = min(epoch / self.total, 1.0)
        if self.decay == 'linear':
            return self.start - progress * (self.start - self.end)
        elif self.decay == 'exponential':
            return self.start * ((self.end / self.start) ** progress)
        else:
            raise ValueError("Unsupported decay type")


if __name__ == '__main__':
    sc = SurgicalMaskComposer(shape=(256, 256), channels=3)
    masks, info = sc._generate_single()

    trans = torchvision.transforms.ToPILImage()
    out = trans(masks[0])
    out.show()
