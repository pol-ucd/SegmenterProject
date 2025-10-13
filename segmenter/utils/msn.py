from typing import Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, transforms as T

from segmenter.utils import pretrain_transform
from segmenter.utils.data import MSNPretrainDatasetHDF5, get_num_samples_from_hdf5, MSNFinetuneDatasetHDF5, \
    HDF5BatchSampler, hdf5_worker_init_fn, HDF5DatasetOptimized, pretrain_transform, HDF5BatchSubsetSampler, \
    finetune_transform

# PRETRAIN_DATASETS = ['../segmenter/data/dresden_preprocessed.h5',
#                      '../segmenter/data/all_data.h5']
PRETRAIN_DATASET = '../segmenter/data/pretrain_images.h5'

FINETUNE_DATASET = '../segmenter/data/Classica.h5'


class MSNDataHandler:
    def __init__(self, config: Any):
        self.config = config
        self.batch_size = config['run']['batch_size']
        self.num_workers = config['run']['num_workers']

        data_opts = config['run']['data']
        self.pretrain_dataset = data_opts['pretrain_dataset']
        self.train_dataset = data_opts['train_dataset']

        self.eval_percent = data_opts['eval_percent']
        self.image_size = data_opts['image_size']

        self.image_augment = T.Compose([T.Resize(self.image_size,
                                                 T.InterpolationMode.BICUBIC),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                        ])

    def load_pretrain_dataset(self):
        pretrain_ds = HDF5DatasetOptimized(hdf5_path=self.pretrain_dataset,
                                           data_keys=['images'],
                                           transform=self.image_augment)

        custom_sampler = HDF5BatchSampler(pretrain_ds.dataset_len,
                                          self.batch_size, shuffle=True)

        pretrain_dl = torch.utils.data.DataLoader(
            pretrain_ds,
            batch_size=None,
            sampler=custom_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=hdf5_worker_init_fn
        )


def load_data(batch_size: int, finetune_percent: float,
              batch_size_finetune: int = 8,
              image_size=(512, 512),
              num_workers: int = 4) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    pretrain_loader = load_pretrain(data_path=PRETRAIN_DATASET,
                                    image_size=image_size,
                                    batch_size=batch_size,
                                    num_workers=num_workers)

    finetune_loader, validation_loader = load_finetune(data_path=FINETUNE_DATASET,
                                                       batch_size=batch_size_finetune,
                                                       num_workers=num_workers,
                                                       image_size=image_size,
                                                       finetune_percent=finetune_percent)

    return finetune_loader, pretrain_loader, validation_loader


def load_pretrain(data_path: Optional[str] = None,
                  batch_size: int = 8,
                  image_size: Tuple[int, int] = (512, 512),
                  num_workers: int = 4) -> DataLoader[Any]:
    if data_path is None:
        data_path = PRETRAIN_DATASET

    pretrain_dataset = HDF5DatasetOptimized(hdf5_path=data_path,
                                            data_keys=['images'],
                                            transform=pretrain_transform)

    custom_sampler = HDF5BatchSampler(pretrain_dataset.dataset_len,
                                      batch_size,
                                      shuffle=True)

    pretrain_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=None,
        sampler=custom_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=hdf5_worker_init_fn
    )

    return pretrain_dataloader


def load_finetune(data_path: Optional[str] = None,
                  batch_size: int = 8,
                  finetune_percent: float = 0.1,
                  image_size: Tuple[int, int] = (512, 512),
                  num_workers: int = 4) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    # Small Annotated set for Fine-tuning

    finetune_data = data_path if data_path is not None else FINETUNE_DATASET

    n_finetune = get_num_samples_from_hdf5(finetune_data)
    shuffled_indices = np.random.permutation(n_finetune)
    n_finetune = int(n_finetune * finetune_percent)
    finetune_indices = shuffled_indices[:n_finetune]
    validation_indices = shuffled_indices[n_finetune:]


    while (len(finetune_indices)) % batch_size != 0:
        finetune_indices = np.append(finetune_indices, validation_indices[-1])


    while (len(validation_indices)) % batch_size != 0:
        validation_indices = np.append(validation_indices, validation_indices[-1])

    subset_sampler_finetune = HDF5BatchSubsetSampler(dataset_size=n_finetune,
                                                     batch_size=batch_size,
                                                     indices=finetune_indices)

    subset_sampler_validation = HDF5BatchSubsetSampler(dataset_size=n_finetune,
                                                       batch_size=batch_size,
                                                       indices=validation_indices)

    finetune_dataset = HDF5DatasetOptimized(hdf5_path=finetune_data,
                                            data_keys=['images', 'masks'],
                                            transform=finetune_transform)

    finetune_dataloader = torch.utils.data.DataLoader(finetune_dataset,
                                                      batch_size=batch_size,
                                                      sampler=subset_sampler_finetune,
                                                      shuffle=False,
                                                      num_workers=num_workers,
                                                      pin_memory=True,
                                                      worker_init_fn=hdf5_worker_init_fn)

    validation_dataloader = torch.utils.data.DataLoader(finetune_dataset,
                                                        batch_size=batch_size,
                                                        sampler=subset_sampler_validation,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        worker_init_fn=hdf5_worker_init_fn)

    return finetune_dataloader, validation_dataloader
