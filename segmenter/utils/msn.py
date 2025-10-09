from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, transforms as T

from segmenter.utils import pretrain_transform
from segmenter.utils.data import MSNPretrainDatasetHDF5, get_num_samples_from_hdf5, MSNFinetuneDatasetHDF5, \
    HDF5BatchSampler, hdf5_worker_init_fn, HDF5DatasetOptimized, pretrain_transform

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
              image_size=(512, 512),
              num_workers: int = 4) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    # image_augment = T.Compose([T.Resize(image_size,
    #                                     T.InterpolationMode.BICUBIC),
    #                            T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                        std=[0.229, 0.224, 0.225])
    #                            ])

    # pretrain_dataset = MSNPretrainDatasetHDF5(hdf5_path=PRETRAIN_DATASET)
    pretrain_dataset = HDF5DatasetOptimized(hdf5_path=PRETRAIN_DATASET,
                                            data_keys=['images'],
                                            transform=pretrain_transform)

    # total_len = get_num_samples_from_hdf5(hdf5_path=PRETRAIN_DATASET)

    custom_sampler = HDF5BatchSampler(pretrain_dataset.dataset_len,
                                      batch_size, shuffle=True)

    pretrain_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=None,
        sampler=custom_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=hdf5_worker_init_fn
    )

    # Small Annotated set for Fine-tuning
    finetune_data = FINETUNE_DATASET
    n_finetune = get_num_samples_from_hdf5(finetune_data)
    shuffled_indices = np.random.permutation(n_finetune)
    n_finetune = int(n_finetune * finetune_percent)
    finetune_indices = shuffled_indices[:n_finetune]
    validation_indices = shuffled_indices[n_finetune:]

    finetune_dataset = MSNFinetuneDatasetHDF5(hdf5_path=finetune_data,
                                              indices=finetune_indices)
    finetune_dataloader = torch.utils.data.DataLoader(
        finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        prefetch_factor=batch_size
    )

    # Annotated set for Validation
    validation_dataset = MSNFinetuneDatasetHDF5(hdf5_path=finetune_data,
                                                indices=validation_indices)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return finetune_dataloader, pretrain_dataloader, validation_dataloader
