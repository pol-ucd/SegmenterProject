from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from segmenter.utils.data import MSNPretrainDatasetHDF5, get_num_samples_from_hdf5, MSNFinetuneDatasetHDF5


PRETRAIN_DATASETS = ['../segmenter/data/dresden_preprocessed.h5',
                     '../segmenter/data/all_data.h5']

FINETUNE_DATASETS = ['../segmenter/data/Classica.h5']


def load_data(batch_size: int, finetune_percent: float) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    # Large Unannotated set for Pre-training
    pretrain_datasets = []
    for ds in PRETRAIN_DATASETS:
        pretrain_datasets.append(MSNPretrainDatasetHDF5(hdf5_path=ds))

    pretrain_dataset = ConcatDataset(pretrain_datasets)

    pretrain_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
        prefetch_factor=batch_size
    )

    # Small Annotated set for Fine-tuning
    finetune_data = FINETUNE_DATASETS[0]
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
