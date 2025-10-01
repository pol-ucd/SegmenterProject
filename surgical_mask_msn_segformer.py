import os
from os import minor

import torch
from torch.utils.data import DataLoader, ConcatDataset

from segmenter.core import freeze_seed
from segmenter.loss import MSNLoss
from segmenter.models.base import SegformerBackbone
from segmenter.models.msn import MoCoSegformer
from segmenter.utils import SurgicalAugmentor
from segmenter.utils.surgical import SurgicalSiameseDatasetHDF5, SurgicalMaskComposer


DATASETS = ['../segmenter/data/dresden_preprocessed.h5',
            '../segmenter/data/all_data.h5',
            '../segmenter/data/Classica.h5']
SHAPE = (512, 512)
BATCH_SIZE = 8
IMG_SIZE = 224
FOCAL_IMG_SIZE = 96
# MODEL_PATCH_SIZE = 16  # SegFormer-B0 uses patch size 4, but effective stride is larger. We use 16 for feature map.
MASK_RATIO = 0.5
FEATURE_DIM = 256  # Should match the backbone's output_dim
LEARNING_RATE = 1e-4
EPOCHS = 100

if __name__ == '__main__':

    # For reproducibility
    freeze_seed()

    # Create a dummy dataset (replace with your endoscopic image dataset) ---
    print("Loading dataset of surgical images...")

    # Instantiate dataset and augmentations
    mask_composer = SurgicalMaskComposer(shape=SHAPE, channels=3)
    augmentor = SurgicalAugmentor(size=SHAPE)
    datasets = []
    for dataset in DATASETS:
        datasets.append(SurgicalSiameseDatasetHDF5(hdf5_path=dataset,
                                                   mask_composer=mask_composer,
                                                   augmentor=augmentor))

    full_dataset = ConcatDataset(datasets)

    # full_dataset = SurgicalSiameseDatasetHDF5(hdf5_path=DATASET,
    #                                           mask_composer=mask_composer,
    #                                           augmentor=augmentor)

    n_workers = min(BATCH_SIZE, os.cpu_count())
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=None, num_workers=n_workers)

    print(f"Dataloader created with {len(full_dataset)} data records in {len(dataloader)} batches.")

    print("Initializing SegFormer backbone and MSN model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segformer_backbone = SegformerBackbone(output_dim=FEATURE_DIM)
    msn_model = MoCoSegformer(backbone=segformer_backbone).to(device)

    optimizer = torch.optim.AdamW(msn_model.online_encoder.parameters(), lr=LEARNING_RATE)
    msn_loss = MSNLoss()

    print(f"Starting self-supervised training for {EPOCHS} epochs on {device}...")
    best_loss = float('inf')
    min_delta = 0.00001
    boredom = 0
    max_boredom = 20
    for epoch in range(EPOCHS):
        total_loss = 0
        for i_batch, data_view in enumerate(dataloader):
            global_views = data_view['view1'].to(device)
            focal_views = data_view['view2'].to(device)

            # Forward pass
            online_preds, target_protos = msn_model(focal_views, global_views)  #, mask)

            # The target prototypes are from the entire batch, so we need to concatenate them
            # B, N, C -> (B*N), C
            target_protos = target_protos.reshape(-1, FEATURE_DIM)

            # Calculate loss
            loss = msn_loss(online_preds, target_protos)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target network using EMA
            msn_model._update_target_network()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Average Loss: {avg_loss:.4f}")
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            boredom = 0
            print("Saving best snapshot `msn_model.online_encoder` state dict for fine-tuning.")
            torch.save(msn_model.online_encoder.state_dict(),
                       '../segmenter/checkpoint/segformer_msn_pretrained_backbone.pth')

        else:
            boredom += 1
        if boredom > max_boredom:
            print(f"No improvement after {boredom} epochs, terminating")
            break

    print("Self-supervised pre-training complete!")
