import argparse
import logging
import math
import sys
import traceback
from typing import Dict, Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import GradScaler, autocast
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm
from transformers import SegformerModel, SegformerConfig

from segmenter.loss import EPSILON
from segmenter.masks import CompositeMask
from segmenter.utils import HDF5DatasetOptimized, HDF5BatchSampler
from segmenter.utils.data import SSLTransformPipeline, hdf5_worker_init_fn

torch.autograd.set_detect_anomaly(True)

backbone = "nvidia/segformer-b4-finetuned-ade-512-512"
data_source = '../segmenter/data/pretrain_images.h5'
image_size = 256
MASK_VALUE = 1e-06


class MIMUpscalerMLP(nn.Module):
    """
    linear layer which will unify the channel dimension of each of the encoder blocks
    to the same 'output_dim' (Usually config.decoder_hidden_size for a Segformer model)
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_state: torch.Tensor):
        if hidden_state.dim() != 4:
            raise ValueError("Expected hidden_states to be a 4D tensor")

        hidden_state = hidden_state.flatten(2).transpose(1, 2)
        hidden_state = self.proj(hidden_state)
        return hidden_state


# class SegformerDecodeHead(SegformerPreTrainedModel):
class MIMSegformerReconstructionHead(nn.Module):
    """
    Reconstruct a full image segmentation model
    """

    def __init__(self, config):
        super().__init__()

        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = MIMUpscalerMLP(input_dim=config.hidden_sizes[i], output_dim=config.decoder_hidden_size)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size, eps=1e-4)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        self.reconstruction_head = nn.Linear(config.decoder_hidden_size,
                                             3)

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        """
        Stack all of the encodings 
        """
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))

        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        b, d, h, w = hidden_states.shape
        hidden_states = hidden_states.permute(1, 0, 2, 3).reshape(d, -1).T

        logits = self.reconstruction_head(hidden_states).reshape(b, -1, h, w)

        return logits


class SimCLRSegFormer(nn.Module):
    def __init__(self, config, projection_dim=128):

        super().__init__()
        self.encoder = SegformerModel.from_pretrained(pretrained_model_name_or_path=backbone,
                                                      config=config,
                                                      ignore_mismatched_sizes=True)

        self.encoder_hidden_sizes = config.hidden_sizes
        self.strides = config.strides
        self.num_blocks = config.num_encoder_blocks

        self.reconstruction_head = MIMSegformerReconstructionHead(config)

    def forward(self, x1, x2):
        # Encode both views
        f1 = self.encoder(x1, output_hidden_states=True,
                               return_dict=True).hidden_states
        f2 = self.encoder(x2, output_hidden_states=True,
                               return_dict=True).hidden_states


        # Project to latent space
        z1 = F.normalize(self.reconstruction_head(f1), dim=1)
        z2 = F.normalize(self.reconstruction_head(f2), dim=1)

        return z1, z2


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature if 0 < temperature < 1 else 0.11

    def forward(self, z1, z2):
        return nt_xent_loss(z1, z2, temperature=self.temperature)

    __call__ = forward


def nt_xent_loss(z1, z2, temperature=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]
    sim_i_j = torch.diag(sim, B)
    sim_j_i = torch.diag(sim, -B)
    positives = torch.cat([sim_i_j, sim_j_i], dim=0)

    mask = ~torch.eye(2 * B, dtype=torch.bool).to(z.device)
    negatives = sim[mask].view(2 * B, -1)

    labels = torch.zeros(2 * B).to(z.device).long()
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    return F.cross_entropy(logits, labels)

def check_is_finite(logger: logging.Logger, x: torch.Tensor, label: str = None) -> bool:
    label = label or ''
    if torch.any(torch.isinf(x)):
        logger.warning(f"{label} Tensor is not finite!")
        return False
    return True

def check_is_attached(logger: logging.Logger, x: torch.Tensor, label: str = None) -> bool:
    try:
        assert x.requires_grad and x.grad_fn is not None
    except AssertionError:
        logger.error(f"{label} Tensor is detached from graph")
        logger.error(
            f":Device {x.device}, dtype:{x.dtype}, requires_grad: {x.requires_grad}, grad_fn: {x.grad_fn}")
        return False
    return True


def nan_hook(name):
    def hook(model, input, output):
        logger = logging.getLogger()
        if isinstance(output, Iterable):
            if not torch.isfinite(output[0]).all():
                logger.error(f"nan_hook: Invalid output in {name}")
        elif isinstance(output, torch.Tensor):
            if not torch.isfinite(output).all():
                logger.error(f"nan_hook: Invalid output in {name}")
        else:
            logger.error(f"nan_hook: Invalid output in {name} - "
                         f"expected Iterable or torch.Tensor type not: {type(output)}")

        if isinstance(input, Iterable):
            if not torch.isfinite(input[0]).all():
                logger.error(f"nan_hook: Invalid input in {name}")
        elif isinstance(input, torch.Tensor):
            if not torch.isfinite(input).all():
                logger.error(f"nan_hook: Invalid input in {name}")
        else:
            logger.error(f"nan_hook: Invalid input in {name} - "
                         f"expected Iterable or torch.Tensor type not: {type(input)}")

    return hook


# Example usage
def main(params: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger()

    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    num_workers = params['num_workers']

    prefix = 'attention_mim'

    ds = HDF5DatasetOptimized(hdf5_path=params['dataset'],
                              transform=SSLTransformPipeline(size=image_size))

    batch_sampler = HDF5BatchSampler(ds.dataset_len,
                                     batch_size=batch_size,
                                     shuffle=True)

    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size=None,
                                             sampler=batch_sampler,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             worker_init_fn=hdf5_worker_init_fn
                                             )

    config = SegformerConfig.from_pretrained(backbone)

    """
    Tweak the config settings to trim it down a bit.
    
    We only care about 2 class output since we'll be using it for yes/no classification
    
    """
    config.image_size = image_size
    # config.num_labels = 2
    # config.id2label = {0: 'negative', 1: 'positive'}
    # config.label2label = {'negative': 0, 'positive': 1}

    model = SimCLRSegFormer(config)

    loss_fn = NTXentLoss(temperature=0.1)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    model.to(device)

    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # -----------------------------

    logger.info("Starting Self-Supervised Pre-training...")

    """ Set up stopping criteria - stop after 'boredom' steps do not improve loss by 'min_delta' """
    best_loss = float('inf')
    min_delta = 0.0000001
    boredom = 0
    max_boredom = 10
    best_model = None

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_epoch_loss = []

        # Iterate over the dataset
        for step, data in enumerate(tqdm(dataloader)):

            x_1 = data['targets'].to(device)
            x_2 = data['anchors'].to(device)
            b, c, h, w = x_1.shape

            check_is_finite(logger, x_1, "x_1, input image")
            check_is_finite(logger, x_2, "x_2, input image")

            optimizer.zero_grad()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float):
                z_1, z_2 = model(x_1, x_2)
                z_1 = z_1.reshape(b, -1)
                z_2 = z_2.reshape(b, -1)

                loss_total = loss_fn(z_1, z_2)

            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.step()

            # 6. Logging and Tracking
            total_epoch_loss += [loss_total.item()]

        if scheduler is not None:
            scheduler.step()

        avg_epoch_loss = np.mean(total_epoch_loss)

        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss + min_delta < best_loss:
            best_loss = avg_epoch_loss
            boredom = 0
            logger.info("Saving best snapshot of SegFormer state dict for fine-tuning.")
            try:
                best_model = model.encoder.state_dict()
                torch.save(best_model,
                           f'../segmenter/checkpoint/{prefix}_segformer_pretrained.pt')
            except Exception as e:
                logger.error(f"Pretraining failed to save `{prefix}_segformer_pretrained.pt`: {e}")

        else:
            logger.info(f"Getting bored after {boredom} epochs with no useful improvement")
            boredom += 1
        if boredom > max_boredom:
            logger.info(f"No improvement after {boredom} epochs, terminating")
            break

    logger.info("Pre-training complete.")


def get_args():
    """
    Command line arguments

    :return: Dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=data_source,
                        type=str, help="Path to the HDF5 file.")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, )
    parser.add_argument("-nw", "--num_workers", type=int, default=4, )
    parser.add_argument("-e", "--num_epochs", type=int, default=200, )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, )
    parser.add_argument("-p", "--prefix", type=str, default='attention_mim', )
    parser.add_argument("-ro", "--run_once", type=bool, default=False, )
    parser.add_argument("-ns", "--num_shapes", type=int, default=24, )

    args = parser.parse_args()

    params = {'dataset': args.input,
              'batch_size': args.batch_size,
              'num_workers': args.num_workers,
              'num_epochs': args.num_epochs,
              'learning_rate': args.learning_rate,
              'prefix': args.prefix,
              'run_once': bool(args.run_once),
              'num_shapes': int(args.num_shapes), }

    return params


if __name__ == "__main__":
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        force=True,  # Resets any previous configuration - in Colab for example
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )
    logger = logging.getLogger()
    try:
        params = get_args()
        main(params)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down gracefully.")
    except Exception as ex:
        logger.error(f"Unknown exception occurred. Error: {ex}")
        logger.error(traceback.format_exc())
    finally:
        # ensure log handlers are flushed.
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.info("Logger handlers flushed and closed. Exiting now.")
        sys.exit(0)
