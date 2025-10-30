import argparse
import logging
import math
import sys
import traceback
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import GradScaler, autocast
from torch.nn import MSELoss
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm
from transformers import SegformerModel, SegformerConfig

from segmenter.utils import HDF5DatasetOptimized, HDF5BatchSampler
from segmenter.utils.data import SSLTransformPipeline, hdf5_worker_init_fn

torch.autograd.set_detect_anomaly(True)

backbone = "nvidia/segformer-b4-finetuned-ade-512-512"
data_source = '../segmenter/data/pretrain_images.h5'
image_size = 256


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
class MIMSegformerDecodeHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = MIMUpscalerMLP(input_dim=config.hidden_sizes[i], output_dim=config.decoder_hidden_size)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

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

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


class AttentionMaskingMIM(nn.Module):
    def __init__(self, config, mask_ratio=0.4):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(pretrained_model_name_or_path=backbone,
                                                      config=config,
                                                      ignore_mismatched_sizes=True)
        self.encoder_hidden_sizes = config.hidden_sizes
        self.strides = config.strides
        self.num_blocks = config.num_encoder_blocks
        self.mask_ratio = mask_ratio
        # self.reconstruction_head = nn.Sequential(
        #     nn.Linear(self.encoder.config.hidden_sizes[-1], 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 3 * 16 * 16)  # Assuming 16x16 patch reconstruction
        # )
        self.reconstruction_head = MIMSegformerDecodeHead(config)

    def generate_attention_mask(self, attention_map):
        B, H, W = attention_map.shape
        flat = attention_map.view(B, -1)
        _, indices = torch.topk(flat, int(H * W * self.mask_ratio), dim=1, largest=False)
        mask = torch.ones_like(flat)
        mask.scatter_(1, indices, 0)
        return mask.view(B, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        print(B, C, H, W)
        scaling = math.prod(self.strides)
        h_0, w_0 = H // scaling, W // scaling
        with torch.no_grad():
            features = self.encoder.encoder(x).last_hidden_state  # [B, N, C]
            print(features.shape)
            attn_map = torch.norm(features, dim=1).view(B, h_0, w_0)  # Approximate attention proxy

        mask = self.generate_attention_mask(attn_map).requires_grad_(True)
        mask = F.interpolate(mask.unsqueeze(1),
                             size=(H, W), mode='nearest')
        x_masked = x * mask

        encoded = self.encoder(x_masked, output_hidden_states=True,
                               return_dict=True).hidden_states

        # reconstructed = self.reconstruction_head(encoded.mean(dim=1))
        reconstructed = self.reconstruction_head(encoded).requires_grad_(True)
        reconstructed = F.interpolate(reconstructed,
                                      size=(H, W), mode='nearest')

        # Pool along dim 1
        reconstructed = torch.argmax(reconstructed, dim=1).unsqueeze(1).float()

        return reconstructed, mask


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
    config.num_labels = 2
    config.id2label = {0: 'negative', 1: 'positive'}
    config.label2label = {'negative': 0, 'positive': 1}

    model = AttentionMaskingMIM(config)

    loss_fn = MSELoss()

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
        total_contrastive_loss = []
        total_reconstruction_loss = []

        # Iterate over the dataset
        for step, data in enumerate(tqdm(dataloader)):

            x = data['images'].to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                reconstructed_mask, mask = model(x)
                if torch.isnan(reconstructed_mask).any():
                    print(f"NaN generated in reconstructed_mask")
                if torch.isnan(mask).any():
                    print(f"NaN generated in mask")

                loss_total  = loss_fn(reconstructed_mask,mask)
                if torch.isnan(loss_total).any():
                    print(f"NaN generated in loss function")


            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.step()

            # 6. Logging and Tracking
            total_epoch_loss += [loss_total.item()]


        if scheduler is not None:
            scheduler.step()

        avg_epoch_loss = np.mean(total_epoch_loss)
        avg_contrastive_loss = np.mean(total_contrastive_loss)
        avg_reconstruction_loss = np.mean(total_reconstruction_loss)

        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
                    f"Average Contrastive Loss: {avg_contrastive_loss:.4f}, "
                    f"Average Reconstruction Loss: {avg_reconstruction_loss:.4f}")

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
    parser.add_argument("-p", "--prefix", type=str, default='moco_msn', )
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

