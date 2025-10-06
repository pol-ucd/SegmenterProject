"""
Common MSN pre-training driver.

Reads command line parameter to select the MSN model to pretrain. Everything else is
common so we have measurement stability between models to allow comparison.

"""
import argparse
import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from msn_simCLR import finetune_percent
from segmenter.core import Config, freeze_seed
from segmenter.torch_utils import get_module_class
from segmenter.utils import load_data
from segmenter.utils.msn import MSNDataHandler

MSN_MAP = {'moco': {'name': 'segmenter.models.MoCoSiameseNetwork',
                    'params': {"pretrained_model": "nvidia/segformer-b4-finetuned-ade-512-512"}},
           'clr':  {'name': 'segmenter.models.SimCLRSegFormer',
                    'params': {"pretrained_model": "nvidia/segformer-b4-finetuned-ade-512-512"}},
           'sim':  {'name': 'segmenter.models.SimSiamSegFormer',
                    'params': {"pretrained_model": "nvidia/segformer-b4-finetuned-ade-512-512"}}
           }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MSNRunManager:
    def __init__(self, msn_model_name: str, config: Config,
                 device: torch.device = None,
                 set_seed:bool = False, ):
        """

        :param msn_model_name: The short name of the MSN model, must be a key in MSN_MAP
        :param config: The current configuration as a dictionary from Config.to_dict()
        :param set_seed: If True freeze seeds for Python, Numpy and Torch
        """
        self.msn_model_name = msn_model_name
        self.config = config
        self.device = device if device is not None else torch.device('cpu')
        try:
            self.random_seed = config['run']['random_seed']
        except KeyError:
            self.random_seed = 42   # Default value if not otherwise set
        if set_seed:
            freeze_seed(self.random_seed)

        self.model = self.msn_load(msn_model_name).to(device=self.device)

        """ Add model parameters for the optimizer """
        self.config['optimizer']['params']['params'] = self.model.parameters()
        self.optimizer = self._load("optimizer")

        """ Add optimizer instance for the scheduler """
        self.config['scheduler']['params']['optimizer'] = self.optimizer
        self.scheduler = self._load("scheduler")

        self.scaler = None
        if torch.cuda.is_available():
            self.scaler =  self._load("scaler")

        data_handler = MSNDataHandler(self.config)
        self.pretrain_loader = data_handler.load_pretrain_dataset()
        self.num_epochs = config['run']['num_epochs']

    def pretrain(self):
        logger = logging.getLogger(__name__)
        self.model.train()
        total_loss = []

        best_loss = float('inf')
        min_delta = 0.00001
        boredom = 0
        max_boredom = 10
        best_model = None
        for epoch in range(self.num_epochs):
            for batch_images in tqdm(self.pretrain_loader):
                # Dataloader yields raw images [B, C, H, W] as the 'images' value
                x = batch_images['images'].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass through the Siamese network
                z_anchor, z_positive = self.model(x)

                # Calculate InfoNCE Loss
                loss = loss_fn(z_anchor, z_positive)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                total_loss += [loss.item()]

            avg_loss = total_loss[-1] / len(self.pretrain_loader)
            logger.info(f"Pretraining Epoch [{epoch + 1}/{self.num_epochs}], Average Loss: {avg_loss:.4f}")
            if avg_loss + min_delta < best_loss:
                best_loss = avg_loss
                boredom = 0
                logger.info("Saving best snapshot `msn_model.online_encoder` state dict for fine-tuning.")
                try:
                    best_model = self.model.online_encoder.state_dict()
                    torch.save(best_model,
                               f'../segmenter/checkpoint/{self.msn_model_name}_segformer_pretrained.pth')
                except Exception as e:
                    logger.error(f"Pretraining failed to save `{self.msn_model_name}_segformer_pretrained.pth`: {e}")

            else:
                boredom += 1
            if boredom > max_boredom:
                logger.info(f"No improvement after {boredom} epochs, terminating")
                break

        return best_model  # Return the pre-trained encoder weights

    def msn_load(self, msn_model_name: str):
        """ Loading form the MSN_MAP - rather than the JSON profile """
        model_config = MSN_MAP[msn_model_name]
        return self._load_component(model_config['name'], **model_config['params'])

    def _load(self, component: str):
        """ Load from the JSON configuration """
        try:
            component_config = self.config[component]
        except KeyError:
            raise KeyError(f"{component} not found in configuration.")
        return self._load_component(component_config['name'], **component_config['params'])

    @staticmethod
    def _load_component(name, **params) -> torch.optim.Optimizer:
        _class_ = get_module_class(name)
        instance = _class_(**params)
        return instance

    def __repr__(self):
        return f"MSNRunManager('{self.msn_model_name}')"



def main(**kwargs):
    print(kwargs)
    config = kwargs['config']  # Passed as a dictionary
    logger = kwargs['logger']
    msn_model_name = kwargs['msn_model']

    logger.info(f"Starting pretraining run for MSN Model: {msn_model_name.upper()}")

    run = MSNRunManager(msn_model_name, config)
    logger.info(f"{run} loaded")

    run.pretrain()




if __name__ == "__main__":
    # --- Logging Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    home_dir = Path.home()
    cwd = Path.cwd()
    if not os.path.exists(os.path.join(home_dir, "segmenter")):
        os.makedirs(os.path.join(home_dir, "segmenter"))
    logfile = os.path.join(home_dir, "segmenter", f"training_{timestamp}.log")

    parser = argparse.ArgumentParser(description="Pre-training for MSN.")
    parser.add_argument("--model", help="Model to pretrain.",
                        choices=list(MSN_MAP.keys()), type=str,
                        default=MSN_MAP['moco'])
    parser.add_argument("--logfile", help="Logfile name to user.",
                        type=str, default=logfile)
    parser.add_argument("--config", help="Configuration JSON or YAML file to use.",
                        required=False, type=str, default="config/msn_common.json")

    msn_args = parser.parse_args()
    msn_model = msn_args.model
    msn_logfile = msn_args.logfile if msn_args.logfile else logfile
    config_args = msn_args.config if msn_args.config else logfile
    msn_config = Config(config_args).as_dict()

    logging.basicConfig(
        level=logging.INFO,
        force=True,  # Resets any previous configuration - in Colab for example
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(msn_logfile)
        ]
    )
    logger = logging.getLogger(__name__)

    main_args = {'msn_model': msn_model,
                 'logger': logger,
                 'config': msn_config}

    try:
        main(**main_args)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down gracefully.")
        sys.exit(0)
    finally:
        # This block will always be executed, allowing you to clean up resources
        # ensure log handlers are flushed.
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.info("Logger handlers flushed and closed. Exiting now.")
