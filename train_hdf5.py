import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from segmenter.data import (get_num_samples_from_hdf5, HDF5ImageDataset)
from segmenter.models import AugurSegformerSegmentation
from segmenter.modules import HybridLoss
from segmenter.torch_utils import RunManager, CheckpointManager

classica_names = ['01.png', '02.png', '05.png', '05182023_171203.png', '1013468013.png',
                  '1014670912-ass2.png', '1014670912.png', '10262023_142726.png', '1028851774.png',
                  '10302023_120402.png', '10302023_120901.png', '1040434567.png', '1061442455.png',
                  '1065516909.png', '107949968.png', '1084008061.png', '1105807382.png', '1120606001.png',
                  '1180608049.png', '12192023_193441.png', '1249860204.png', '1251528661.png',
                  '1321881235.png', '1370388080.png', '13845452.png', '1387851265.png', '1399586236-ass2.png',
                  '1399586236.png', '1474607936.png', '1508973689.png', '1521359056.png', '1567384095.png',
                  '16062023_1.png', '16090101.png', '16090102.png', '16090201.png', '16090301.png',
                  '16090401.png', '16090403.png', '16090501.png', '16090601.png', '16090602.png',
                  '16090701.png', '16090901.png', '16091001.png', '16091101.png', '16091201.png', '16091301.png',
                  '16091401.png', '16091402.png', '16091601.png', '16091801.png', '16091802.png', '16091901.png',
                  '16092001.png', '16092101.png', '16092102.png', '16092201.png', '16092301.png', '16092401.png',
                  '16092501.png', '16092601.png', '16092701.png', '16092801.png', '16093001.png', '16093101.png',
                  '16093201.png', '16093301.png', '16093401.png', '16093501.png', '16093601.png', '16093701.png',
                  '16093801.png', '1628446929.png', '170101.png', '170102.png', '170103.png', '170104.png',
                  '170108.png', '170109.png', '170110.png', '171245830.png', '1783345236.png', '1792998993.png',
                  '18078369.png', '1810823168.png', '1900025002.png', '1902791310.png', '2136784449.png',
                  '2189953076.png', '2250960458.png', '225602854.png', '2275449896.png', '228865380.png',
                  '2317004287.png', '2364792262.png', '2394014758.png', '2395724121.png', '2421910406.png',
                  '2441596969.png', '2443311490.png', '2472636703.png', '2473690117.png', '2486397623.png',
                  '2495375922.png', '252971098.png', '2529883711.png', '2535259739.png', '2567912066.png',
                  '2613069662.png', '2624989663.png', '2738202439.png', '2742950888.png', '280111445.png',
                  '2819213769.png', '2857256799.png', '2872439574.png', '2901309315.png', '2986309540.png',
                  '3014914240.png', '307631163.png', '3085924945.png', '3089001819.png', '3267962182.png',
                  '3272896089.png', '328309737.png', '3299431987.png', '3317328509.png', '3323937101.png',
                  '3360888566.png', '3366591515.png', '3367664281.png', '3372990809.png', '3430084361.png',
                  '3431381547.png', '3479712599-ass2.png', '3479712599.png', '3521412980.png', '35485434.png',
                  '3597727100.png', '3605407893.png', '3630237943.png', '3639539128.png', '3704101606.png',
                  '3727603818.png', '3734795729.png', '3744440603.png', '3748519529.png', '3755357963.png',
                  '3788048503.png', '3801559388.png', '3842700551.png', '3861726478.png', '3997056794.png',
                  '4006888708.png', '4038565078.png', '405183445.png', '4055844607.png', '4083104944.png',
                  '410309938.png', '4209107693.png', '4223363606.png', '4243043869.png', '4252141892.png',
                  '432381542.png', '44789482.png', '448602072.png', '452919406.png', '531885378.png',
                  '566507557.png', '596694107.png', '641224979.png', '672718769.png', '806365608.png',
                  '895818311.png', '920985152.png', '921260366.png', 'AMST_0001.png', 'AMST_0002.png',
                  'AMST_0019.png', 'Copy of Video One.png', 'Copy of Video Three.png', 'IBM_32.png',
                  'IBM_35.png', 'IBM_36.png', 'IBM_38.png', 'IBM_4.png', 'IBM_42.png', 'IBM_45.png',
                  'IBM_47.png', 'IBM_48.png', 'IBM_50.png', 'IBM_52.png', 'IBM_53.png', 'IBM_54.png',
                  'IBM_8.png', 'MMUH_DTIF_0058.png', 'MMUH_DTIF_0094.png', 'MMUH_DTIF_0095.png',
                  'MMUH_DTIF_0096.png', 'MMUH_DTIF_0100.png', 'MMUH_DTIF_0101.png', 'MMUH_DTIF_0103.png',
                  'REINERO__01092024_173849.png', 'Video TEM 22.4.png', 'WAT_DTIF_0005.png',
                  'WAT_DTIF_0007.png', 'WAT_DTIF_0008.png', 'WAT_DTIF_0010.png', 'ch1_video_01.png']


def main():
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger()
    home = Path.home()
    if not os.path.exists(os.path.join(home, "segmenter")):
        logger.warning("Creating 'segmenter' directory in $HOME directory.")
        os.makedirs(os.path.join(home, "segmenter"))
    if not os.path.exists(os.path.join(home, "segmenter/data")):
        logger.warning("Creating 'data' directory in $HOME/segmenter directory.")
        os.makedirs(os.path.join(home, "segmenter/data"))
    if os.path.isfile(os.path.join(os.path.join(home, "segmenter"),
                                                "lesion_params.json")):
        params_file = os.path.join(os.path.join(home, "segmenter"),
                                   "lesion_params.json")
    else:
        params_file = "lesion_params.json"


    # --- Load parameters from JSON file ---
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: {params_file} file not found. Please ensure it is in the same directory.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {params_file}: {e}")
        return


    pretrained_model = params['pretrained_model']
    checkpoint_path = params['checkpoints']['path']
    checkpoint_prefix = params['checkpoints']['prefix']
    checkpoint_patience = params['checkpoints']['patience']
    checkpoint_min_delta = params['checkpoints']['min_delta']
    if not os.path.isdir(checkpoint_path):
        logger.info(
            f"Checkpoint directory '{checkpoint_path}' not found. Saving to '[current directory]/checkpoints' instead.")
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints")

    """ Configure the run """
    run_params = params['run']
    test_split = run_params['test_split']
    num_classes = run_params['num_classes']
    batch_size = run_params['batch_size']
    num_workers = run_params['num_workers']
    n_augments = run_params['n_augments']
    image_size = tuple(run_params['image_size'])
    n_epochs = run_params['n_epochs']

    """ Optimiser settings """
    opt_params = params['optimizer']['params']
    learning_rate = opt_params['learning_rate']
    l2_decay_penalty = opt_params['l2_decay_penalty']

    """ Data settings """
    hdf5_path = params["datasets"]["hdf5_dir"]
    hdf5_files = [os.path.join(hdf5_path, _h) for _h in params["datasets"]["hdf5_files"]]
    logger.info(f"Loaded parameters: {params}")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using {device} device for model training.")

    """ Load datasets for test and training """
    # Now we correctly split the indices and create new datasets
    train_datasets = []
    test_datasets = []

    n_records = 0
    for hdf5_file in hdf5_files:
        len_hdf5 = get_num_samples_from_hdf5(hdf5_file)
        # Ensure reproducibility of training and test with a fixed seed
        # torch.manual_seed(42)
        # shuffled_indices = torch.randperm(len_hdf5)
        # test_indices = shuffled_indices[:int(len_hdf5 * test_split)]
        # train_indices = shuffled_indices[int(len_hdf5 * test_split):]
        with h5py.File(hdf5_file, 'r', swmr=True) as hdf:
            original_name_hdf = hdf['original_name']
            original_name_np = np.array([h.decode('utf-8') for h in original_name_hdf])
            train_indices = [idx for idx, name in enumerate(original_name_np) if name not in classica_names]
            test_indices = [idx for idx, name in enumerate(original_name_np) if name in classica_names]
            train_indices = torch.LongTensor(train_indices)
            test_indices = torch.LongTensor(test_indices)


        train_datasets.append(HDF5ImageDataset(
            hdf5_path=hdf5_file,
            indices=train_indices,
            is_train_split=True,
            image_size=image_size,
            n_augment=n_augments,
            light_control=False,
        ))


        test_datasets.append(HDF5ImageDataset(
            hdf5_path=hdf5_file,
            indices=test_indices,
            is_train_split=False,
            image_size=image_size,
            n_augment=0,
            light_control=False
        ))

        n_records += len_hdf5
    logger.info(f"Using {n_records} total records for training and testing.")

    final_train_dataset = train_datasets[0]
    final_test_dataset = test_datasets[0]

    train_loader = DataLoader(
        final_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        final_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    logger.info(f"Successfully loaded training and testing dataset for {n_records} records.")
    logger.info(f"Number of batches in the training DataLoader: {len(train_loader)}")
    logger.info(f"Number of batches in the test DataLoader: {len(test_loader)}")

    cp_manager = CheckpointManager(checkpoint_dir=checkpoint_path,
                                   prefix=checkpoint_prefix,
                                   patience=checkpoint_patience,
                                   min_delta=checkpoint_min_delta)
    """
    Setup the model 
    """
    model = AugurSegformerSegmentation(pretrained_model=pretrained_model,
    # model = AugurSegformerSegmentation(pretrained_model=None,
                                       num_classes=num_classes)

    latest_checkpoint = None
    try:
        # Get the list of saved checkpoints
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_path, checkpoint_prefix + "*.pt")))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            cp_manager.load(model, latest_checkpoint, device=device)
            logger.info(f"Loaded model checkpoint {latest_checkpoint}.")
        else:
            logger.info(f"No checkpoints were saved to load in {checkpoint_path}.")
    except FileNotFoundError as e:
        logger.info(f"Unable to load checkpoint {e}")

    model.to(device)
    loss_params = params['loss_function']
    loss_fn = HybridLoss(loss_params['params'])

    # Initial freeze all parameters of the model
    # logger.info("Freezing encoder layers...")
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    #
    # # Unfreeze the decoder head and segmentation head because we replaced these
    # logger.info("Unfreezing decoder and segmentation head...")
    # for param in model.base_model.decode_head.parameters():
    #     param.requires_grad = True

    # Only pass the parameters that require gradients to the optimizer
    optimizer = torch.optim.AdamW(
        # params=filter(lambda p: p.requires_grad, model.parameters()),
        params=model.parameters(),
        lr=learning_rate,
        # weight_decay=l2_decay_penalty  # L2 regularization to prevent large weights
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['scheduler']['T_max'])


    """
    Only use GradScaler if we have CUDA
    """
    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()



    trainer = RunManager(model,
                         optimizer,
                         criterion=loss_fn,
                         scaler=scaler,
                         scheduler=scheduler,
                         train_loader=train_loader,
                         eval_loader=test_loader,
                         save_preds=False,
                         save_preds_path="",
                         config_path=params_file
                         )
    train_params = {}
    eval_params = {}

    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")

        train_metrics = trainer.train(**train_params)
        val_metrics = trainer.evaluate(**eval_params)

        train_loss = train_metrics['loss']
        train_miou = train_metrics['iou']
        train_dice = train_metrics['dice']
        val_loss = val_metrics['loss']
        val_miou = val_metrics['iou']
        val_dice = val_metrics['dice']

        logger.info(f"Training Losses  : | Compound: {train_loss:.4f} | Dice: {train_dice:.4f} | IOU: {train_miou:.4f}")
        logger.info(f"Evaluation Losses: | Compound: {val_loss:.4f} | Dice: {val_dice:.4f} | IOU: {val_miou:.4f}")

        # scheduler.step(epoch + 1)

        stop_training = cp_manager.save(model, 1 - val_loss)
        if stop_training:
            logger.info(f"Training stopped early at epoch {epoch} with mIOU Score: {val_miou:.4f}")
            break


if __name__ == "__main__":
    # --- Logging Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    home_dir = Path.home()
    if not os.path.exists(os.path.join(home_dir, "segmenter")):
        os.makedirs(os.path.join(home_dir, "segmenter"))
    logfile = os.path.join(home_dir, "segmenter", f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        force=True,  # Resets any previous configuration - in Colab for example
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logfile)
        ]
    )
    logger = logging.getLogger(__name__)
    try:
        main()
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
