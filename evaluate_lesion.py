import glob
import logging
import os
import sys
from enum import Enum

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from segmenter.data import get_num_samples_from_hdf5
from segmenter.models import AugurSegformerSegmentation


class Colours(Enum):
    """
    Colours for contours, use BGR format for OpenCV
    """
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (255, 255, 0)
    red = (0, 0, 255)

"""
Locate images and masks for glob.glob
"""
image_path = "../polyp_data/Classica/images/val"
mask_path = "../polyp_data/Classica/masks/val"
image_pattern = "*.png"
mask_pattern = "*.png"

"""
Setup models for inference
"""
pretrained_model = "nvidia/segformer-b4-finetuned-ade-512-512"
model_prefixes = ["10_classica_focal_only",
                  "10_classica_tversky_only"]

"""
Parameters for image preprocessing as model input 
"""
image_size=(512, 512)           # The backbone Huggingface model expects images of this size
mean = (0.485, 0.456, 0.406)    # Use standard image_net values for normalising
std = (0.229, 0.224, 0.225)

# Configure logging to write to a file and the console
LOG_FILE = 'classica_evaluation.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_classica_test_train_names():
    hdf5_files = ["../polyp_data/data/all_data.h5"]
    test_split = 0.1

    classica_names = sorted(glob.glob("../polyp_data/Classica/images/val/*.png"))
    classica_names = [os.path.basename(name) for name in classica_names]

    for hdf5_file in hdf5_files:
        len_hdf5 = get_num_samples_from_hdf5(hdf5_file)
        # Ensure reproducibility of training and test with a fixed seed
        torch.manual_seed(42)
        shuffled_indices = torch.randperm(len_hdf5)
        test_indices = shuffled_indices[:int(len_hdf5 * test_split)]
        train_indices = shuffled_indices[int(len_hdf5 * test_split):]

        with h5py.File(hdf5_file, 'r', swmr=True) as hdf:
            original_name_hdf = hdf['original_name']
            original_name_np = np.array([h.decode('utf-8') for h in original_name_hdf])
            train_names = original_name_np[train_indices]
            test_names = original_name_np[test_indices]

        test_classica = [name for name in test_names if name in classica_names]
        train_classica = [name for name in train_names if name in classica_names]
        return test_classica, train_classica

def check_path_and_clear(path: str, pattern: str):
    if os.path.exists(path):
        for f in glob.glob(pattern):
            os.remove(f)
    else:
        os.makedirs(path)


def detect_and_draw_contours(original_pil: Image.Image,
                             mask_pil: Image.Image,
                             contour_color=Colours.green.value,
                             thickness=2) -> Image.Image:
    """
    Detects contours from a binary mask and overlays them on the original image using PIL.

    Parameters:
        original_pil (PIL.Image): The original RGB image.
        mask_pil (PIL.Image): The binary mask image (mode 'L').
        contour_color (tuple): BGR color for the contour (default green).
        thickness (int): Thickness of the contour lines.

    Returns:
        PIL.Image: Image with contours drawn.
    """
    # Convert PIL images to NumPy arrays
    original_np = np.array(original_pil)
    mask_np = np.array(mask_pil)

    # Ensure mask is binary
    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Find contours using OpenCV
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image
    contour_overlay = original_np.copy()
    cv2.drawContours(contour_overlay, contours, -1, contour_color, thickness)

    # Convert back to PIL
    result_pil = Image.fromarray(contour_overlay)

    return result_pil


def main():
    logger = logging.getLogger(__name__)

    """ 
    Use CUDA if available. Also check for Apple MPS GPU for Mac users.
    """
    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else device_cuda)
    device = torch.device("cpu")

    image_list, mask_list = load_data(logger)
    for pred_prefix in model_prefixes:
        pred_path = f"../polyp_data/Classica/predictions/val_{pred_prefix}"
        pred_type = ".png"
        overlay_path = f"../polyp_data/Classica/overlays/val_{pred_prefix}"
        overlay_type = ".png"
        model_checkpoint = f"../segmenter/checkpoint/{pred_prefix}_lesion_segmentation.pt"
        check_path_and_clear(pred_path, pred_type)
        check_path_and_clear(overlay_path, overlay_type)

        """
            Setup the model 
        """
        model = AugurSegformerSegmentation(pretrained_model=pretrained_model,
                                           num_classes=2).to(device)

        try:
            logger.info(f"Loading model parameters from checkpoint {model_checkpoint}")
            # Load the state dictionary and apply it to the model
            model.load_state_dict(torch.load(model_checkpoint,
                                             # map_location=device,
                                             map_location=next(model.parameters()).device,
                                             weights_only=False))
            logger.info(f"Checkpoint loaded successfully from: {model_checkpoint}")
        except FileNotFoundError as e:
            msg = f"Unable to load checkpoint from {model_checkpoint}"
            logger.error(msg)
            raise ValueError(msg)

        image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        metrics = {"case": [],
                   "dice": [],
                   "iou": [],
                   "precision": [],
                   "recall": []}

        classica_test_names, _ = get_classica_test_train_names()

        for image_file, mask_file in tqdm(zip(image_list, mask_list)):
            if os.path.basename(image_file) not in classica_test_names:
                continue

            image_pil = Image.open(image_file).convert("RGB")
            mask_pil = Image.open(mask_file).convert("L")
            mask_size = mask_pil.size
            mask = np.array(mask_pil)

            resize_fn = transforms.Resize((mask_size[1], mask_size[0]),
                                          interpolation=transforms.InterpolationMode.BICUBIC)

            image = image_transform(image_pil).unsqueeze(0)  # mask: [1, H, W]

            logits = model(pixel_values=image.to(device))  # logits; [1, 2, H, W]

            pred_mask = F.softmax(logits,
                                  dim=1).argmax(dim=1)

            pred_mask_array = resize_fn(pred_mask).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            pred_file_name = os.path.splitext(os.path.basename(image_file))[0] + ".png"
            out_name = os.path.join(pred_path, pred_file_name)

            # denoised_pred_mask = Image.fromarray(pred_mask_array*255).filter(ImageFilter.ModeFilter(size = 3))
            denoised_pred_mask = Image.fromarray(pred_mask_array * 255)
            denoised_pred_mask.save(out_name)

            # Overlay original mask (green) and predicted mask (yellow)
            overlay_pil = detect_and_draw_contours(image_pil,
                                                   mask_pil,
                                                   contour_color=Colours.green.value,
                                                   thickness=3)

            overlay_pil = detect_and_draw_contours(overlay_pil,
                                                   denoised_pred_mask,
                                                   contour_color=Colours.yellow.value,
                                                   thickness=3)

            # Save or show result
            out_name = os.path.join(overlay_path, pred_file_name)
            overlay_pil.save(out_name)

            """ Now perform all the metrics and save results """
            logger.info(f"Processing: {pred_file_name}")

            """ Flatten everything so we can use regular SKLearn metrics"""
            predicted = np.array(denoised_pred_mask).reshape(-1)
            expected = np.array(mask).reshape(-1)
            metrics['case'].append(pred_file_name)
            metrics['dice'].append(f1_score(expected, predicted, average='macro'))
            metrics['iou'].append(jaccard_score(expected, predicted, average='macro'))
            metrics['precision'].append(precision_score(expected, predicted, average='macro'))
            metrics['recall'].append(recall_score(expected, predicted, average='macro'))

        pd.DataFrame(metrics).to_csv(f"{pred_prefix}_evaluation_metrics.csv")


def load_data(logger):
    image_list = sorted(glob.glob(os.path.join(image_path, image_pattern)))
    mask_list = sorted(glob.glob(os.path.join(mask_path, mask_pattern)))
    if not image_list:
        msg = f"No images found in {image_path} with pattern {image_pattern}."
        logger.error(msg)
        raise ValueError(msg)
    if not mask_list:
        msg = f"No images found in {mask_path} with pattern {mask_pattern}."
        logger.error(msg)
        raise ValueError(msg)
    logger.info(f"Found {len(image_list)} images and {len(mask_list)} masks.")
    if len(image_list) != len(mask_list):
        mask_names = [os.path.basename(m) for m in mask_list]
        image_names = [os.path.basename(i) for i in image_list]
        missing_masks = list(set(image_names).difference(mask_names))
        missing_images = list(set(mask_names).difference(image_names))
        if len(missing_masks) > 0:
            logger.error(f"Missing {len(missing_masks)} masks: : {', '.join(missing_masks)}")
        if len(missing_images) > 0:
            logger.error(f"Missing {len(missing_images)} images: : {', '.join(missing_images)}")
        raise ValueError(f"There are missing images and/or masks. Please check the logs.")
    return image_list, mask_list


if __name__=="__main__":
    logger = logging.getLogger(__name__)

    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down gracefully.")
        sys.exit(0)
    finally:
        # ensure log handlers are flushed.
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.info("Logger handlers flushed and closed. Exiting now.")





