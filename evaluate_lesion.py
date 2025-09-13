import glob
import logging
import os
import sys
from enum import Enum

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from segmenter.models import AugurSegformerSegmentation


class Colours(Enum):
    """
    Colours for contours, use BGR format for OpenCV
    """
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (255, 255, 0)
    red = (0, 0, 255)


IMAGE_PATH = "../polyp_data/Classica/images/val"
MASK_PATH = "../polyp_data/Classica/masks/val"
IMAGE_PATTERN = "*.png"
MASK_PATTERN = "*.png"
PRED_PATH = "../polyp_data/Classica/predictions/val"
PRED_TYPE = ".png"
OVERLAY_PATH = "../polyp_data/Classica/overlays/val"
OVERLAY_TYPE = ".png"

pretrained_model = "nvidia/segformer-b4-finetuned-ade-512-512"  # Huggingface backbone model
num_classes = 2     # Not_lesion = 0, Lesion = 1
model_checkpoint = "../segmenter/checkpoint/lighting_clean_model_lesion_segmentation_20250912_173722.pt"

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

    image_list, mask_list = load_data(logger)

    """
        Setup the model 
        """
    model = AugurSegformerSegmentation(pretrained_model=pretrained_model,
                                       num_classes=2)

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
    for image_file, mask_file in tqdm(zip(image_list, mask_list)):
        image_pil = Image.open(image_file).convert("RGB")
        mask_pil = Image.open(mask_file).convert("L")
        mask_size = mask_pil.size
        mask = np.array(mask_pil)

        resize_fn = transforms.Resize((mask_size[1], mask_size[0]),
                                      interpolation=transforms.InterpolationMode.BICUBIC)

        image = image_transform(image_pil).unsqueeze(0)  # mask: [1, H, W]

        logits = model(pixel_values=image)  # logits; [1, 2, H, W]

        pred_mask = F.softmax(logits, dim=1).argmax(dim=1)
        pred_mask_array = resize_fn(pred_mask).squeeze(0).cpu().numpy().astype(np.uint8)

        pred_file_name = os.path.splitext(os.path.basename(image_file))[0] + ".png"
        out_name = os.path.join(PRED_PATH, pred_file_name)

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
        out_name = os.path.join(OVERLAY_PATH, pred_file_name)
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

    pd.DataFrame(metrics).to_csv("evaluation_metrics.csv")


def load_data(logger):
    image_list = sorted(glob.glob(os.path.join(IMAGE_PATH, IMAGE_PATTERN)))
    mask_list = sorted(glob.glob(os.path.join(MASK_PATH, MASK_PATTERN)))
    if not image_list:
        msg = f"No images found in {IMAGE_PATH} with pattern {IMAGE_PATTERN}."
        logger.error(msg)
        raise ValueError(msg)
    if not mask_list:
        msg = f"No images found in {MASK_PATH} with pattern {MASK_PATTERN}."
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





