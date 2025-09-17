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
from segmenter.data import preprocess_image_pipeline
from segmenter.models import AugurSegformerSegmentation

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
model_prefixes = [
                  # "base_tversky",
                  "base_smooth_intensity_tversky"
]
smooth_intensity = True

"""
Parameters for image preprocessing as model input 
"""
image_size=(512, 512)           # The backbone Huggingface model expects images of this size
mean = (0.485, 0.456, 0.406)    # Use standard image_net values for normalising
std = (0.229, 0.224, 0.225)
test_split = 1.0

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

    classica_names = sorted(glob.glob("../polyp_data/Classica/images/val/*.png"))
    classica_names = [os.path.basename(name) for name in classica_names]

    for hdf5_file in hdf5_files:
        len_hdf5 = get_num_samples_from_hdf5(hdf5_file)


        with h5py.File(hdf5_file, 'r', swmr=True) as hdf:
            original_name_hdf = hdf['original_name']
            original_name_np = np.array([h.decode('utf-8') for h in original_name_hdf])

            rng = np.random.default_rng(42)
            shuffled_indices = rng.permutation(len(classica_names))
            """ Only add test_split % to the training set """

            classica_test_indices = shuffled_indices[:int(len(classica_names) * test_split)]
            classica_names_np = np.array(classica_names)
            classica_test_names = classica_names_np[classica_test_indices]

            train_indices = [idx for idx, name in enumerate(original_name_np) if name not in classica_test_names]
            test_indices = [idx for idx, name in enumerate(original_name_np) if name in classica_test_names]

            train_names = original_name_np[train_indices]
            test_names = original_name_np[test_indices]

        return test_names, train_names

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
            image_original = image_pil.copy()
            if smooth_intensity:
                image_pil_np = np.array(image_pil)
                image_pil_np = preprocess_image_pipeline(image_pil_np)
                image_pil = Image.fromarray(image_pil_np)
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
            overlay_pil = detect_and_draw_contours(image_original,
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





