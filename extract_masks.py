"""
Run through the JSON files in the training and test folders to extract each mask.
Extracted masks are saved as B&W images into 'train_masks' and 'valid_masks'
folders .jpg files.
"""
import argparse
import json
import os
from glob import glob
from pycocotools import mask as maskUtils

import cv2
import numpy as np
from PIL import Image

TRAIN_DIR = 'data/Polyp Segmentation/train'
VALID_DIR = 'data/Polyp Segmentation/valid'
TRAIN_IMGS = os.path.join(TRAIN_DIR, '*.jpg')
VALID_IMGS = os.path.join(VALID_DIR, '*.jpg')
TRAIN_JSONS = os.path.join(TRAIN_DIR, '*.json')
VALID_JSONS = os.path.join(VALID_DIR, '*.json')

TRAIN_MASK_DIR = TRAIN_DIR + "_masks"
VALID_MASK_DIR = VALID_DIR + "_masks"

def extract_masks(path_imgs, path_jsons, path_mask_dir):
    masks = []
    if not os.path.exists(path_mask_dir):
        os.makedirs(path_mask_dir)

    for idx, img_path in enumerate(path_imgs):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        file_name = os.path.join(path_mask_dir,
                                 os.path.basename(img_path))

        with open(path_jsons[idx], 'r') as f:
            data = json.load(f)

            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in data['annotations']:
                mask = np.maximum(mask, maskUtils.decode(ann['segmentation']))*255
            masks += [mask]
            cv2.imwrite(filename=file_name, img=mask)

    return masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #TODO: Convert file paths into configuration options and/or command line parameters
    train_imgs = sorted(glob(TRAIN_IMGS))
    train_jsons = sorted(glob(TRAIN_JSONS))
    print(f"Found {len(train_imgs)} training images")

    test_imgs = sorted(glob(VALID_IMGS))
    test_jsons = sorted(glob(VALID_JSONS))
    print(f"Found {len(test_imgs)} validation images")

    train_masks = extract_masks(train_imgs, train_jsons, TRAIN_MASK_DIR)
    test_masks = extract_masks(test_imgs, test_jsons, VALID_MASK_DIR)

    print("All done!")


