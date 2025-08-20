import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import cv2
from tqdm import tqdm


def get_transform(crop_height, crop_width):
    return A.Compose([
        A.RandomCrop(height=crop_height, width=crop_width, p=0.5),
        A.Resize(height=512, width=512),  # Rescale back to original
        A.Perspective(scale=(0.05, 0.1), p=0.4),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.ElasticTransform(p=0.2),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
        A.GridDropout(ratio=0.5, p=0.3),
    ], additional_targets={"mask": "mask"})


def augment_single_image(args_tuple):
    fname, args, transform = args_tuple

    img_path = os.path.join(args.input_images, fname)
    mask_path = os.path.join(args.input_masks, fname)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    results = []
    for i in range(args.num_augments):
        augmented = transform(image=image, mask=mask)
        aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
        aug_mask = augmented["mask"]

        base_name = os.path.splitext(fname)[0]
        img_out_path = os.path.join(args.output_images, f"{base_name}_aug{i}.png")
        mask_out_path = os.path.join(args.output_masks, f"{base_name}_aug{i}.png")

        cv2.imwrite(img_out_path, aug_img)
        cv2.imwrite(mask_out_path, aug_mask)
        results.append((img_out_path, mask_out_path))

    return results

def augment_images(args):
    os.makedirs(args.output_images, exist_ok=True)
    os.makedirs(args.output_masks, exist_ok=True)

    image_filenames = sorted(os.listdir(args.input_images))
    transform = get_transform(args.crop_height, args.crop_width)

    # Prepare arguments for each image
    task_args = [(fname, args, transform) for fname in image_filenames]

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(augment_single_image, task_args), total=len(task_args), desc="Augmenting"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Albumentations CLI Augmentor with Multiprocessing")
    parser.add_argument("-ii", "--input-images", type=str, required=True, help="Path to input images")
    parser.add_argument("-im", "--input-masks", type=str, required=True, help="Path to input masks")
    parser.add_argument("-oi", "--output-images", type=str, required=True, help="Path to save augmented images")
    parser.add_argument("-om", "--output-masks", type=str, required=True, help="Path to save augmented masks")
    parser.add_argument("-n", "--num-augments", type=int, default=5, help="Number of augmentations per image")
    parser.add_argument("-ch", "--crop-height", type=int, default=256, help="Height for RandomCrop")
    parser.add_argument("-cw", "--crop-width", type=int, default=256, help="Width for RandomCrop")

    args = parser.parse_args()
    augment_images(args)
