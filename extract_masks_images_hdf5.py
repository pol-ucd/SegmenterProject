import os

import h5py
from PIL import Image
from tqdm import tqdm

hdf5_file = "/home/pol/segmenter/data/dresden_preprocessed.h5"
base_dir = "/home/pol/dresden_data"
outdir_images = os.path.join(base_dir, "images")
outdir_masks = os.path.join(base_dir, "masks")

if __name__ == "__main__":
    if not os.path.exists(outdir_images):
        os.makedirs(outdir_images)
    if not os.path.exists(outdir_masks):
        os.makedirs(outdir_masks)

    hdf5_file = h5py.File(hdf5_file, 'r', swmr=True)

    images = hdf5_file['images']
    masks = hdf5_file['masks']

    for idx, (image, mask) in enumerate(tqdm(zip(images, masks))):
        image_pil = Image.fromarray(image).convert('RGB')
        mask_pil = Image.fromarray(mask[1] * 255).convert('L')

        name = f"Dresden_{idx:06d}.jpg"

        image_pil.save(os.path.join(outdir_images, name))
        mask_pil.save(os.path.join(outdir_masks, name))
