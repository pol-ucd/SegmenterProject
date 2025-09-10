import os

import h5py
from PIL import Image

hdf5_file = "/Users/polmacaonghusa/Library/CloudStorage/GoogleDrive-pol.macaonghusa@ucd.ie/My Drive/polyp_data/data/Classica.h5"
base_dir = "/Users/polmacaonghusa/Documents/Projects/polyp_data/Classica"
outdir_images = os.path.join(base_dir, "images/val")
outdir_masks = os.path.join(base_dir, "masks/val")

if __name__ == "__main__":
    hdf5_file = h5py.File(hdf5_file, 'r', swmr=True)

    images = hdf5_file['images']
    masks = hdf5_file['masks']
    orig_sizes = hdf5_file['orig_size']
    orig_names = hdf5_file['original_name']


    for image, mask, orig_size, orig_name in zip(images, masks, orig_sizes, orig_names):
        image_pil = Image.fromarray(image).convert('RGB').resize(orig_size,
                                                                 Image.Resampling.LANCZOS)
        mask_pil = Image.fromarray(mask[1]*255).convert('L').resize(orig_size,
                                                                    Image.Resampling.NEAREST)

        image_pil.save(os.path.join(outdir_images, orig_name.decode('utf-8')))
        mask_pil.save(os.path.join(outdir_masks, orig_name.decode('utf-8')))

    print(hdf5_file.keys())