"""
Chained Illumination-Normalization Pipeline
-------------------------------------------
Below is a self-contained Python example that loads an endoscopic image and applies, in sequence:
1. Flat-field (shading) correction
2. Homomorphic filtering
3. Single-Scale Retinex
4. Base/detail decomposition via bilateral filtering

Each step evens out low-frequency lighting variations while preserving tissue texture.

How It Works
- shading_correction removes smooth lighting gradients.
- homomorphic_filter treats illumination as multiplicative noise and high-passes in log space.
- single_scale_retinex further flattens color channels by subtracting a Gaussian-blurred version
    in log domain.
- detail_enhancement uses a bilateral filter to split the image into base/detail,
    then recombines to boost fine tissue texture.

Adjust the σ parameters and blending weights to match your surgical camera’s characteristics.
"""
import glob
import numbers
import os

import cv2
import numpy as np
from PIL import ImageFilter, Image
from skimage import exposure


class GaussianSmoothing(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))


def shading_correction(img, sigma=50, eps=1e-6):
    """Flat-field shading correction."""
    # float_img = img.astype(np.float32) + eps
    # illum = cv2.GaussianBlur(float_img, (0, 0), sigma)
    illum = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred = GaussianSmoothing(radius=sigma)(Image.fromarray(illum))

    illum = cv2.cvtColor(np.array(blurred), cv2.COLOR_RGB2BGR).astype(np.float32) + eps

    float_img = img.astype(np.float32) + eps
    corrected = (float_img/ illum) * np.mean(illum)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def homomorphic_filter(img, sigma=30, high_gain=1.5, low_gain=0.5):
    """Log-domain high-pass (homomorphic) filter."""
    img_f = img.astype(np.float32) + 1.0
    log_img = np.log(cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY))
    dft = cv2.dft(log_img,
                  flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # build Butterworth-style high-pass mask
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    D2 = (Y - crow) ** 2 + (X - ccol) ** 2
    H = (high_gain - low_gain) * (1 - np.exp(-D2 / (2 * (sigma ** 2)))) + low_gain
    H = H[..., np.newaxis]  # shape (rows,cols,1)

    # apply to each channel
    filtered = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(f_ishift)
    img_exp = np.expm1(img_back[..., 0])
    img_exp = cv2.cvtColor(img_exp, cv2.COLOR_GRAY2BGR)
    return np.clip(img_exp, 0, 255).astype(np.uint8)


def single_scale_retinex(img, sigma=30):
    """Single-Scale Retinex (SSR) on each channel separately."""
    img_f = img.astype(np.float32) + 1.0
    ssr = np.zeros_like(img_f)
    if len(img.shape) == 2:
        blur = cv2.GaussianBlur(img_f[:, :], (0, 0), sigma)
        ssr[:, :] = np.log(img_f[:, :]) - np.log(blur)
    else:
        for c in range(img_f.shape[-1]):
            blur = cv2.GaussianBlur(img_f[:, :, c], (0, 0), sigma)
            ssr[:, :, c] = np.log(img_f[:, :, c]) - np.log(blur)
    # normalize to [0,255]
    ssr = (ssr - ssr.min()) / (ssr.max() - ssr.min()) * 255.0
    return ssr.astype(np.uint8)


def detail_enhancement(img, d=2, sigma_color=75, sigma_space=75):
    """
    Edge-preserving decomposition:
    base = bilateralFilter(img), detail = img/base, recombine.
    """
    img_f = img.astype(np.float32)
    base = cv2.bilateralFilter(img_f, d, sigma_color, sigma_space)
    # avoid divide-by-zero
    detail = cv2.divide(img_f, base + 1e-6, scale=255.0)
    # blend detail and base layers
    result = cv2.addWeighted(detail, 0.6, base, 0.4, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def find_tools(image: np.ndarray) -> np.ndarray:
    # Step 1: Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Step 2: Initial Mask via Otsu Thresholding
    _, initial_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Fluid Detection using HSV Color Segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define fluid color range (example range, may need tuning)
    lower_fluid_color = np.array([0, 0, 50])
    upper_fluid_color = np.array([180, 50, 255])
    fluid_mask = cv2.inRange(hsv, lower_fluid_color, upper_fluid_color)

    # Remove fluid regions from initial mask
    tool_mask = cv2.bitwise_and(initial_mask, cv2.bitwise_not(fluid_mask))

    # Step 4: Morphological Refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(tool_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    return final_mask

def match_histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match src histogram to ref."""
    matched = exposure.match_histograms(src, ref, multichannel=True)
    return (matched * 255).astype(np.uint8)

def gray_world(img: np.ndarray) -> np.ndarray:
    """Simple Gray World color constancy."""
    # Compute average per channel
    avg_b, avg_g, avg_r = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    # Scale     each channel
    img[:, :, 0] = np.clip(img[:, :, 0] * (avg_gray / avg_b), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (avg_gray / avg_g), 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return img.astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit=2.0, tile_grid_size=(8,8)) -> np.ndarray:
    """Apply CLAHE on the L-channel of LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    lab = cv2.merge([cl, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpening_kernel(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    if kernel is None:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
    sharpened_cv2_kernel = cv2.filter2D(img, -1, kernel)
    return sharpened_cv2_kernel.astype(np.uint8)


def preprocess_pipeline(img_path, out_path=None):
    img = cv2.imread(img_path)
    img_sharp = sharpening_kernel(img)
    img_gw = gray_world(img_sharp)
    # img_ic = shading_correction(img_gw)
    # final = apply_clahe(img_gw, clip_limit=5.0, tile_grid_size=(8,8))
    final =  img_gw.copy()
    out_img = np.concatenate([img, final], axis=1)
    if out_path:
        cv2.imwrite(out_path, out_img)
    return final


if __name__ == "__main__":
    image_dir = "/Users/polmacaonghusa/Documents/Projects/polyp_data/Classica/images/val"
    out_dir = "/Users/polmacaonghusa/Documents/Projects/polyp_data/Classica/images/compensated"
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    for image in images:
        basename = os.path.basename(image)
        input_path = os.path.join(image_dir, basename)
        output_path = os.path.join(out_dir, basename)
        result = preprocess_pipeline(input_path, output_path)

