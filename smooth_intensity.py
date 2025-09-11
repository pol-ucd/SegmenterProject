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
import numbers
import os

import cv2
import numpy as np
from PIL import ImageFilter, Image


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


def detail_enhancement(img, d=9, sigma_color=75, sigma_space=75):
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


def preprocess_pipeline(img_path, out_path=None):
    # 1. Load BGR image
    img = cv2.imread(img_path)
    # 2. Flat-field correction
    flat = shading_correction(img, sigma=50)
    final = flat.copy()

    # # 3. Homomorphic filter - convert 3D -> 2D gray image first for DFT
    # flat = cv2.cvtColor(np.float32(flat), cv2.COLOR_BGR2GRAY).astype(np.float32)
    # homo = homomorphic_filter(flat, sigma=30)
    # final = homo.copy()
    #
    # 4. Single-Scale Retinex
    # homo = img.copy()
    # ssr = single_scale_retinex(homo, sigma=30)
    # final = ssr.copy()
    # 5. Detail enhancement
    # final = detail_enhancement(ssr, d=5, sigma_color=75, sigma_space=75)

    # 6. Save or return
    if out_path:
        cv2.imwrite(out_path, final)
    return final


if __name__ == "__main__":
    image_dir = "/Users/polmacaonghusa/Documents/Projects/polyp_data/Classica/images/val"
    out_dir = "/Users/polmacaonghusa/Documents/Projects/polyp_data/Classica/images/compensated"
    input_path = os.path.join(image_dir, "170103.png")
    output_path = os.path.join(out_dir, "170103_normalized_endoscope.png")
    result = preprocess_pipeline(input_path, output_path)
    # To visualize with OpenCV (BGR→RGB):
    # cv2.imshow("Normalized", result); cv2.waitKey(0)
