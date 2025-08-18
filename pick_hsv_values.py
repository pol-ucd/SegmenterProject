import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

IMG_PATH = "data/Classica/Initial_frames_v2/annotated_ann_0"
IMG_TYPE = "*.png"


if __name__ == "__main__":
    all_images = sorted(glob(os.path.join(IMG_PATH, IMG_TYPE)))
    print(f"Found {len(all_images)} images in {IMG_PATH} matching {IMG_TYPE}")

    for img in tqdm(all_images):
        base_name = os.path.splitext(os.path.basename(img))[0][:-6]
        image = cv2.imread(img)
        h, w = image.shape[:2]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Callback to print HSV value on mouse click
        def show_hsv(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pixel = hsv[y, x]
                print(f"HSV at ({x},{y}): {pixel}")

        cv2.namedWindow('Original')
        cv2.setMouseCallback('Original', show_hsv)

        # Create trackbars for HSV thresholds
        def nothing(x):
            pass

        cv2.namedWindow('Mask Controls')
        cv2.createTrackbar('H Min', 'Mask Controls', 0, 179, nothing)
        cv2.createTrackbar('H Max', 'Mask Controls', 179, 179, nothing)
        cv2.createTrackbar('S Min', 'Mask Controls', 255, 255, nothing)
        cv2.createTrackbar('S Max', 'Mask Controls', 255, 255, nothing)
        cv2.createTrackbar('V Min', 'Mask Controls', 250, 255, nothing)
        cv2.createTrackbar('V Max', 'Mask Controls', 255, 255, nothing)

        while True:
            # Get current trackbar positions
            h_min = cv2.getTrackbarPos('H Min', 'Mask Controls')
            h_max = cv2.getTrackbarPos('H Max', 'Mask Controls')
            s_min = cv2.getTrackbarPos('S Min', 'Mask Controls')
            s_max = cv2.getTrackbarPos('S Max', 'Mask Controls')
            v_min = cv2.getTrackbarPos('V Min', 'Mask Controls')
            v_max = cv2.getTrackbarPos('V Max', 'Mask Controls')

            # Create mask based on HSV range
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)


            # Show original and mask
            cv2.imshow('Original', image)
            cv2.imshow('Mask', mask)

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
