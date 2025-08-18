import os
import random
from glob import glob

import cv2
import pytesseract
import numpy as np
from tqdm import tqdm

IMG_PATH = "data/Classica/Initial_frames_v2/annotated_ann_0"
IMG_TYPE = "*.png"
MASK_OUT_PATH = "data/Classica/masks/"
MASK_TYPE = ".png"


def sort_contour(contour):

    x = contour.squeeze(axis=1)[:,0]
    y = contour.squeeze(axis=1)[:,1]
    sorted_contour = np.zeros_like(contour)

    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]
    sorted_contour[:,:,0] = x_sorted[:,np.newaxis]
    sorted_contour[:,:,1] = y_sorted[:,np.newaxis]
    return sorted_contour


def close_contour(contours):
    """
    Very simple approach - just set the endpoint = startpoint
    :param contour:
    :return: joined contour
    """
    try:
        points = np.vstack(contours)
        contour = sort_contour(points)
    except ValueError:
        contour = None
    # hull = cv2.convexHull(points)
    return contour


def find_lesion(bgr_image: np.ndarray) -> np.array:
    # Add border to help close edge-touching curves
    border_size = 20

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # lower_red = np.array([0, 0, 230])  # Lower bound for red in BGR
    # upper_red = np.array([77, 150, 255])  # Upper bound for red in BGR

    lower_vivid_red = np.array([0, 255, 255])
    upper_vivid_red = np.array([0, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower_vivid_red, upper_vivid_red)

    # Yellow
    lower_yellow = np.array([20, 255, 255])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Green
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Cyan
    # lower_cyan = np.array([85, 100, 100])
    # upper_cyan = np.array([95, 255, 255])
    lower_cyan = np.array([85, 255, 255])
    upper_cyan = np.array([95, 255, 255])


    cyan_mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)


    """
        Numeric Text may appear in BLACK and overlap the region contours
    """
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 220, 80])
    text_mask = cv2.inRange(hsv_image, lower_black, upper_black)

    """
    Start with a catch-all mask to find all contours and text
    """
    lower_all = np.array([0, 255, 250])
    upper_all = np.array([179, 255, 255])
    all_mask = cv2.inRange(hsv_image, lower_all, upper_all)
    # all_mask = cv2.bitwise_or(all_mask, text_mask)

    """
    Lesions are marked in RED (Cancer) or CYAN (BENIGN) 
    """
    # lesion_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, cyan_mask),
    #                              text_mask)
    lesion_mask = cv2.bitwise_or(red_mask, cyan_mask)

    """
    Non-lesion areas (healthy tissue + tools) are GREEN or YELLOW 
    """
    non_lesion_mask = cv2.bitwise_or(green_mask, yellow_mask)

    # full_mask = cv2.bitwise_xor(non_lesion_mask, all_mask)
    full_mask = lesion_mask

    padded_mask = cv2.copyMakeBorder(full_mask,
                                     border_size,
                                     border_size,
                                     border_size,
                                     border_size,
                                     cv2.BORDER_CONSTANT,
                                     value=0)

    # Find contours in padded mask
    contours, _ = cv2.findContours(padded_mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    print(f"Found {len(contours)} contours")

    # Create blank mask for padded image
    mask_padded = np.zeros_like(padded_mask, dtype=np.uint8)

    for contour in contours:

        largest_contour = close_contour([contour])

        if largest_contour is not None:
            cv2.drawContours(mask_padded,
                             contours=[largest_contour],
                             contourIdx=-1,
                             color=1,
                             thickness=cv2.FILLED)



            # start = largest_contour[0][0]
            # end = largest_contour[-1][0]
            # gap = np.linalg.norm(start - end)
            #
            # if gap > 1:
            #     print(f"Contour has a break of {gap:.2f} pixels")

            kernel = np.ones((5, 5), np.uint8)
            mask_padded = cv2.morphologyEx(mask_padded, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Crop back to original size
    mask_final = mask_padded[border_size:-border_size, border_size:-border_size]

    return mask_final


if __name__ == '__main__':

    all_images = sorted(glob(os.path.join(IMG_PATH, IMG_TYPE)))
    print(f"Found {len(all_images)} images in {IMG_PATH} matching {IMG_TYPE}")

    for image in tqdm(all_images):
        base_name = os.path.splitext(os.path.basename(image))[0][:-6]
        image = cv2.imread(image)
        h, w = image.shape[:2]

        mask_lesion = find_lesion(image)

        # Fill the largest contour (assuming it's the annotated curve)
        # if contours:
        #     largest_contour = max(contours, key=cv2.contourArea)
        #     cv2.drawContours(mask,
        #                      [largest_contour],
        #                      -1,
        #                      color=1,
        #                      thickness=cv2.FILLED)

        # output_file = os.path.join(MASK_OUT_PATH, base_name + ".png")
        # cv2.imwrite(output_file, mask_lesion * 255)
        while True:
            show_img = np.hstack([image,
                                  np.stack([mask_lesion * 255] * image.shape[-1], axis=-1)])
            cv2.imshow(base_name, show_img)  # Multiply by 255 to make it visible
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
