# HeatMap
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, zoom
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# TODO: Fix to also include Apple and to set some defaults
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
config.num_labels = 1
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512",
                                                         config=config,
                                                         ignore_mismatched_sizes=True
                                                         )
model.classifier = nn.Sequential(
    nn.Conv2d(config.hidden_sizes[-1], 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 1, kernel_size=1)
)
model.load_state_dict(torch.load("/home/user/Documents/hanija/fracture_model.pt", map_location=device))
model.to(device)
model.eval()
transform = Compose([Resize(512, 512), Normalize(), ToTensorV2()])
nRows, nCols = 6, 6
duration_after_detection = 30  # seconds
fps = 60

try:
    user_input_time = float(input("Enter start time (in seconds) for white light detection: "))
except ValueError:
    print("Invalid input. Using default of 5.0 seconds.")
    user_input_time = 5.0


def calculateMeanIntensities(frame, nRows, nCols):
    J3 = np.split(frame, nCols, axis=1)
    J3 = np.array(J3)
    J3 = np.mean(J3, 2, keepdims=True)
    J3 = np.squeeze(J3)
    J3 = np.ravel(J3)
    J3 = np.reshape(J3, (nCols, nRows, -1))
    J3 = np.transpose(J3, (0, 2, 1))
    J3 = np.mean(J3, 1, keepdims=True)
    J3 = np.transpose(np.squeeze(J3), (1, 0))
    return J3


def interpolateFeature(fGrid, mask):
    x, y = np.meshgrid(np.arange(fGrid.shape[1]), np.arange(fGrid.shape[0]))
    points = np.column_stack((x[mask], y[mask]))
    values = fGrid[mask]
    grid_x, grid_y = np.mgrid[0:fGrid.shape[0], 0:fGrid.shape[1]]
    return griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.nan)


def features(time, data, nRows, nCols, speckleMask):
    m, n = data.shape
    time = np.array(time)

    dIramp = 0.5
    dTramp = 45
    kramp = np.ones((m, 1), dtype=np.int64)
    kpeak = np.ones((m, 1), dtype=np.int64)
    khalf = np.ones((m, 1), dtype=np.int64)
    Ibase = np.full((m, 1), np.nan)
    Ipeak = np.full((m, 1), np.nan)

    for i, row in enumerate(data):
        diff = (row - row[0]) > dIramp
        if np.any(diff):
            krmp = np.argmax(diff)
            kramp[i] = krmp
            Ibase[i] = row[krmp]
            idx = (time <= (time[krmp] + dTramp))
        if np.any(idx):
            kpeak[i] = np.argmax(row[idx])
            Ipeak[i] = row[kpeak[i]]
            Ihalf = Ibase[i] + (Ipeak[i] - Ibase[i]) / 2
            rng = np.arange(kramp[i], kpeak[i] + 1)
        if len(rng) > 0:
            khalf[i] = np.argmin(np.abs(row[rng] - Ihalf)) + rng[0] - 1

        tRamp = time[kramp.flatten()]
        dThalf = time[khalf.flatten()] - time[kramp.flatten()]
        dTpeak = time[kpeak.flatten()] - time[kramp.flatten()]
        ingress = np.divide(Ipeak.flatten(), dTpeak.flatten(),
                            out=np.full_like(Ipeak.flatten(), np.nan),
                            where=dTpeak.flatten() != 0)

        timeToPeak = interpolateFeature(np.transpose(np.reshape(dTpeak, (nCols, nRows)), (1, 0)), speckleMask)
        timeToHalf = interpolateFeature(np.transpose(np.reshape(dThalf, (nCols, nRows)), (1, 0)), speckleMask)
        maxIntensity = interpolateFeature(np.transpose(np.reshape(Ipeak.flatten(), (nCols, nRows)), (1, 0)),
                                          speckleMask)
        maxIngress = interpolateFeature(np.transpose(np.reshape(ingress, (nCols, nRows)), (1, 0)), speckleMask)

        featuresTable = pd.DataFrame(np.stack((tRamp, dThalf, dTpeak,
                                               Ibase.flatten(), Ipeak.flatten()), axis=1),
                                     columns=["tRamp", "dThalf", "dTpeak", "Ibase", "Ipeak"])
        return featuresTable, timeToPeak, timeToHalf, maxIntensity, maxIngress


def smooth_and_zoom(data, sigma=0.5, zoom_factor=2.0):
    smoothed = gaussian_filter(data, sigma=sigma)
    return zoom(smoothed, zoom=zoom_factor, order=3)


def plot_feature_maps(timeToPeak, timeToHalf, maxIntensity, maxIngress):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    cmap = 'hot'
    sigma = 0.5
    zoom_factor = 2

    smoothed = [smooth_and_zoom(x, sigma, zoom_factor) for x in [maxIntensity, maxIngress, timeToHalf, timeToPeak]]
    titles = ["Maximum Intensity", "Maximum Ingress", "Time to 50% Max Intensity", "Time to Peak"]

    for ax, data, title in zip(axs.ravel(), smoothed, titles):
        im = ax.imshow(data, cmap=cmap, interpolation='bicubic', origin='upper')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = "/home/user/Documents/hanija/feature_maps.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Feature map plot saved to {save_path}")


cap = cv2.VideoCapture("/home/user/Documents/hanija/sample.mov")
frame_count = 0
detected_time_sec = None
downscaled_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    if detected_time_sec is None and current_time_sec >= user_input_time:
        detected_time_sec = user_input_time
        print(f"Manual start time reached at {current_time_sec:.2f}s (frame {frame_count})")

    if detected_time_sec is not None:
        elapsed = current_time_sec - detected_time_sec
        if elapsed > duration_after_detection:
            print("30 seconds of data collected. Done.")
            break

    h, w, _ = frame.shape
    left_w = w // 4
    top = h // 3
    bottom = 2 * h // 3
    roi = frame[top:bottom, 0:left_w]

    augmented = transform(image=roi)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(pixel_values=input_tensor)
        pred_mask = torch.sigmoid(output.logits)[0, 0].cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (left_w, bottom - top))

        overlay = roi.copy()
        colored_mask = np.zeros_like(overlay)
        colored_mask[:, :, 1] = pred_mask
        blended = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
        small_overlay = cv2.resize(blended, (96, 72), interpolation=cv2.INTER_AREA)

        downscaled_frames.append(small_overlay)

cap.release()
gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in downscaled_frames]
frame_stack = np.stack(gray_frames)
Iregion = []
for f in frame_stack:
    mean_intensities = calculateMeanIntensities(f, nRows, nCols)
    Iregion.append(mean_intensities)
Iregion = np.array(Iregion)
Iregion = np.transpose(Iregion, (1, 2, 0))
Iregion = np.reshape(Iregion, (nRows * nCols, -1))
times = np.linspace(0, duration_after_detection, Iregion.shape[1])
speckleMask = np.ones((nRows, nCols), dtype=bool)
for i in range(min(5, Iregion.shape[0])):
    plt.plot(times, Iregion[i], label=f"Region {i}")
    plt.title("Sample Region Intensities Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid()
    plt.show()

featuresTable, timeToPeak, timeToHalf, maxIntensity, maxIngress = features(times, Iregion, nRows, nCols, speckleMask)
print(featuresTable)
plot_feature_maps(timeToPeak, timeToHalf, maxIntensity, maxIngress)
output_path = "/home/user/Documents/hanija/overlay_output.mp4"
height, width, _ = downscaled_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
for frame in downscaled_frames:
    out.write(frame)

out.release()
print(f"Overlay video saved to: {output_path}")
