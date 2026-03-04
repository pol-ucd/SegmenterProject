"""
Inference Script for AugurSegformerSegmentation
- Load model from checkpoint
- Run inference on images
- Save predictions as masks
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from segmenter.core import Config
from segmenter.models import AugurSegformerSegmentation
from segmenter.utils.metrics import get_device
from segmenter.utils.mps_patch import patch_segformer_for_mps


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(image_path, image_size=512):
    """Load and preprocess an image for inference."""
    img = Image.open(image_path).convert('RGB')
    
    original_size = img.size
    
    img = img.resize((image_size, image_size), Image.BILINEAR)
    img_array = np.array(img).astype(np.float32) / 255.0
    
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    
    return img_tensor.unsqueeze(0), original_size


def postprocess_mask(mask, original_size, threshold=0.5):
    """Postprocess the output mask."""
    mask = mask.squeeze().cpu().numpy()

    mask = (mask > threshold).astype(np.uint8) * 255

    mask_image = Image.fromarray(mask, mode='L')
    mask_image = mask_image.resize(original_size, Image.NEAREST)

    return mask_image


def predict_image(model, image_tensor, device, foreground_index=1):
    """Run inference on a single image."""
    model.eval()

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)

    if isinstance(output, tuple):
        output = output[0]

    probs = torch.softmax(output, dim=1)
    foreground_prob = probs[:, foreground_index, :, :]

    return foreground_prob


def main():
    parser = argparse.ArgumentParser(description='Inference with AugurSegformerSegmentation')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (default: config.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory or file')
    parser.add_argument('--image_size', type=int, default=None, help='Image size for model')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for inference')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Pretrained model')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--prediction_threshold', type=float, default=None, help='Threshold for binary mask prediction')
    parser.add_argument('--foreground_class_index', type=int, default=None, help='Class index for foreground probability')

    args = parser.parse_args()

    config = Config(path=args.config)

    overrides = {
        'training': {
            'image_size': args.image_size,
            'batch_size': args.batch_size,
        },
        'model': {
            'pretrained_model': args.pretrained_model,
            'num_classes': args.num_classes,
        },
        'inference': {
            'prediction_threshold': args.prediction_threshold,
            'foreground_class_index': args.foreground_class_index,
        },
    }
    config.merge(overrides)

    patch_segformer_for_mps()

    device = get_device()
    print(f"Using device: {device}")

    print("Loading model...")
    model = AugurSegformerSegmentation(
        pretrained_model=config.get('model.pretrained_model', 'nvidia/segformer-b2-finetuned-ade-512-512'),
        num_classes=config.get('model.num_classes', 2),
        k=config.get('model.median_kernel_size', 3),
        intermediate_channels=config.get('model.intermediate_channels', 256),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    prediction_threshold = config.get('inference.prediction_threshold', 0.5)
    foreground_class_index = config.get('inference.foreground_class_index', 1)
    image_size = config.get('training.image_size', 512)
    image_extensions = config.get('inference.image_extensions', ['*.jpg', '*.png'])

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        image_files = [input_path]
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(image_files)} images...")

    for image_path in tqdm(image_files):
        try:
            img_tensor, orig_size = preprocess_image(image_path, image_size)

            mask_prob = predict_image(model, img_tensor, device, foreground_index=foreground_class_index)

            mask_image = postprocess_mask(mask_prob, orig_size, threshold=prediction_threshold)
            
            output_file = output_path / f"{image_path.stem}_mask.png"
            mask_image.save(output_file)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"\nDone! Saved {len(image_files)} masks to {output_path}")


if __name__ == '__main__':
    main()
