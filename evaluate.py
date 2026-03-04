"""
Validation/Evaluation Script for AugurSegformerSegmentation
- Evaluate model on test set
- Compute metrics (IoU, Dice, Precision, Recall)
- Log results to JSON
"""

import argparse
import json

import torch
from tqdm import tqdm

from segmenter.core import Config
from segmenter.models import AugurSegformerSegmentation
from segmenter.utils import SurgicalDataLoader
from segmenter.utils.metrics import get_device, compute_metrics
from segmenter.utils.mps_patch import patch_segformer_for_mps


def evaluate(model, data_loader, device):
    """Evaluate model on data loader."""
    model.eval()
    
    total_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            
            pred_masks = model(images)
            
            metrics = compute_metrics(pred_masks, masks)
            
            for k, v in metrics.items():
                total_metrics[k] += v
    
    n_batches = len(data_loader)
    for k in total_metrics:
        total_metrics[k] /= n_batches
    
    return total_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate AugurSegformerSegmentation')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (default: config.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_root', type=str, default=None, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--image_size', type=int, default=None, help='Image size')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Split to evaluate')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Pretrained model')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')

    args = parser.parse_args()

    config = Config(path=args.config)

    overrides = {
        'paths': {'data_root': args.data_root},
        'training': {
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'num_workers': args.num_workers,
        },
        'model': {
            'pretrained_model': args.pretrained_model,
            'num_classes': args.num_classes,
        },
    }
    config.merge(overrides)

    patch_segformer_for_mps()

    device = get_device()
    print(f"Using device: {device}")

    image_size = config.get('training.image_size', 512)

    print("Loading data...")
    dataloader = SurgicalDataLoader(
        data_root=config.get('paths.data_root', 'data/augur/data'),
        image_size=(image_size, image_size),
        batch_size=config.get('training.batch_size', 8),
        num_workers=config.get('training.num_workers', 4),
        augmentation_config=config.get('augmentation'),
        endoscopy_config=config.get('endoscopy'),
    )
    train_loader, val_loader, test_loader = dataloader.get_dataloaders()

    if args.split == 'test':
        data_loader = test_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = train_loader

    print(f"Evaluating on {args.split} set: {len(data_loader)} batches")

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
    
    print("Evaluating...")
    metrics = evaluate(model, data_loader, device)
    
    print(f"\nResults on {args.split} set:")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    if args.output:
        output_data = {
            'checkpoint': args.checkpoint,
            'split': args.split,
            'metrics': metrics,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
