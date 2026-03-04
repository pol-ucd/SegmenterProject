"""
Training Script for AugurSegformerSegmentation
- Training loop with validation
- Checkpointing (best + latest based on IoU)
- Warm restart from latest checkpoint
- TensorBoard logging (loss, metrics, images)
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch import amp as torch_amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from segmenter.core import Config
from segmenter.models import AugurSegformerSegmentation
from segmenter.loss import ComboLoss
from segmenter.utils import SurgicalDataLoader
from segmenter.utils.device import get_device
from segmenter.utils.mps_patch import patch_segformer_for_mps


def get_amp_device_type(device: torch.device) -> str:
    """Get the device type string for AMP autocast (cuda or mps)."""
    if device.type == 'cuda':
        return 'cuda'
    elif device.type == 'mps':
        return 'mps'
    return 'cpu'


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.loss_fn = ComboLoss(
            dice_weight=config.get('dice_weight', 0.5),
            focal_weight=config.get('focal_weight', 0.3),
            boundary_weight=config.get('boundary_weight', 0.2),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            boundary_sigma=config.get('boundary_sigma', 5.0),
            label_smoothing=config.get('label_smoothing', 0.0),
        )
        
        lr = config.get('learning_rate', 1e-4)
        backbone_lr_scale = config.get('backbone_lr_scale', 0.1)
        backbone_param_ids = {id(p) for p in model.base_model.segformer.parameters()}
        backbone_params = [p for p in model.parameters() if id(p) in backbone_param_ids]
        head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
        self.optimizer = optim.AdamW(
            [
                {'params': backbone_params, 'lr': lr * backbone_lr_scale},
                {'params': head_params, 'lr': lr},
            ],
            weight_decay=config.get('weight_decay', 1e-4),
        )
        
        warmup_epochs = config.get('warmup_epochs', 0)
        epochs = config.get('epochs', 50)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=config.get('min_lr', 1e-6),
        )
        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            self.scheduler = cosine
        
        self.current_epoch = 0
        self.best_iou = 0.0
        self.accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        self.log_dir = Path(config.get('log_dir', 'runs'))
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.image_log_interval = config.get('image_log_interval', 100)
        self.checkpoint_interval = config.get('checkpoint_interval', 1)

        self.amp_device_type = get_amp_device_type(device)
        self.use_amp = self.amp_device_type in ('cuda', 'mps')
        self.scaler = torch_amp.GradScaler(self.amp_device_type) if self.use_amp else None

    def _cleanup_checkpoints(self, keep: int = 3):
        checkpoints = sorted(
            self.checkpoint_dir.glob('latest_*.pt'), key=os.path.getmtime
        )
        for ckpt in checkpoints[:-keep]:
            ckpt.unlink()

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'config': self.config,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        latest_path = self.checkpoint_dir / f'latest_{timestamp}.pt'
        torch.save(checkpoint, latest_path)
        self._cleanup_checkpoints(keep=self.config.get('checkpoint_keep', 3))

        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint (IoU: {self.best_iou:.4f})")

        print(f"  Saved latest checkpoint: {latest_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint for warm restart."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_iou = checkpoint.get('best_iou', 0.0)
        
        print(f"  Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best IoU so far: {self.best_iou:.4f}")
        
        return checkpoint.get('config', {})

    def find_latest_checkpoint(self):
        """Find the latest checkpoint in checkpoint directory."""
        checkpoints = list(self.checkpoint_dir.glob('latest_*.pt'))
        if not checkpoints:
            return None
        return max(checkpoints, key=os.path.getmtime)

    def _accumulation_step(self, batch_idx: int, loss: torch.Tensor):
        """E6: single accumulation helper — branches only on AMP vs plain."""
        scaled = loss / self.accumulation_steps

        if self.use_amp:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        is_update_step = (
            (batch_idx + 1) % self.accumulation_steps == 0
            or (batch_idx + 1) == len(self.train_loader)
        )
        if not is_update_step:
            return

        if self.config.get('grad_clip', None):
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            masks = batch['masks'].to(self.device)

            with torch_amp.autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                pred_masks = self.model(images)
                pred_masks = torch.softmax(pred_masks, dim=1)[:, 1:2, :, :]
                loss, loss_dict = self.loss_fn(pred_masks, masks)

            self._accumulation_step(batch_idx, loss)

            # E5: metrics come from loss_dict — no second forward pass
            epoch_loss += loss.item()
            epoch_metrics['iou'] += loss_dict['iou']
            epoch_metrics['dice'] += loss_dict['dice_metric']
            epoch_metrics['precision'] += loss_dict['precision']
            epoch_metrics['recall'] += loss_dict['recall']

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{loss_dict["iou"]:.4f}'
            })

            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.image_log_interval == 0:
                self._log_images(images, masks, pred_masks.float(), global_step, prefix='train')

        n_batches = len(self.train_loader)
        epoch_loss /= n_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        return epoch_loss, epoch_metrics

    def validate(self):
        """Validate the model."""
        self.model.eval()

        val_loss = 0.0
        total_tp = total_fp = total_fn = 0.0
        smooth = 1e-6

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['images'].to(self.device)
                masks = batch['masks'].to(self.device)

                pred_masks = self.model(images)
                pred_masks = torch.softmax(pred_masks, dim=1)[:, 1:2, :, :]

                loss, _ = self.loss_fn(pred_masks, masks)
                val_loss += loss.item()

                pred_binary = (pred_masks > 0.5).float()
                total_tp += (pred_binary * masks).sum().item()
                total_fp += (pred_binary * (1.0 - masks)).sum().item()
                total_fn += ((1.0 - pred_binary) * masks).sum().item()

        n_batches = len(self.val_loader)
        val_loss /= n_batches

        iou = (total_tp + smooth) / (total_tp + total_fp + total_fn + smooth)
        dice = (2.0 * total_tp + smooth) / (2.0 * total_tp + total_fp + total_fn + smooth)
        precision = (total_tp + smooth) / (total_tp + total_fp + smooth)
        recall = (total_tp + smooth) / (total_tp + total_fn + smooth)

        val_metrics = {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall}

        return val_loss, val_metrics

    def _log_images(self, images, masks, pred_masks, step, prefix='train'):
        """Log images to TensorBoard."""
        n = min(4, images.shape[0])
        
        img_grid = torch.clamp(images[:n], -1, 1)
        self.writer.add_images(f'{prefix}/images', img_grid, step)
        
        mask_vis = masks[:n].repeat(1, 3, 1, 1) * 255
        self.writer.add_images(f'{prefix}/masks', mask_vis.float(), step)
        
        pred_vis = (pred_masks[:n] > 0.5).repeat(1, 3, 1, 1) * 255
        self.writer.add_images(f'{prefix}/predictions', pred_vis.float(), step)
        
        probs = pred_masks[:n].repeat(1, 3, 1, 1)
        self.writer.add_images(f'{prefix}/probabilities', probs * 255, step)

    def train(self, resume=True):
        """Main training loop."""
        if resume:
            latest = self.find_latest_checkpoint()
            if latest:
                print(f"Found latest checkpoint: {latest}")
                self.load_checkpoint(latest)
        
        for epoch in range(self.current_epoch, self.config.get('epochs', 50)):
            self.current_epoch = epoch
            
            train_loss, train_metrics = self.train_epoch()
            
            self.scheduler.step()
            
            val_loss, val_metrics = self.validate()
            
            self.writer.add_scalar('loss/train', train_loss, epoch)
            self.writer.add_scalar('loss/val', val_loss, epoch)
            
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'metrics/train_{k}', v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'metrics/val_{k}', v, epoch)
            
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
            
            is_best = val_metrics['iou'] > self.best_iou
            if is_best:
                self.best_iou = val_metrics['iou']
            
            if epoch % self.checkpoint_interval == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        self.writer.close()
        print(f"\nTraining complete! Best IoU: {self.best_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train AugurSegformerSegmentation')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (default: config.yaml)')
    parser.add_argument('--data_root', type=str, default=None, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--image_size', type=int, default=None, help='Image size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Pretrained model')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--dice_weight', type=float, default=None, help='Dice loss weight')
    parser.add_argument('--focal_weight', type=float, default=None, help='Focal loss weight')
    parser.add_argument('--boundary_weight', type=float, default=None, help='Boundary loss weight')
    parser.add_argument('--no_resume', action='store_true', help='Do not resume from latest checkpoint')
    parser.add_argument('--cache_dir', type=str, default=None, help='Model cache directory')

    args = parser.parse_args()

    config = Config(path=args.config)

    overrides = {
        'paths': {
            'data_root': args.data_root,
            'log_dir': args.log_dir,
            'checkpoint_dir': args.checkpoint_dir,
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'grad_clip': args.grad_clip,
            'num_workers': args.num_workers,
        },
        'model': {
            'pretrained_model': args.pretrained_model,
            'num_classes': args.num_classes,
            'cache_dir': args.cache_dir,
        },
        'loss': {
            'dice_weight': args.dice_weight,
            'focal_weight': args.focal_weight,
            'boundary_weight': args.boundary_weight,
        },
    }
    config.merge(overrides)

    patch_segformer_for_mps()

    device = get_device()
    print(f"Using device: {device}")

    if device.type == 'mps':
        print("Note: MPS has limited support. For production use, CUDA is recommended.")

    image_size = config.get('training.image_size', 512)

    print("Loading data...")
    dataloader = SurgicalDataLoader(
        data_root=config.get('paths.data_root', 'data/augur/data'),
        image_size=(image_size, image_size),
        batch_size=config.get('training.batch_size', 8),
        num_workers=config.get('training.num_workers', 4),
        enable_endoscopy_augmentations=True,
        augmentation_config=config.get('augmentation'),
        endoscopy_config=config.get('endoscopy'),
        mask_threshold=config.get('data.mask_threshold', 127),
    )
    train_loader, val_loader, _ = dataloader.get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("Loading model...")
    model = AugurSegformerSegmentation(
        pretrained_model=config.get('model.pretrained_model', 'nvidia/segformer-b2-finetuned-ade-512-512'),
        num_classes=config.get('model.num_classes', 2),
        k=config.get('model.median_kernel_size', 3),
        intermediate_channels=config.get('model.intermediate_channels', 256),
        cache_dir=config.get('model.cache_dir'),
    )
    model = model.to(device)

    trainer_config = {
        'epochs': config.get('training.epochs', 50),
        'learning_rate': config.get('training.learning_rate', 1e-4),
        'weight_decay': config.get('training.weight_decay', 1e-4),
        'min_lr': config.get('training.min_lr', 1e-6),
        'grad_clip': config.get('training.grad_clip', 1.0),
        'gradient_accumulation_steps': config.get('training.gradient_accumulation_steps', 1),
        'checkpoint_interval': config.get('training.checkpoint_interval', 1),
        'image_log_interval': config.get('training.image_log_interval', 100),
        'dice_weight': config.get('loss.dice_weight', 0.5),
        'focal_weight': config.get('loss.focal_weight', 0.3),
        'boundary_weight': config.get('loss.boundary_weight', 0.2),
        'focal_alpha': config.get('loss.focal_alpha', 0.25),
        'focal_gamma': config.get('loss.focal_gamma', 2.0),
        'boundary_sigma': config.get('loss.boundary_sigma', 5.0),
        'label_smoothing': config.get('loss.label_smoothing', 0.0),
        'warmup_epochs': config.get('training.warmup_epochs', 0),
        'backbone_lr_scale': config.get('training.backbone_lr_scale', 0.1),
        'checkpoint_keep': config.get('training.checkpoint_keep', 3),
        'log_dir': config.get('paths.log_dir', 'runs'),
        'checkpoint_dir': config.get('paths.checkpoint_dir', 'checkpoints'),
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_config,
    )

    print("Starting training...")
    trainer.train(resume=not args.no_resume)


if __name__ == '__main__':
    main()
