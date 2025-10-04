import random


class MaskedTiledViewGenerator:
    def __init__(self, mask_composer, tile_size=(64, 64), return_metadata=False):
        self.mask_composer = mask_composer  # e.g., SurgicalMaskComposer
        self.tile_size = tile_size
        self.return_metadata = return_metadata

    def tile_image(self, image):
        B, C, H, W = image.shape
        th, tw = self.tile_size
        assert H % th == 0 and W % tw == 0, "Image must be divisible by tile size"
        tiles = image.unfold(2, th, th).unfold(3, tw, tw)
        tiles = tiles.contiguous().view(B, C, -1, th, tw)  # (B, C, N_tiles, th, tw)
        return tiles

    def apply_masks(self, tiles):
        B, C, N, th, tw = tiles.shape
        masked_tiles = []
        metadata = []

        for b in range(B):
            batch_tiles = []
            batch_meta = []
            for n in range(N):
                tile = tiles[b, :, n]
                masked_tile, mask_info = self.mask_composer(tile)
                batch_tiles.append(masked_tile)
                batch_meta.append(mask_info)
            masked_tiles.append(torch.stack(batch_tiles))  # (N, C, th, tw)
            metadata.append(batch_meta)

        masked_tiles = torch.stack(masked_tiles)  # (B, N, C, th, tw)
        return masked_tiles, metadata

    def stitch_tiles(self, masked_tiles, H, W):
        B, N, C, th, tw = masked_tiles.shape
        tiles_per_row = W // tw
        tiles = masked_tiles.view(B, tiles_per_row, -1, C, th, tw)
        rows = [torch.cat([tiles[b, r] for r in range(tiles_per_row)], dim=2)
                for b in range(B)]
        stitched = torch.cat(rows, dim=2)  # (B, C, H, W)
        return stitched

    def __call__(self, image):
        B, C, H, W = image.shape
        tiles = self.tile_image(image)  # (B, C, N, th, tw)
        masked_tiles, metadata = self.apply_masks(tiles)
        masked_tiles = masked_tiles.permute(0, 2, 1, 3, 4)  # (B, N, C, th, tw)
        masked_image = self.stitch_tiles(masked_tiles, H, W)

        if self.return_metadata:
            return masked_image, metadata
        return masked_image


class SurgicalMaskComposer:
    def __init__(self, instrument_prob=0.3, fluid_prob=0.3, fold_prob=0.4):
        self.mask_types = ['instrument', 'fluid', 'fold']
        self.probs = [instrument_prob, fluid_prob, fold_prob]

    def __call__(self, tile):
        mask_type = random.choices(self.mask_types, weights=self.probs, k=1)[0]
        masked_tile, params = getattr(self, f"mask_{mask_type}")(tile)
        return masked_tile, {'type': mask_type, 'params': params}

    def mask_instrument(self, tile):
        # Simulate rigid occlusion (e.g., scalpel, grasper)
        occlusion = torch.zeros_like(tile)
        x = random.randint(0, tile.shape[2] // 2)
        y = random.randint(0, tile.shape[1] // 2)
        w = random.randint(tile.shape[2] // 4, tile.shape[2] // 2)
        h = random.randint(tile.shape[1] // 8, tile.shape[1] // 4)
        occlusion[:, y:y+h, x:x+w] = 1.0
        masked = tile * (1 - occlusion)
        return masked, {'x': x, 'y': y, 'w': w, 'h': h}

    def mask_fluid(self, tile):
        # Simulate semi-transparent smear or pooling
        alpha = random.uniform(0.3, 0.7)
        smear = torch.randn_like(tile) * 0.2 + 0.5
        masked = tile * (1 - alpha) + smear * alpha
        return masked.clamp(0, 1), {'alpha': alpha}

    def mask_fold(self, tile):
        # Simulate tissue fold with curved occlusion
        fold = torch.ones_like(tile)
        cx = random.randint(tile.shape[2] // 4, tile.shape[2] * 3 // 4)
        cy = random.randint(tile.shape[1] // 4, tile.shape[1] * 3 // 4)
        radius = random.randint(tile.shape[1] // 6, tile.shape[1] // 3)
        yy, xx = torch.meshgrid(torch.arange(tile.shape[1]), torch.arange(tile.shape[2]), indexing='ij')
        mask = ((xx - cx)**2 + (yy - cy)**2) < radius**2
        fold[:, mask] = 0.0
        masked = tile * fold
        return masked, {'cx': cx, 'cy': cy, 'radius': radius}


import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCoMSNTrainer(nn.Module):
    def __init__(self, encoder, projection_head, mask_composer, tile_size=(64, 64), temperature=0.2, momentum=0.999):
        super().__init__()
        self.encoder_q = nn.Sequential(encoder, projection_head)
        self.encoder_k = nn.Sequential(encoder.__class__(), projection_head.__class__())  # clone structure
        self._init_momentum_encoder()

        self.view_generator = MaskedTiledViewGenerator(mask_composer, tile_size, return_metadata=True)
        self.temperature = temperature
        self.momentum = momentum

        self.register_buffer("queue", torch.randn(12800, projection_head.output_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _init_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, x):
        # Generate masked views
        x_q, meta_q = self.view_generator(x)
        x_k, meta_k = self.view_generator(x)

        # Encode query
        q = self.encoder_q(x_q)  # (B, D)
        q = F.normalize(q, dim=1)

        # Encode key (no grad)
        with torch.no_grad():
            self._update_momentum_encoder()
            k = self.encoder_k(x_k)
            k = F.normalize(k, dim=1)

        # Compute contrastive loss
        logits = torch.mm(q, self.queue.T) / self.temperature
        labels = torch.arange(q.size(0), device=q.device)
        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        return loss, {'meta_q': meta_q, 'meta_k': meta_k}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[ptr:ptr + batch_size] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.queue.shape[0]


import torch
from torch.utils.data import DataLoader


class MoCoMSNTrainingLoop:
    def __init__(self, trainer, dataloader, optimizer, scheduler=None, device='cuda'):
        self.trainer = trainer
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.loss_log = []
        self.mask_stats = {'instrument': 0, 'fluid': 0, 'fold': 0}

    def visualize_masks(self, images, metadata, step):
        # Visualize first few masked tiles with occlusion annotations
        for i in range(min(4, len(images))):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.title(f"Step {step} | Mask: {metadata[i]['type']}")
            plt.axis('off')
            plt.savefig(f"mask_vis_step{step}_img{i}.png")
            plt.close()

    def update_mask_distribution(self, metadata_batch):
        for meta in metadata_batch:
            for m in meta:
                self.mask_stats[m['type']] += 1

    def adapt_mask_composer(self):
        # Example: reduce overused mask types
        total = sum(self.mask_stats.values())
        if total == 0: return
        freqs = {k: v / total for k, v in self.mask_stats.items()}
        for k in self.trainer.view_generator.mask_composer.mask_types:
            self.trainer.view_generator.mask_composer.probs[self.trainer.view_generator.mask_composer.mask_types.index(k)] = 1.0 - freqs[k]

    def train(self, epochs):
        for epoch in range(epochs):
            for step, (x, _) in enumerate(self.dataloader):
                x = x.to(self.device)
                loss, meta = self.trainer(x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                self.loss_log.append(loss.item())
                self.update_mask_distribution(meta['meta_q'])

                if step % 50 == 0:
                    print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                    self.visualize_masks(x, meta['meta_q'], step)

            self.adapt_mask_composer()


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class OcclusionBenchmarkLogger:
    def __init__(self, tile_grid=(8, 8)):
        self.mask_loss = defaultdict(list)
        self.tile_loss = defaultdict(list)
        self.tile_grid = tile_grid  # e.g., 8x8 for 512x512 images

    def log_step(self, loss, metadata_batch):
        for meta in metadata_batch:
            for m in meta:
                mask_type = m['type']
                self.mask_loss[mask_type].append(loss.item())

                if 'tile_index' in m:
                    self.tile_loss[m['tile_index']].append(loss.item())

    def summarize(self):
        print("\n🔍 Occlusion Benchmark Summary:")
        for mask_type, losses in self.mask_loss.items():
            avg = np.mean(losses)
            print(f"  - {mask_type}: {avg:.4f} avg loss over {len(losses)} samples")

        print("\n🧭 Tile Position Sensitivity:")
        for tile_idx, losses in sorted(self.tile_loss.items()):
            avg = np.mean(losses)
            print(f"  - Tile {tile_idx}: {avg:.4f} avg loss")

    def plot_mask_loss(self):
        plt.figure(figsize=(8, 4))
        for mask_type, losses in self.mask_loss.items():
            plt.plot(losses, label=mask_type)
        plt.title("Loss per Mask Type")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("mask_loss_trends.png")
        plt.close()

    def plot_tile_heatmap(self):
        heatmap = np.zeros(self.tile_grid)
        for (i, j), losses in self.tile_loss.items():
            heatmap[i, j] = np.mean(losses)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.title("Tile Sensitivity Heatmap")
        plt.colorbar(label="Avg Loss")
        plt.savefig("tile_loss_heatmap.png")
        plt.close()


class MaskCurriculumScheduler:
    def __init__(self, mask_composer, logger, warmup_epochs=5, max_complexity=1.0):
        self.mask_composer = mask_composer
        self.logger = logger
        self.warmup_epochs = warmup_epochs
        self.max_complexity = max_complexity
        self.current_complexity = 0.3  # start mild

    def update(self, epoch):
        if epoch < self.warmup_epochs:
            return  # no curriculum yet

        # Analyze mask loss trends
        avg_losses = {k: np.mean(v) for k, v in self.logger.mask_loss.items() if len(v) > 0}
        worst_mask = max(avg_losses, key=avg_losses.get)
        best_mask = min(avg_losses, key=avg_losses.get)

        # Increase complexity if model is resilient
        if avg_losses[best_mask] < 0.5:
            self.current_complexity = min(self.max_complexity, self.current_complexity + 0.1)

        # Adjust mask composer probabilities
        for i, mask_type in enumerate(self.mask_composer.mask_types):
            if mask_type == worst_mask:
                self.mask_composer.probs[i] = max(0.1, 1.0 - self.current_complexity)
            elif mask_type == best_mask:
                self.mask_composer.probs[i] = min(0.9, self.current_complexity)

        # Optional: adjust mask parameters (e.g., opacity, size)
        if hasattr(self.mask_composer, 'fluid_opacity_range'):
            min_alpha = 0.3 + self.current_complexity * 0.4
            max_alpha = 0.7 + self.current_complexity * 0.2
            self.mask_composer.fluid_opacity_range = (min_alpha, max_alpha)

        print(f"📈 Curriculum updated at epoch {epoch}: complexity={self.current_complexity:.2f}")


# Instantiate components
mask_composer = SurgicalMaskComposer()
view_generator = MaskedTiledViewGenerator(mask_composer, tile_size=(64, 64), return_metadata=True)

encoder = YourBackbone()  # e.g., ResNet, ViT
projection_head = YourProjectionHead()  # must expose output_dim
trainer = MoCoMSNTrainer(encoder, projection_head, mask_composer)

dataloader = DataLoader(your_dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(trainer.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

logger = OcclusionBenchmarkLogger(tile_grid=(8, 8))
curriculum = MaskCurriculumScheduler(mask_composer, logger)
loop = MoCoMSNTrainingLoop(trainer, dataloader, optimizer, scheduler)

# Training loop
for epoch in range(50):
    loop.train(epochs=1)
    logger.summarize()
    logger.plot_mask_loss()
    logger.plot_tile_heatmap()
    curriculum.update(epoch)
