"""MUSE Segmentation Linear Probe on ADE20K.

Evaluates spatial feature quality by training a lightweight segmentation head
(BN + Conv1x1) on frozen MUSE / baseline ViT features.

Supports two modes:
  - muse:     Use MUSE Synergistic Block features (512-dim).
  - baseline: Use raw InternViT / CLIP / SigLIP features.

Usage:
    torchrun --nproc_per_node=8 scripts/segment_probe.py \
        --config configs/muse_1b/stage1.yaml \
        --mode muse \
        --checkpoint /path/to/checkpoint.bin \
        --train_url /path/to/ade20k-train-{000000..000020}.tar \
        --val_url /path/to/ade20k-validation-{000000..000002}.tar \
        --epochs 10 --batch_size 8
"""

import os
import sys
import io
import glob
import math
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import webdataset as wds
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision import transforms
from transformers import AutoModel, AutoConfig

try:
    from braceexpand import braceexpand
except ImportError:
    print("Please install braceexpand: pip install braceexpand")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from muse.models.muse_vit import MUSE_ViT


# =============================================================================
# 1. DDP Utilities
# =============================================================================

def setup_ddp():
    """Initialize DDP if environment variables are set, otherwise single-GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=datetime.timedelta(minutes=60),
        )
        return rank, world_size, local_rank
    else:
        print("[Info] Running in Single GPU Mode (No DDP detected)")
        return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# 2. Segmentation Head
# =============================================================================

class SegLinearProbe(nn.Module):
    """Lightweight segmentation head: SyncBN + Conv1x1."""

    def __init__(self, in_channels: int, num_classes: int = 150):
        super().__init__()
        self.bn = nn.SyncBatchNorm(in_channels)
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_feat, target_size):
        """
        Args:
            x_feat: [B, C, H, W] feature map.
            target_size: (H_out, W_out) for upsampling.
        """
        x = self.bn(x_feat)
        x = self.head(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


# =============================================================================
# 3. Universal Wrapper (MUSE + Baseline)
# =============================================================================

class UniversalSegWrapper(nn.Module):
    """Wraps a frozen backbone + trainable segmentation head."""

    def __init__(self, backbone: nn.Module, mode: str, feature_dim: int, num_classes: int = 150):
        super().__init__()
        self.backbone = backbone
        self.mode = mode

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.head = SegLinearProbe(in_channels=feature_dim, num_classes=num_classes)
        print(f">>> Initialized SegProbe | Mode: {mode} | Dim: {feature_dim} | Classes: {num_classes}")

    def extract_muse_features(self, x):
        """Extract spatial features from MUSE Synergistic Blocks."""
        # A. Vision Encoder
        vit_embeds = self.backbone.encoder.embeddings(x)
        for layer in self.backbone.encoder.encoder.layers:
            vit_embeds = layer(vit_embeds)
        spatial_tokens = vit_embeds[:, 1:, :].contiguous().float()

        # B. Pixel Unshuffle
        feat = self.backbone.pixel_unshuffle(spatial_tokens)

        # C. Adapter
        feat = self.backbone.adapter_mlp(feat)

        # D. RoPE
        seq_len = feat.shape[1]
        if self.backbone.rope_cos.device != feat.device:
            self.backbone.rope_cos = self.backbone.rope_cos.to(feat.device)
            self.backbone.rope_sin = self.backbone.rope_sin.to(feat.device)
        cur_cos = self.backbone.rope_cos[:seq_len]
        cur_sin = self.backbone.rope_sin[:seq_len]

        # E. MUSE Blocks (use last layer output)
        current_feat = feat
        for block in self.backbone.muse_blocks:
            current_feat, _, _ = block(current_feat, cur_cos, cur_sin)

        # F. Reshape [B, N, C] → [B, C, H, W]
        B, N, C = current_feat.shape
        H = int(math.sqrt(N))
        feats = current_feat.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
        return feats

    def extract_baseline_features(self, x):
        """Extract spatial features from standard ViT (InternViT/CLIP/SigLIP)."""
        out = self.backbone(x, output_hidden_states=True)

        if hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state
        else:
            feats = out[0]

        B, L, C = feats.shape
        H_raw = int(math.sqrt(L))

        if H_raw * H_raw == L:
            spatial_feats = feats
            H = H_raw
        else:
            spatial_feats = feats[:, 1:, :]
            H = int(math.sqrt(spatial_feats.shape[1]))

        feats = spatial_feats.permute(0, 2, 1).contiguous().view(B, C, H, H)
        return feats

    def forward(self, img):
        with torch.no_grad():
            if self.mode == "muse":
                feats = self.extract_muse_features(img)
            else:
                feats = self.extract_baseline_features(img)

        logits = self.head(feats, target_size=img.shape[-2:])
        return logits


# =============================================================================
# 4. WebDataset Loader for ADE20K
# =============================================================================

def get_seg_wds_loader(url_pattern, rank, world_size, batch_size=16, num_workers=4, train=True):
    """Create a WebDataset loader for ADE20K segmentation data.

    Expected tar format: each sample has {.jpg, .seg.png}.
    seg.png: 0=background, 1..150=object classes.
    """
    urls = []
    if isinstance(url_pattern, str):
        expanded = list(braceexpand(url_pattern))
        for u in expanded:
            if "*" in u:
                urls.extend(sorted(glob.glob(u)))
            else:
                urls.append(u)
    elif isinstance(url_pattern, (list, tuple)):
        for u in url_pattern:
            urls.extend(list(braceexpand(str(u))))
    urls = sorted(list(set(urls)))

    my_urls = urls[rank::world_size]
    actual_workers = max(1, min(len(my_urls), num_workers)) if len(my_urls) > 0 else 0

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    def preprocess(sample):
        img = sample.get("jpg")
        mask_bytes = sample.get("seg.png")
        if img is None or mask_bytes is None:
            return None
        try:
            # Image transform
            img_t = transforms.functional.to_tensor(img)
            img_t = transforms.functional.normalize(img_t, mean=norm_mean, std=norm_std)

            # Mask (PNG → Class ID)
            mask_pil = Image.open(io.BytesIO(mask_bytes))
            mask_np = np.array(mask_pil)
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            mask_t = torch.from_numpy(mask_np).long()

            # Resize to fixed size
            target_size = (512, 512)
            img_t = F.interpolate(
                img_t.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
            ).squeeze(0)
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode="nearest"
            ).squeeze().long()

            # ADE20K label correction: 0=bg → 255 (ignore), shift 1..150 → 0..149
            mask_t[mask_t == 0] = 255
            mask_t = mask_t - 1
            mask_t[mask_t == 254] = 255

            return img_t, mask_t
        except Exception:
            return None

    dataset = (
        wds.WebDataset(my_urls, nodesplitter=lambda x: x, empty_check=False)
        .decode("pil")
        .map(preprocess)
        .select(lambda x: x is not None)
    )

    est_total = 20210 if train else 2000
    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=actual_workers, drop_last=train)
    if train:
        loader = loader.shuffle(1000)
    loader = loader.with_length(est_total // world_size // batch_size)
    return loader


# =============================================================================
# 5. Training & Validation
# =============================================================================

def run_training(model, train_loader, val_loader, device, rank, epochs=10, lr=0.01):
    """Train segmentation head with SGD, evaluate mIoU each epoch."""
    optimizer = torch.optim.SGD(model.module.head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_miou = 0.0

    for epoch in range(epochs):
        model.module.backbone.eval()
        model.module.head.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}") if rank == 0 else train_loader

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            if rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        miou = validate(model, val_loader, device, rank)
        if rank == 0:
            print(f"Epoch {epoch + 1} | mIoU: {miou:.2f}%")
            if miou > best_miou:
                best_miou = miou
                torch.save(model.module.head.state_dict(), "best_seg_head.pth")

    return best_miou


@torch.no_grad()
def validate(model, loader, device, rank, num_classes=150):
    """Compute mIoU on the validation set."""
    model.eval()
    intersection = torch.zeros(num_classes).to(device)
    union = torch.zeros(num_classes).to(device)

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)

        keep = masks != 255
        preds = preds[keep]
        masks = masks[keep]

        for i in range(num_classes):
            pred_i = preds == i
            mask_i = masks == i
            intersection[i] += (pred_i & mask_i).sum()
            union[i] += (pred_i | mask_i).sum()

    if dist.is_initialized():
        dist.all_reduce(intersection)
        dist.all_reduce(union)

    iou = intersection / (union + 1e-6)
    return iou.mean().item() * 100


# =============================================================================
# 6. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MUSE Segmentation Linear Probe on ADE20K")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--mode", type=str, required=True, choices=["muse", "baseline"],
                        help="'muse' for MUSE features, 'baseline' for raw ViT features")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to MUSE checkpoint")
    parser.add_argument("--train_url", type=str,
                        default="./ade20k_wds/ade20k-train-{000000..000020}.tar",
                        help="ADE20K training tar shards")
    parser.add_argument("--val_url", type=str,
                        default="./ade20k_wds/ade20k-validation-{000000..000002}.tar",
                        help="ADE20K validation tar shards")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=150, help="Number of semantic classes")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    conf = OmegaConf.load(args.config)

    # ------------------------------------------------------------------
    # Load backbone
    # ------------------------------------------------------------------
    if args.mode == "muse":
        if rank == 0:
            print(">>> Loading MUSE model...")
        backbone = MUSE_ViT(conf)
        if args.checkpoint:
            if rank == 0:
                print(f">>> Loading Checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            state_dict = ckpt.get("ema_model", ckpt.get("model", ckpt))
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            backbone.load_state_dict(state_dict, strict=False)
        else:
            if rank == 0:
                print("[Warning] MUSE mode selected but no checkpoint provided!")
        feature_dim = 512  # MUSE fixed output

    elif args.mode == "baseline":
        model_path = conf.model.mllm_path
        if rank == 0:
            print(f">>> Loading Baseline from: {model_path}")

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(hf_config, "vision_config"):
            feature_dim = hf_config.vision_config.hidden_size
        else:
            feature_dim = hf_config.hidden_size

        full_model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if hasattr(full_model, "vision_model"):
            backbone = full_model.vision_model
        else:
            backbone = full_model
        backbone = backbone.float()

    backbone.to(device)

    # ------------------------------------------------------------------
    # Wrap & DDP
    # ------------------------------------------------------------------
    model = UniversalSegWrapper(backbone, mode=args.mode, feature_dim=feature_dim,
                                num_classes=args.num_classes).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if rank == 0:
        print(f">>> Model Ready. Feature Dim: {feature_dim}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader = get_seg_wds_loader(args.train_url, rank, world_size, args.batch_size, train=True)
    val_loader = get_seg_wds_loader(args.val_url, rank, world_size, args.batch_size, train=False)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if rank == 0:
        print(f">>> Starting Segmentation Probe ({args.mode})")
    final_miou = run_training(model, train_loader, val_loader, device, rank, args.epochs, args.lr)

    if rank == 0:
        print(f"\n{'=' * 40}")
        print(f"FINAL mIoU ({args.mode.upper()}): {final_miou:.2f}%")
        print(f"{'=' * 40}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
