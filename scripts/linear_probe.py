"""MUSE Linear Probe Evaluation on ImageNet.

Extracts features via model.encode(), applies L2 normalization + BatchNorm,
then trains a linear classifier with SGD.

Usage:
    torchrun --nproc_per_node=8 scripts/linear_probe.py \
        --config configs/muse_1b/stage3.yaml \
        --mode muse \
        --checkpoint /path/to/checkpoint.bin
"""

import os
import sys
import glob
import argparse
import datetime
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import webdataset as wds
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision import transforms
from transformers import AutoModel

try:
    from braceexpand import braceexpand
except ImportError:
    print("Please install braceexpand: pip install braceexpand")
    sys.exit(1)

from muse.models.muse_vit import MUSE_ViT


# =============================================================================
# DDP Utilities
# =============================================================================

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=60)
        )
        return rank, world_size, local_rank
    else:
        print("[Info] Running in Single GPU Mode (No DDP detected)")
        return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Data Loading
# =============================================================================

def get_distributed_wds_loader(url_pattern, rank, world_size, batch_size=256, num_workers=8, train=True):
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
    if len(urls) == 0:
        if rank == 0:
            print(f"[Error] No files found for pattern: {url_pattern}")
        sys.exit(1)

    my_urls = urls[rank::world_size]
    actual_workers = max(1, min(len(my_urls), num_workers)) if len(my_urls) > 0 else 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    def preprocess(sample):
        img = sample.get("jpg") or sample.get("png") or sample.get("jpeg")
        cls_key = next((k for k in ["cls", "class", "label"] if k in sample), None)
        if img is None or cls_key is None:
            return None
        try:
            return transform(img), int(sample[cls_key])
        except Exception:
            return None

    def identity_splitter(source):
        return source

    dataset = (
        wds.WebDataset(my_urls, nodesplitter=identity_splitter, empty_check=False)
        .decode("pil")
        .map(preprocess)
        .select(lambda x: x is not None)
    )

    est_total_size = 1281167 if train else 50000
    est_local_size = est_total_size // world_size

    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=actual_workers)
    loader = loader.with_length(est_local_size // batch_size)
    return loader


# =============================================================================
# Feature Extraction
# =============================================================================

@torch.no_grad()
def extract_features_local(model, loader, device, mode, rank):
    model.eval()
    features_list, labels_list = [], []

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    iterator = tqdm(loader, desc=f"Rank {rank} Extracting") if rank == 0 else loader

    for images, targets in iterator:
        images = images.to(device, dtype=model_dtype)
        feats = forward_batch(model, images, mode)
        features_list.append(feats.float().cpu().numpy())
        labels_list.append(targets.numpy())

    if len(features_list) > 0:
        return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)
    return np.array([]), np.array([])


def forward_batch(model, images, mode):
    if mode == "muse":
        # MUSE encode returns: (z_map, attn_topo, feat_itc)
        _, _, feat_itc = model.encode(images)
        return feat_itc.mean(dim=1)  # Global Average Pooling
    else:
        out = model(images, output_hidden_states=True)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        else:
            return out[0].mean(dim=1)


# =============================================================================
# Linear Classifier
# =============================================================================

class LinearClassifier(nn.Module):
    def __init__(self, dim, num_classes=1000):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.linear(self.bn(x))


def train_linear_probe_gpu(X_train, y_train, X_val, y_val, device, batch_size=4096, epochs=100, lr=0.3):
    print(f"\n{'=' * 20} Start Linear Probe Training (GPU) {'=' * 20}")
    print(f"Train Set: {X_train.shape}, Val Set: {X_val.shape}")

    X_train_t = F.normalize(torch.from_numpy(X_train).to(device), p=2, dim=1)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = F.normalize(torch.from_numpy(X_val).to(device), p=2, dim=1)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LinearClassifier(dim=X_train.shape[1], num_classes=1000).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for bx, by in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X_val_t).argmax(dim=1)
                acc = (preds == y_val_t).float().mean().item() * 100
            if acc > best_acc:
                best_acc = acc
            print(f"Epoch {epoch + 1:03d}/{epochs} | Val Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

    return best_acc


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MUSE Linear Probe Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--mode", type=str, required=True, choices=["muse", "baseline"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--temp_dir", type=str, default="./temp_feats", help="Temp dir for features")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lp_lr", type=float, default=0.1, help="Linear Probe LR")
    parser.add_argument("--lp_epochs", type=int, default=100, help="Linear Probe epochs")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(args.temp_dir, exist_ok=True)
        print(f">>> Running MUSE Linear Probe on {world_size} GPUs.")

    conf = OmegaConf.load(args.config)

    # Load model
    if args.mode == "muse":
        model = MUSE_ViT(conf)
        if args.checkpoint:
            if rank == 0:
                print(f">>> Loading Checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            state_dict = ckpt.get("ema_model", ckpt.get("model", ckpt))
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        model.to(device)
    elif args.mode == "baseline":
        path = conf.model.mllm_path
        model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float16)
        if hasattr(model, "vision_model"):
            model = model.vision_model
        model.to(device)

    # Extract features
    def process_split(split_name, url):
        is_train = split_name == "train"
        loader = get_distributed_wds_loader(url, rank, world_size, batch_size=args.batch_size, train=is_train)
        feats, labels = extract_features_local(model, loader, device, args.mode, rank)

        np.save(os.path.join(args.temp_dir, f"{split_name}_feat_{rank}.npy"), feats)
        np.save(os.path.join(args.temp_dir, f"{split_name}_label_{rank}.npy"), labels)

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            print(f">>> Merging {split_name} features...")
            f_list, l_list = [], []
            for r in range(world_size):
                f_path = os.path.join(args.temp_dir, f"{split_name}_feat_{r}.npy")
                l_path = os.path.join(args.temp_dir, f"{split_name}_label_{r}.npy")
                if os.path.exists(f_path):
                    f_data = np.load(f_path)
                    l_data = np.load(l_path)
                    if f_data.shape[0] > 0:
                        f_list.append(f_data)
                        l_list.append(l_data)
            if f_list:
                return np.concatenate(f_list), np.concatenate(l_list)
            print(f"[Error] No features for {split_name}!")
            sys.exit(1)
        return None, None

    train_url = conf.dataset.params.train_shards_path_or_url
    val_url = conf.dataset.params.eval_shards_path_or_url

    if rank == 0:
        print("\n>>> Phase 1: Extracting Train Features...")
    X_train, y_train = process_split("train", train_url)

    if rank == 0:
        print("\n>>> Phase 2: Extracting Val Features...")
    X_val, y_val = process_split("val", val_url)

    # Train classifier (rank 0 only)
    if rank == 0:
        del model
        torch.cuda.empty_cache()
        final_acc = train_linear_probe_gpu(
            X_train, y_train, X_val, y_val, device, epochs=args.lp_epochs, lr=args.lp_lr
        )
        print(f"\n{'=' * 40}")
        print(f"FINAL RESULT ({args.mode.upper()})")
        print(f"Top-1 Accuracy: {final_acc:.2f}%")
        print(f"{'=' * 40}")
        shutil.rmtree(args.temp_dir, ignore_errors=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
