"""MUSE Zero-Shot ImageNet Classification.

Uses CLIP text encoder + MUSE semantic projector for zero-shot classification.

Usage:
    python scripts/zero_shot.py \
        --config configs/muse_1b/stage3.yaml \
        --checkpoint /path/to/checkpoint.bin
"""

import os
import sys
import glob
import argparse

import torch
import open_clip
import webdataset as wds
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms

from muse.models.muse_vit import MUSE_ViT

# ImageNet class names and templates
try:
    from scripts.zero_shot_meta import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
except ImportError:
    try:
        from zero_shot_meta import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
    except ImportError:
        print("Warning: Could not import zero_shot_meta. "
              "Please ensure scripts/zero_shot_meta.py exists.")
        IMAGENET_CLASSNAMES = None
        OPENAI_IMAGENET_TEMPLATES = None


def build_zero_shot_classifier(model, tokenizer, classnames, templates, device):
    """Builds zero-shot classifier weights from text embeddings."""
    print(f"Building zero-shot classifier ({len(classnames)} classes)...")
    with torch.no_grad():
        zeroshot_weights = []
        batch_size = 100
        for i in range(0, len(classnames), batch_size):
            batch_classnames = classnames[i : i + batch_size]
            texts = [t(c) for c in batch_classnames for t in templates]
            texts = tokenizer(texts).to(device)
            embeddings = model.encode_text(texts, normalize=True)
            embeddings = embeddings.reshape(len(batch_classnames), len(templates), -1).mean(dim=1)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(embeddings)
        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).T
    return zeroshot_weights


def get_wds_loader(url, batch_size=64):
    """Creates a WebDataset loader for ImageNet validation."""
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess(sample):
        img = sample.get("jpg") or sample.get("png") or sample.get("jpeg")
        label = sample.get("cls") or sample.get("class") or sample.get("label")
        if img is None or label is None:
            return None
        return transform(img), int(label)

    dataset = wds.WebDataset(url).decode("pil").map(preprocess).select(lambda x: x is not None)
    return wds.WebLoader(dataset, batch_size=batch_size, num_workers=32)


def load_models(config_path, checkpoint_path, device):
    """Loads MUSE model and CLIP text encoder from config."""
    conf = OmegaConf.load(config_path)

    # Auto-find checkpoint if not specified
    if checkpoint_path is None:
        output_dir = conf.experiment.output_dir
        ckpts = glob.glob(os.path.join(output_dir, "**", "*model.bin"), recursive=True)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")
        checkpoint_path = max(ckpts, key=os.path.getmtime)
        print(f">>> Auto-detected checkpoint: {checkpoint_path}")

    # Load CLIP text encoder
    text_enc_path = conf.model.text_encoder_path
    text_enc_name = "ViT-L-14"
    print(f">>> Loading CLIP from: {text_enc_path}")

    pretrained_weight = None
    if os.path.exists(text_enc_path):
        if os.path.isdir(text_enc_path):
            for candidate in ["open_clip_pytorch_model.bin", "pytorch_model.bin", "model.safetensors"]:
                p = os.path.join(text_enc_path, candidate)
                if os.path.exists(p):
                    pretrained_weight = p
                    break
        else:
            pretrained_weight = text_enc_path

    if pretrained_weight:
        clip_model, _, _ = open_clip.create_model_and_transforms(
            text_enc_name, pretrained=pretrained_weight, device=device
        )
    else:
        clip_model, _, _ = open_clip.create_model_and_transforms(
            text_enc_name, pretrained="laion2b_s32b_b82k", device=device
        )

    tokenizer = open_clip.get_tokenizer(text_enc_name)
    clip_model.eval()

    # Load MUSE model
    print(f">>> Initializing MUSE...")
    conf.model.use_active_itc = True
    conf.model.use_dino_structure = False

    muse_model = MUSE_ViT(conf)

    print(f">>> Loading Weights: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("ema_model", ckpt.get("model", ckpt))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = muse_model.load_state_dict(state_dict, strict=False)
    print(f"Loaded. Missing keys: {len(msg.missing_keys)}")

    if not any("semantic_projector" in k for k in state_dict):
        print("\n[WARN] 'semantic_projector' weights MISSING. Zero-Shot will be RANDOM.\n")

    muse_model.to(device).eval()

    # Data loader
    val_url = conf.dataset.params.eval_shards_path_or_url
    loader = get_wds_loader(val_url)

    return muse_model, clip_model, tokenizer, loader


def main():
    parser = argparse.ArgumentParser(description="MUSE Zero-Shot ImageNet Classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (auto-detect if not set)")
    args = parser.parse_args()

    if IMAGENET_CLASSNAMES is None:
        print("Error: IMAGENET_CLASSNAMES not available. Please provide zero_shot_meta.py.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    muse, clip_model, tokenizer, loader = load_models(args.config, args.checkpoint, device)

    classifier = build_zero_shot_classifier(clip_model, tokenizer, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, device)

    print("\n>>> Starting Zero-Shot Evaluation...")
    top1, top5, n = 0.0, 0.0, 0
    max_steps = 50000 // 64 + 10

    with torch.no_grad():
        for images, targets in tqdm(loader, total=max_steps):
            images = images.to(device)
            targets = targets.to(device)

            # MUSE encode -> semantic features -> project
            _, _, feat_value = muse.encode(images)
            global_sem = feat_value.mean(dim=1)
            image_embeds = muse.semantic_projector(global_sem)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_embeds @ classifier

            # Metrics
            top1 += logits.argmax(dim=-1).eq(targets).sum().item()
            _, pred_top5 = logits.topk(5, dim=-1)
            top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()
            n += images.size(0)

    if n == 0:
        print("[Error] No samples processed.")
        return

    acc1 = top1 / n * 100
    acc5 = top5 / n * 100

    print(f"\n{'=' * 40}")
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-5 Accuracy: {acc5:.2f}%")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
