"""MUSE Attention Map Visualization Script.

Visualizes the topology attention maps from MUSE Synergistic Blocks.

Supports two modes:
  1. Single image mode:  Provide --img_path to visualize one image.
  2. Batch mode:         Uses the eval dataset from config to process many images.

Usage:
    # Single image
    python scripts/visualize_attention.py \
        config=configs/muse_1b/stage1.yaml \
        checkpoint_path=/path/to/checkpoint.bin \
        img_path=/path/to/image.jpg \
        output_dir=./attention_vis

    # Batch mode (eval dataset)
    python scripts/visualize_attention.py \
        config=configs/muse_1b/stage1.yaml \
        checkpoint_path=/path/to/checkpoint.bin \
        output_dir=./attention_vis \
        max_images=500 \
        batch_size=8
"""

import os
import sys
import concurrent.futures

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from muse.models.muse_vit import MUSE_ViT
from muse.utils.logger import setup_logger
from muse.utils.viz_utils import (
    process_attention_refined,
    denormalize_image,
    apply_heatmap_overlay,
    save_attention_visualization,
    make_attention_grid_pil,
)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def load_model(config, logger, device):
    """Build MUSE model and load checkpoint."""
    logger.info("Building MUSE model...")
    model = MUSE_ViT(config)

    checkpoint_path = config.get("checkpoint_path", "")
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle various checkpoint formats
        if "shadow_params" in state_dict:
            logger.info("Detected EMA checkpoint (shadow_params).")
            state_dict = state_dict["shadow_params"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("module.", "").replace("ema_model.", "")
            clean_state_dict[clean_key] = v

        msg = model.load_state_dict(clean_state_dict, strict=False)
        logger.info(f"Loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    else:
        logger.warning("No checkpoint_path specified; using random weights.")

    model.to(device).eval()
    return model


def get_image_transform(config):
    """Build eval image transform from config."""
    crop_size = config.dataset.preprocessing.crop_size
    mean = list(config.dataset.preprocessing.normalize_mean)
    std = list(config.dataset.preprocessing.normalize_std)
    interpolation = transforms.InterpolationMode.BICUBIC

    return transforms.Compose([
        transforms.Resize(crop_size, interpolation=interpolation, antialias=True),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]), mean, std


def visualize_single_image(model, config, logger, device):
    """Single image attention visualization mode."""
    img_path = config.get("img_path", "")
    output_dir = config.get("output_dir", "./attention_vis")
    temperature = config.get("temperature", 0.1)
    remove_diagonal = config.get("remove_diagonal", True)
    use_cc_filter = config.get("use_cc_filter", True)

    os.makedirs(output_dir, exist_ok=True)

    transform, mean, std = get_image_transform(config)

    logger.info(f"Processing image: {img_path}")
    original_image = Image.open(img_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        z_map, attn_topo, _ = model.encode(input_tensor)

    # attn_topo: [B, Heads, N, N]
    attn_single = attn_topo[0]  # [Heads, N, N]

    # Process attention
    refined_map, clean_mask = process_attention_refined(
        attn_single,
        temperature=temperature,
        remove_diagonal=remove_diagonal,
        use_cc_filter=use_cc_filter,
    )

    # De-normalize for saving
    src_bgr = denormalize_image(input_tensor[0], mean, std)

    # Save
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(output_dir, img_name)
    save_attention_visualization(save_dir, src_bgr, refined_map, clean_mask)

    # Also save a summary grid
    grid_img = make_attention_grid_pil(
        input_tensor, attn_topo,
        mean=mean, std=std,
        temperature=temperature,
        remove_diagonal=remove_diagonal,
        use_cc_filter=use_cc_filter,
    )
    grid_img.save(os.path.join(output_dir, f"{img_name}_grid.png"))

    logger.info(f"Saved attention visualization to: {save_dir}")
    logger.info(f"Saved grid to: {output_dir}/{img_name}_grid.png")


def visualize_batch(model, config, logger, device):
    """Batch attention visualization using eval dataset."""
    import webdataset as wds

    output_dir = config.get("output_dir", "./attention_vis")
    max_images = config.get("max_images", 500)
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 2)
    temperature = config.get("temperature", 0.1)
    remove_diagonal = config.get("remove_diagonal", True)
    use_cc_filter = config.get("use_cc_filter", True)

    os.makedirs(output_dir, exist_ok=True)

    transform, mean, std = get_image_transform(config)

    # Get eval dataset URL from config
    eval_url = config.dataset.params.get("eval_shards_path_or_url", "")
    if not eval_url:
        logger.error("No eval_shards_path_or_url in config. Cannot run batch mode.")
        return

    logger.info(f"Loading eval data from: {eval_url}")

    def preprocess(sample):
        for k in ["jpg", "png", "jpeg", "webp"]:
            if k in sample:
                return transform(sample[k]), sample.get("__key__", "unknown")
        return None

    dataset = (
        wds.WebDataset(eval_url)
        .decode("pil")
        .map(preprocess)
        .select(lambda x: x is not None)
    )
    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Thread pool for async IO
    io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    count = 0

    logger.info(f"Processing up to {max_images} images (batch_size={batch_size})...")

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Attention Visualization"):
            imgs_tensor, keys = batch_data
            imgs_tensor = imgs_tensor.to(device)

            # Encode
            z_map, attn_topo, _ = model.encode(imgs_tensor)

            # Process each sample in the batch
            for i in range(len(keys)):
                key = keys[i]
                img_tensor = imgs_tensor[i]
                attn_single = attn_topo[i]  # [Heads, N, N]

                # De-normalize original image
                src_bgr = denormalize_image(img_tensor, mean, std)

                # Process attention
                refined_map, clean_mask = process_attention_refined(
                    attn_single.cpu(),
                    temperature=temperature,
                    remove_diagonal=remove_diagonal,
                    use_cc_filter=use_cc_filter,
                )

                # Async save
                save_dir = os.path.join(output_dir, key)
                io_pool.submit(
                    save_attention_visualization,
                    save_dir, src_bgr, refined_map, clean_mask,
                )

                count += 1

            if count >= max_images:
                break

    logger.info("Waiting for IO tasks to finish...")
    io_pool.shutdown(wait=True)
    logger.info(f"Done. Saved {count} attention maps to: {output_dir}")


def main():
    config = get_config()
    logger = setup_logger(name="MUSE_AttentionViz", log_level="INFO", use_accelerate=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config, logger, device)

    img_path = config.get("img_path", "")
    if img_path:
        visualize_single_image(model, config, logger, device)
    else:
        visualize_batch(model, config, logger, device)


if __name__ == "__main__":
    main()
