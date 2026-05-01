"""MUSE Inference Script: Single image reconstruction.

Usage:
    python scripts/inference.py config=configs/muse_1b/stage3.yaml \
        checkpoint_path=/path/to/checkpoint.bin \
        img_path=/path/to/image.jpg \
        output_path=recon.jpg
"""

import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf

from muse.models.muse_vit import MUSE_ViT
from muse.utils.logger import setup_logger


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def main():
    config = get_config()
    logger = setup_logger(name="MUSE_Inference", log_level="INFO", use_accelerate=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    logger.info("Building MUSE model...")
    model = MUSE_ViT(config)

    # Load checkpoint
    checkpoint_path = config.get("checkpoint_path", "")
    if checkpoint_path:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded: {msg}")

    model = model.to(device)
    model.eval()

    # Image transform
    crop_size = config.dataset.preprocessing["crop_size"]
    mean = config.dataset.preprocessing.normalize_mean
    std = config.dataset.preprocessing.normalize_std
    interpolation = transforms.InterpolationMode.BICUBIC

    img_transform = transforms.Compose(
        [
            transforms.Resize(crop_size, interpolation=interpolation, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Load and process image
    img_path = config.get("img_path", "")
    if not img_path:
        logger.error("Please provide img_path in config or CLI.")
        return

    logger.info(f"Processing image: {img_path}")
    original_image = Image.open(img_path).convert("RGB")
    input_tensor = img_transform(original_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        z_map, _, _ = model.encode(input_tensor)
        reconstructed = model.decode(z_map)

    # Post-process: convert from [-1, 1] to [0, 255]
    reconstructed = (reconstructed + 1) / 2
    reconstructed = torch.clamp(reconstructed, 0.0, 1.0)
    reconstructed = (reconstructed * 255.0).cpu()
    save_img = TF.to_pil_image(reconstructed[0].byte())

    # Save
    output_path = config.get("output_path", "recon.jpg")
    save_img.save(output_path)
    logger.info(f"Saved reconstruction to: {output_path}")


if __name__ == "__main__":
    main()
