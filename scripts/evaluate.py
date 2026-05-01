"""MUSE Evaluation Script: Reconstruction quality metrics.

Computes rFID, IS, PSNR, and SSIM on the evaluation set.

Usage:
    accelerate launch --num_processes=1 scripts/evaluate.py \
        config=configs/muse_1b/stage3.yaml \
        checkpoint_path=/path/to/checkpoint.bin
"""

import os
import json
import pprint
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from muse.models.muse_vit import MUSE_ViT
from muse.utils.logger import setup_logger
from muse.utils.train_utils import (
    get_config,
    create_dataloader,
    create_evaluator,
    eval_reconstruction,
    reconstruct_images,
)


def main():
    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    output_dir = os.path.join(config.experiment.output_dir, "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        split_batches=False,
    )

    logger = setup_logger(
        name="MUSE_Eval",
        log_level="INFO",
        output_file=f"{output_dir}/eval_log.txt",
    )

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # Build model
    logger.info("Building MUSE model...")
    model = MUSE_ViT(config)

    # Load checkpoint
    checkpoint_path = config.get("checkpoint_path", config.experiment.get("init_weight", ""))
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle EMA checkpoints
        if "shadow_params" in state_dict:
            logger.info(">>> Detected EMA checkpoint.")
            state_dict = state_dict["shadow_params"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        # Remove DDP "module." prefix if present
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k[7:] if k.startswith("module.") else k
            clean_state_dict[clean_key] = v

        msg = model.load_state_dict(clean_state_dict, strict=False)
        logger.info(f"Weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

    model.eval()

    # Data
    _, eval_dataloader = create_dataloader(config, logger, accelerator)
    evaluator = create_evaluator(config, logger, accelerator)

    model = accelerator.prepare(model)

    # Visualization
    logger.info(">>> Running visualization...")
    try:
        batch = next(iter(eval_dataloader))
        images = batch["image"]
        fnames = batch.get("__key__", [f"vis_{i}" for i in range(len(images))])
        reconstruct_images(
            model=model,
            original_images=images[:8].to(accelerator.device),
            fnames=fnames[:8],
            accelerator=accelerator,
            global_step=0,
            output_dir=output_dir,
            logger=logger,
            config=config,
        )
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    # Metric Evaluation
    logger.info(">>> Running metric evaluation...")
    torch.cuda.empty_cache()

    scores = eval_reconstruction(model, eval_dataloader, accelerator, evaluator)

    logger.info("\n================ EVALUATION RESULTS ================")
    logger.info(pprint.pformat(scores))

    if accelerator.is_main_process:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(scores, f, indent=4)
        logger.info(f"Results saved to {output_dir}/metrics.json")


if __name__ == "__main__":
    main()
