"""MUSE Stage 3 Training: Synergistic Tuning.

Narrative: "The Grand Unification" (Topological Manifold Alignment)
    - Loads: Stage 2 checkpoint.
    - UNFREEZES: InternViT Backbone with differential LR.
    - Strategy: Differential Learning Rates (Low LR for Backbone, High LR for Heads).
    - Objectives: All Active (Rec + Topo + ITC).

Usage:
    accelerate launch scripts/train_stage3.py config=configs/muse_1b/stage3.yaml
"""

import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from muse.utils.logger import setup_logger
from muse.utils.train_utils import (
    get_config,
    create_model_stage3,
    create_optimizer_differential,
    create_lr_scheduler,
    create_dataloader,
    create_evaluator,
    auto_resume,
    save_checkpoint,
    train_one_epoch_stage3,
)


def main():
    workspace = os.environ.get("WORKSPACE", "")
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    tracker = "wandb" if config.training.enable_wandb else "tensorboard"
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        kwargs_handlers=[ddp_kwargs],
    )

    logger = setup_logger(
        name="MUSE_Stage3",
        log_level="INFO",
        output_file=f"{output_dir}/log_rank{accelerator.process_index}.txt",
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
        logger.info(">>> MUSE Stage 3: Synergistic Tuning Started.")

    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)
    accelerator.wait_for_everyone()

    model, ema_model, loss_module, tokenizer, text_encoder = create_model_stage3(
        config, logger, accelerator
    )

    # Stage 3 uses differential LR optimizer
    optimizer, discriminator_optimizer = create_optimizer_differential(config, logger, model, loss_module)
    lr_scheduler, discriminator_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer
    )

    train_dataloader, eval_dataloader = create_dataloader(
        config, logger, accelerator, force_text_label=True
    )
    evaluator = create_evaluator(config, logger, accelerator)

    if discriminator_optimizer:
        model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
            model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler, train_dataloader, eval_dataloader
        )
    else:
        model, loss_module, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
            model, loss_module, optimizer, lr_scheduler, train_dataloader, eval_dataloader
        )

    if config.training.use_ema and ema_model:
        ema_model.to(accelerator.device)

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(config.experiment.max_train_examples / total_batch_size_without_accum)
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running MUSE Stage 3 Training *****")
    logger.info(f"  Narrative Goal = Manifold Alignment (Backbone Tuning)")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Num Epochs = {num_train_epochs}")

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch, strict=True
    )

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs - 1} started.")
        global_step = train_one_epoch_stage3(
            config, logger, accelerator,
            model, ema_model, loss_module,
            optimizer, discriminator_optimizer,
            lr_scheduler, discriminator_lr_scheduler,
            train_dataloader, eval_dataloader,
            evaluator, global_step,
            tokenizer, text_encoder,
        )
        if global_step >= config.training.max_train_steps:
            break

    accelerator.wait_for_everyone()
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema and ema_model:
            ema_model.copy_to(model.parameters())
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    accelerator.end_training()
    logger.info(">>> MUSE Stage 3 (The Grand Unification) Completed Successfully.")


if __name__ == "__main__":
    main()
