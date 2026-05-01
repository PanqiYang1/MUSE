"""Consolidated training utilities for MUSE three-stage training pipeline.

This module provides shared utilities (checkpointing, data loading, evaluation,
gradient analysis) and stage-specific functions (model creation, optimizer,
training loops) for the MUSE tokenizer's three-stage training:
    - Stage 1: Topology Warmup (freeze encoder + semantic head)
    - Stage 2: Semantic Injection (unfreeze semantic head, add ITC loss)
    - Stage 3: Synergistic Tuning (unfreeze encoder with differential LR)
"""

import json
import os
import time
import glob
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from omegaconf import OmegaConf
from torch.optim import AdamW
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextConfig

from muse.models.muse_vit import MUSE_ViT
from muse.models.ema_model import EMAModel
from muse.losses import MUSE_Loss
from muse.evaluation import Evaluator
from muse.data import TextImageDataset
from muse.utils.lr_schedulers import get_scheduler
from muse.utils.viz_utils import make_viz_from_samples


# =============================================================================
# 1. Common Helpers
# =============================================================================

def get_config():
    """Reads configs from a YAML file and CLI overrides."""
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def measure_gradient_orthogonality(model, loss_a, loss_b, accelerator):
    """[ICML Analysis Hook] Measures cosine similarity between gradients of two losses.

    Computes gradients of loss_a and loss_b with respect to MUSE block parameters,
    then returns their cosine similarity. Values near 0 indicate orthogonality,
    positive values indicate alignment, negative values indicate conflict.

    Args:
        model: The MUSE model (possibly DDP-wrapped).
        loss_a: First scalar loss tensor (must have requires_grad=True).
        loss_b: Second scalar loss tensor (must have requires_grad=True).
        accelerator: Accelerate accelerator instance.

    Returns:
        Float cosine similarity score.
    """
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod

    # Target the MUSE blocks (shared parameters between topology and semantics)
    target_module = None
    if hasattr(unwrapped, "muse_blocks"):
        target_module = unwrapped.muse_blocks
    elif hasattr(unwrapped, "muse_block"):
        target_module = unwrapped.muse_block

    if target_module is None:
        return 0.0

    shared_params = [p for p in target_module.parameters() if p.requires_grad]
    if not shared_params:
        return 0.0

    # Compute gradients
    grads_a = torch.autograd.grad(loss_a, shared_params, retain_graph=True, allow_unused=True)
    grads_b = torch.autograd.grad(loss_b, shared_params, retain_graph=True, allow_unused=True)

    vec_a, vec_b = [], []
    for ga, gb in zip(grads_a, grads_b):
        if ga is not None and gb is not None:
            vec_a.append(ga.view(-1))
            vec_b.append(gb.view(-1))

    if not vec_a:
        return 0.0

    vec_a = torch.cat(vec_a)
    vec_b = torch.cat(vec_b)
    cos_sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), eps=1e-8)
    return cos_sim.item()


# =============================================================================
# 2. Checkpointing & Resume
# =============================================================================

def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    """Saves model checkpoint with metadata."""
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "save_pretrained_weight"):
            unwrapped_model.save_pretrained_weight(
                save_path / "unwrapped_model",
                save_function=accelerator.save,
                state_dict=state_dict,
            )
        else:
            accelerator.save(state_dict, save_path / "model.bin")
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")
    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    """Loads checkpoint and returns the global step."""
    logger.info(f"Load checkpoint from {checkpoint_path}")
    accelerator.load_state(checkpoint_path, strict=strict)
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])
    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def auto_resume(config, logger, accelerator, ema_model, num_update_steps_per_epoch, strict=True):
    """Automatically finds and loads the latest checkpoint."""
    global_step = 0
    first_epoch = 0
    if config.experiment.resume:
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(config.experiment.output_dir, "checkpoint*")))
        if len(local_ckpt_list) >= 1:
            logger.info(f"Found checkpoints: {local_ckpt_list}")
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split("/")[-1].split("-")[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            global_step = load_checkpoint(Path(checkpoint_paths[0]), accelerator, logger=logger, strict=strict)
            if config.training.use_ema and ema_model is not None:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("No checkpoints found. Training from scratch.")
    return global_step, first_epoch


# =============================================================================
# 3. Common Setup Functions
# =============================================================================

def _setup_ema(config, model, accelerator, ema_model_ref):
    """Sets up EMA model with accelerator hooks. Returns EMAModel or None."""
    if not config.training.use_ema:
        return None

    ema_model = EMAModel(model.parameters(), decay=0.999, model_cls=MUSE_ViT, config=config)

    def load_model_hook(models, input_dir):
        load_model = EMAModel.from_pretrained(
            os.path.join(input_dir, "ema_model"), model_cls=MUSE_ViT, config=config
        )
        ema_model.load_state_dict(load_model.state_dict())
        ema_model.to(accelerator.device)
        del load_model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)
    return ema_model


def create_loss_module(config):
    """Creates the MUSE loss module."""
    return MUSE_Loss(config=config)


def create_text_encoder(config, logger):
    """Loads a frozen CLIP text encoder for ITC loss (used in Stages 2 & 3)."""
    path = config.model.text_encoder_path
    logger.info(f">>> Loading CLIP Text Encoder: {path}")
    tokenizer = CLIPTokenizer.from_pretrained(path)
    try:
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        text_config_dict = config_dict.get("text_config", config_dict)

        target_dim = config.model.get("text_embed_dim", config_dict.get("projection_dim", 768))
        logger.info(f"    Forcing projection_dim to: {target_dim}")
        text_config_dict["projection_dim"] = target_dim

        text_config_obj = CLIPTextConfig.from_dict(text_config_dict)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            path, config=text_config_obj, ignore_mismatched_sizes=True
        )
    except Exception as e:
        logger.error(f"Manual config loading failed: {e}")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(path)

    text_encoder.requires_grad_(False)
    text_encoder.eval()
    return tokenizer, text_encoder


def create_dataloader(config, logger, accelerator, force_text_label=False):
    """Creates train and eval dataloaders.

    Args:
        config: OmegaConf config object.
        logger: Logger instance.
        accelerator: Accelerate accelerator instance.
        force_text_label: If True, forces text label loading (for Stages 2 & 3).
    """
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset_with_text = force_text_label or dataset_config.get("dataset_with_text_label", False)

    dataset = TextImageDataset(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=total_batch_size_without_accum,
        num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        resize_shorter_edge=preproc_config.resize_shorter_edge,
        crop_size=preproc_config.crop_size,
        random_crop=preproc_config.random_crop,
        random_flip=preproc_config.random_flip,
        dataset_with_class_label=dataset_config.get("dataset_with_class_label", True),
        dataset_with_text_label=dataset_with_text,
        res_ratio_filtering=preproc_config.get("res_ratio_filtering", False),
        normalize_mean=preproc_config.get("normalize_mean", [0.0, 0.0, 0.0]),
        normalize_std=preproc_config.get("normalize_std", [1.0, 1.0, 1.0]),
        sample_ratio=dataset_config.get("sample_ratio", []),
    )
    return dataset.train_dataloader, dataset.eval_dataloader


def create_evaluator(config, logger, accelerator):
    """Creates the evaluator with rFID and IS metrics."""
    logger.info("Creating evaluator.")
    evaluator = Evaluator(
        device=accelerator.device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_codebook_usage_measure=False,
        enable_codebook_entropy_measure=False,
    )
    return evaluator


def create_optimizer_standard(config, logger, model, loss_module, need_discriminator=True):
    """Creates standard optimizers (uniform LR) for Stages 1 & 2."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate
    optimizer_cls = AdamW

    exclude = lambda n, p: (
        p.ndim < 2
        or "ln" in n
        or "bias" in n
        or "latent_tokens" in n
        or "mask_token" in n
        or "embedding" in n
        or "norm" in n
        or "gamma" in n
    )
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
    )

    discriminator_optimizer = None
    if need_discriminator and loss_module is not None:
        d_params = list(loss_module.named_parameters())
        trainable_d_params = [p for p in d_params if p[1].requires_grad]
        if len(trainable_d_params) > 0:
            discriminator_learning_rate = optimizer_config.discriminator_learning_rate
            d_gain_bias = [p for n, p in d_params if exclude(n, p) and p.requires_grad]
            d_rest = [p for n, p in d_params if include(n, p) and p.requires_grad]
            discriminator_optimizer = optimizer_cls(
                [
                    {"params": d_gain_bias, "weight_decay": 0.0},
                    {"params": d_rest, "weight_decay": optimizer_config.weight_decay},
                ],
                lr=discriminator_learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
            )

    return optimizer, discriminator_optimizer


def create_optimizer_differential(config, logger, model, loss_module, need_discriminator=True):
    """Creates optimizers with differential LR for Stage 3 (lower LR for backbone)."""
    logger.info("Creating optimizers with Differential Learning Rates.")
    optimizer_config = config.optimizer.params
    base_lr = optimizer_config.learning_rate
    backbone_lr_scale = optimizer_config.get("backbone_lr_scale", 0.1)
    logger.info(f">>> Backbone LR Scale: {backbone_lr_scale} (Target LR: {base_lr * backbone_lr_scale:.2e})")

    optimizer_cls = AdamW

    exclude = lambda n, p: (
        p.ndim < 2
        or "ln" in n
        or "bias" in n
        or "latent_tokens" in n
        or "mask_token" in n
        or "embedding" in n
        or "norm" in n
        or "gamma" in n
    )

    backbone_params_decay = []
    backbone_params_no_decay = []
    head_params_decay = []
    head_params_no_decay = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = n.startswith("encoder.")
        if is_backbone:
            if exclude(n, p):
                backbone_params_no_decay.append(p)
            else:
                backbone_params_decay.append(p)
        else:
            if exclude(n, p):
                head_params_no_decay.append(p)
            else:
                head_params_decay.append(p)

    optim_groups = [
        {"params": head_params_decay, "weight_decay": optimizer_config.weight_decay, "lr": base_lr},
        {"params": head_params_no_decay, "weight_decay": 0.0, "lr": base_lr},
        {"params": backbone_params_decay, "weight_decay": optimizer_config.weight_decay, "lr": base_lr * backbone_lr_scale},
        {"params": backbone_params_no_decay, "weight_decay": 0.0, "lr": base_lr * backbone_lr_scale},
    ]

    optimizer = optimizer_cls(optim_groups, betas=(optimizer_config.beta1, optimizer_config.beta2))

    discriminator_optimizer = None
    if need_discriminator and loss_module is not None:
        d_params = list(loss_module.named_parameters())
        trainable_d_params = [p for p in d_params if p[1].requires_grad]
        if len(trainable_d_params) > 0:
            discriminator_learning_rate = optimizer_config.discriminator_learning_rate
            d_gain_bias = [p for n, p in d_params if exclude(n, p) and p.requires_grad]
            d_rest = [p for n, p in d_params if not exclude(n, p) and p.requires_grad]
            discriminator_optimizer = optimizer_cls(
                [
                    {"params": d_gain_bias, "weight_decay": 0.0},
                    {"params": d_rest, "weight_decay": optimizer_config.weight_decay},
                ],
                lr=discriminator_learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
            )

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate schedulers for generator and discriminator."""
    logger.info("Creating lr_schedulers.")
    num_procs = accelerator.num_processes
    max_steps = config.training.max_train_steps * num_procs
    warmup_steps = config.lr_scheduler.params.warmup_steps * num_procs

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_steps,
        num_warmup_steps=warmup_steps,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )

    discriminator_lr_scheduler = None
    if discriminator_optimizer is not None:
        disc_start = config.losses.discriminator_start
        disc_steps = max(0, max_steps - disc_start)
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=disc_steps,
            num_warmup_steps=warmup_steps,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    return lr_scheduler, discriminator_lr_scheduler


# =============================================================================
# 4. Stage-Specific Model Creation
# =============================================================================

def _load_init_weight(model, config, logger):
    """Loads initial weights from a checkpoint if specified."""
    init_weight = config.experiment.get("init_weight", "")
    if init_weight:
        try:
            logger.info(f">>> Loading init weights from: {init_weight}")
            model_weight = torch.load(init_weight, map_location="cpu")
            msg = model.load_state_dict(model_weight, strict=False)
            logger.info(f"    Weights Loaded. {msg}")
        except Exception as e:
            logger.warning(f"Failed to load init_weight: {e}")


def _log_param_stats(model, accelerator, logger):
    """Logs trainable vs total parameter counts."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(
            f">>> Trainable Params: {trainable_params:,} / {total_params:,} "
            f"({(trainable_params / total_params) * 100:.2f}%)"
        )


def create_model_stage1(config, logger, accelerator):
    """Creates MUSE model for Stage 1 (Topology Warmup).

    Freezes: encoder, semantic projector, DINO teacher.
    Trains: adapter, MUSE blocks, latent projector, decoder.
    """
    logger.info(">>> [MUSE Stage 1] Creating model and loss module.")
    logger.info(">>> Narrative: Establishing Structural Manifold via Gradient Orthogonality.")

    model = MUSE_ViT(config)
    _load_init_weight(model, config, logger)

    # Freezing Strategy
    if accelerator.is_main_process:
        logger.info(">>> [Stage 1] Freezing Strategy: Backbone Frozen, Topology Stream Active.")

    model.encoder.requires_grad_(False)
    if hasattr(model, "dino_teacher"):
        model.dino_teacher.requires_grad_(False)

    # Freeze Semantic Stream
    if hasattr(model, "semantic_projector"):
        model.semantic_projector.requires_grad_(False)
        model.logit_scale.requires_grad = False
        logger.info(">>> [Info] Semantic Projector Frozen (Stage 1).")

    # Unfreeze Structural & Reconstructive Components
    trainable_list = []
    if hasattr(model, "adapter_mlp"):
        for p in model.adapter_mlp.parameters():
            p.requires_grad = True
        trainable_list.append("Adapter MLP")

    if hasattr(model, "muse_blocks"):
        for p in model.muse_blocks.parameters():
            p.requires_grad = True
        trainable_list.append("MUSE Synergy Blocks")
    elif hasattr(model, "muse_block"):
        for p in model.muse_block.parameters():
            p.requires_grad = True
        trainable_list.append("MUSE Synergy Block")

    if hasattr(model, "to_latent"):
        model.to_latent.requires_grad_(True)
        trainable_list.append("Latent Projector")

    if hasattr(model, "decoder"):
        model.decoder.requires_grad_(True)
        trainable_list.append("Decoder")
    if hasattr(model, "decoder_adapter"):
        model.decoder_adapter.requires_grad_(True)
        trainable_list.append("Decoder Adapter")

    if accelerator.is_main_process:
        logger.info(f">>> Active Trainable Components: {', '.join(trainable_list)}")

    _log_param_stats(model, accelerator, logger)

    # EMA
    ema_model = _setup_ema(config, model, accelerator, None)

    # Loss
    loss_module = create_loss_module(config)

    return model, ema_model, loss_module


def create_model_stage2(config, logger, accelerator):
    """Creates MUSE model for Stage 2 (Semantic Injection).

    Freezes: encoder, DINO teacher.
    Unfreezes: semantic projector, logit_scale (in addition to Stage 1 trainable components).
    Also creates and returns the CLIP text encoder.
    """
    logger.info(">>> [MUSE Stage 2] Creating model for Active Semantic Injection.")
    logger.info(">>> Narrative: Breaking the Impossible Triangle via Orthogonal Optimization.")

    model = MUSE_ViT(config)
    _load_init_weight(model, config, logger)

    if accelerator.is_main_process:
        logger.info(">>> [Stage 2] Strategy: Encoder Frozen, ITC Head Unfrozen, Topo Stream Active.")

    # Freeze encoder and teacher
    model.encoder.requires_grad_(False)
    if hasattr(model, "dino_teacher"):
        model.dino_teacher.requires_grad_(False)

    # UNFREEZE Semantic Components
    trainable_list = []
    if hasattr(model, "semantic_projector"):
        for p in model.semantic_projector.parameters():
            p.requires_grad = True
        model.logit_scale.requires_grad = True
        trainable_list.append("Semantic Projector (ITC)")

    # Keep structural components trainable
    if hasattr(model, "adapter_mlp"):
        for p in model.adapter_mlp.parameters():
            p.requires_grad = True
        trainable_list.append("Adapter MLP")
    if hasattr(model, "muse_blocks"):
        for p in model.muse_blocks.parameters():
            p.requires_grad = True
        trainable_list.append("MUSE Synergy Blocks")
    if hasattr(model, "to_latent"):
        model.to_latent.requires_grad_(True)
        trainable_list.append("Latent Projector")
    if hasattr(model, "decoder"):
        model.decoder.requires_grad_(True)
        trainable_list.append("Decoder")
    if hasattr(model, "decoder_adapter"):
        model.decoder_adapter.requires_grad_(True)
        trainable_list.append("Decoder Adapter")

    if accelerator.is_main_process:
        logger.info(f">>> Active Trainable Components: {', '.join(trainable_list)}")

    _log_param_stats(model, accelerator, logger)

    # EMA
    ema_model = _setup_ema(config, model, accelerator, None)

    # Loss
    loss_module = create_loss_module(config)

    # Text Encoder
    tokenizer, text_encoder = create_text_encoder(config, logger)
    text_encoder.to(accelerator.device)

    return model, ema_model, loss_module, tokenizer, text_encoder


def create_model_stage3(config, logger, accelerator):
    """Creates MUSE model for Stage 3 (Synergistic Tuning).

    Unfreezes: everything except DINO teacher.
    Uses differential learning rates (backbone_lr_scale).
    Also creates and returns the CLIP text encoder.
    """
    logger.info(">>> [MUSE Stage 3] Creating model for Topological Manifold Alignment.")
    logger.info(">>> Narrative: The Grand Unification (End-to-End Fine-tuning).")

    model = MUSE_ViT(config)
    _load_init_weight(model, config, logger)

    if accelerator.is_main_process:
        logger.info(">>> [Stage 3] Strategy: UNFREEZING ENCODER. Full model trainable.")

    # Freeze teachers
    if hasattr(model, "dino_teacher"):
        model.dino_teacher.requires_grad_(False)

    # UNFREEZE everything else
    model.requires_grad_(True)

    # Double check teachers are still frozen after global unfreeze
    if hasattr(model, "dino_teacher"):
        model.dino_teacher.requires_grad_(False)

    _log_param_stats(model, accelerator, logger)

    # EMA
    ema_model = _setup_ema(config, model, accelerator, None)

    # Loss
    loss_module = create_loss_module(config)

    # Text Encoder
    tokenizer, text_encoder = create_text_encoder(config, logger)
    text_encoder.to(accelerator.device)

    return model, ema_model, loss_module, tokenizer, text_encoder


# =============================================================================
# 5. Evaluation & Visualization
# =============================================================================

@torch.no_grad()
def eval_reconstruction(model, eval_loader, accelerator, evaluator):
    """Evaluates reconstruction quality on the eval set."""
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        original_images = torch.clone(images)
        reconstructed_images, _ = local_model(images)
        reconstructed_images = torch.clamp(reconstructed_images, -1.0, 1.0)
        reconstructed_images = (torch.round((reconstructed_images + 1) / 2 * 255.0) - 127.5) / 127.5
        evaluator.update(original_images, reconstructed_images.squeeze(2), None)

    model.train()
    return evaluator.result()


def extract_topo_saliency(attn_tensor, remove_diagonal=True):
    """Extracts topology saliency map from attention tensor.

    Args:
        attn_tensor: Attention tensor of shape [Heads, N, N].
        remove_diagonal: Whether to remove self-attention (diagonal).

    Returns:
        attn_map: Normalized numpy array of shape [H, W].
        top_indices: List of indices of the most structural heads.
    """
    num_heads, num_tokens, _ = attn_tensor.shape

    if remove_diagonal:
        diag_mask = torch.eye(num_tokens, device=attn_tensor.device, dtype=torch.bool)
        attn_tensor.masked_fill_(diag_mask, 0.0)

    # Column mean saliency
    head_saliency = attn_tensor.mean(dim=1)

    # Select top-K heads by variance
    head_variance = head_saliency.var(dim=-1)
    topk = min(3, num_heads)
    _, top_indices = torch.topk(head_variance, topk)

    # Fuse top-K heads
    best_saliency = head_saliency[top_indices].mean(dim=0)

    # Reshape to spatial grid
    grid_size = int(num_tokens**0.5)
    if grid_size**2 != num_tokens:
        if int((num_tokens - 1) ** 0.5) ** 2 == num_tokens - 1:
            best_saliency = best_saliency[1:]
            grid_size = int((num_tokens - 1) ** 0.5)
        else:
            best_saliency = best_saliency[: grid_size**2]

    attn_map = best_saliency.view(grid_size, grid_size).detach().cpu().float().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return attn_map, top_indices.tolist()


def generate_triplet_viz(img_tensor, attn_map, mean, std):
    """Generates triplet visualization: [original | heatmap | overlay].

    Requires OpenCV (cv2).
    """
    if cv2 is None:
        return None

    mean_t = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std_t = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std_t + mean_t
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    src_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    h, w = src_bgr.shape[:2]

    # Sharpen and resize heatmap
    attn_sharpened = attn_map**0.7
    heatmap_resized = cv2.resize(attn_sharpened, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(src_bgr, 0.6, colored_map, 0.4, 0)

    # Concatenate with padding
    pad = np.zeros((h, 5, 3), dtype=np.uint8)
    combined = np.hstack([src_bgr, pad, colored_map, pad, overlay])

    return combined


@torch.no_grad()
def reconstruct_images(
    model, original_images, fnames, accelerator, global_step, output_dir, logger, config=None
):
    """Generates and saves reconstruction visualizations with optional attention maps."""
    logger.info("Reconstructing images...")
    model.eval()

    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        outputs = unwrapped.encode(original_images)
        if isinstance(outputs, tuple):
            z_map = outputs[0]
            attn_topo = outputs[1] if len(outputs) > 1 else None
        else:
            z_map = outputs
            attn_topo = None
        reconstructed_images = unwrapped.decode(z_map)

    # Denormalize for visualization
    if config and config.dataset.preprocessing.get("imagenet_norm", False):
        mean = config.dataset.preprocessing.normalize_mean
        std = config.dataset.preprocessing.normalize_std
        std_t = torch.tensor(std).to(original_images.device).view(1, 3, 1, 1)
        mean_t = torch.tensor(mean).to(original_images.device).view(1, 3, 1, 1)
        vis_original = original_images * std_t + mean_t
        reconstructed_images = (reconstructed_images + 1) / 2
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        vis_original = (original_images + 1) / 2
        reconstructed_images = (reconstructed_images + 1) / 2

    vis_original = torch.clamp(vis_original, 0, 1)
    reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

    images_for_saving, _ = make_viz_from_samples(vis_original, reconstructed_images)

    if accelerator.is_main_process:
        root = Path(output_dir) / "train_images"
        os.makedirs(root, exist_ok=True)

        for i, img in enumerate(images_for_saving):
            fname = fnames[i].split("/")[-1] if i < len(fnames) else f"img_{i}"
            fname = os.path.splitext(fname)[0]
            filename = f"{global_step:08}_{fname}.png"
            img.save(root / filename)

        # Attention visualization (requires cv2)
        if attn_topo is not None and cv2 is not None:
            attn_root = root / "attention"
            os.makedirs(attn_root, exist_ok=True)
            num_viz = min(4, original_images.shape[0])

            for i in range(num_viz):
                try:
                    attn_map, top_heads = extract_topo_saliency(attn_topo[i], remove_diagonal=True)

                    # Auto-invert logic
                    h_map, w_map = attn_map.shape
                    margin = h_map // 8
                    top_edge = attn_map[:margin, :]
                    bottom_edge = attn_map[-margin:, :]
                    left_edge = attn_map[:, :margin]
                    right_edge = attn_map[:, -margin:]

                    edge_mean = (
                        np.sum(top_edge) + np.sum(bottom_edge) + np.sum(left_edge) + np.sum(right_edge)
                    ) / (top_edge.size + bottom_edge.size + left_edge.size + right_edge.size + 1e-6)

                    center_region = attn_map[margin:-margin, margin:-margin]
                    center_mean = np.mean(center_region)

                    note = ""
                    if edge_mean > center_mean:
                        attn_map = 1.0 - attn_map
                        note = "(Inv)"

                    triplet_img = generate_triplet_viz(original_images[i], attn_map, mean, std)
                    if triplet_img is not None:
                        label = f"Heads:{top_heads} {note}"
                        cv2.putText(
                            triplet_img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )
                        fname = fnames[i].split("/")[-1] if i < len(fnames) else f"img_{i}"
                        fname = os.path.splitext(fname)[0]
                        save_name = str(attn_root / f"{global_step:08}_ATTN_{fname}.png")
                        cv2.imwrite(save_name, triplet_img)

                except Exception as e:
                    logger.warning(f"Attention viz failed for img {i}: {e}")

        logger.info(f"Saved visualizations to {root}")

    model.train()


# =============================================================================
# 6. Training Loops
# =============================================================================

def _common_log_and_save(
    config,
    logger,
    accelerator,
    model,
    ema_model,
    loss_dict,
    global_step,
    batch_time_meter,
    data_time_meter,
    lr_scheduler,
    images,
    fnames,
    eval_dataloader,
    evaluator,
    ortho_score=None,
    stage=1,
):
    """Handles logging, checkpointing, visualization, and evaluation (shared logic)."""
    if config.training.use_ema and ema_model is not None:
        ema_model.step(model.parameters())
    batch_time_meter.update(time.time() - batch_time_meter._end_time)

    if (global_step + 1) % config.experiment.log_every == 0:
        samples_per_sec = (
            config.training.gradient_accumulation_steps
            * config.training.per_gpu_batch_size
            / batch_time_meter.val
        )
        lr = lr_scheduler.get_last_lr()[0]

        loss_total = loss_dict.get("total_loss", 0.0)
        loss_rec = loss_dict.get("rec_loss", 0.0)
        loss_topo = loss_dict.get("loss_topo", 0.0)
        loss_itc = loss_dict.get("loss_itc", 0.0)
        loss_lpips = loss_dict.get("p_loss", loss_dict.get("perceptual_loss", 0.0))
        loss_gan = loss_dict.get("gen_loss", loss_dict.get("gan_loss", 0.0))

        log_msg = (
            f"[S{stage}] Step: {global_step + 1} | "
            f"LR: {lr:.2e} | "
            f"Total: {loss_total:.4f} | "
            f"Rec: {loss_rec:.4f} | "
            f"LPIPS: {loss_lpips:.4f} | "
            f"Topo: {loss_topo:.4f}"
        )
        if stage >= 2:
            log_msg += f" | ITC: {loss_itc:.4f}"
        if loss_gan != 0.0:
            log_msg += f" | GAN: {loss_gan:.4f}"
        if ortho_score is not None:
            log_msg += f" | Ortho: {ortho_score:.4f}"
        log_msg += f" | {samples_per_sec:.1f} img/s"

        logger.info(log_msg)

        logs = {"lr": lr, "stage": stage, "time/batch_time": batch_time_meter.val}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                if v.device.type != "cuda":
                    v = v.to(accelerator.device)
                logs[f"train/{k}"] = accelerator.gather(v).mean().item()
            else:
                logs[f"train/{k}"] = v

        if ortho_score is not None:
            logs["analysis/grad_orthogonality"] = ortho_score

        accelerator.log(logs, step=global_step + 1)
        batch_time_meter.reset()
        data_time_meter.reset()

    # Save checkpoint
    if (global_step + 1) % config.experiment.save_every == 0:
        save_checkpoint(model, config.experiment.output_dir, accelerator, global_step + 1, logger)
        accelerator.wait_for_everyone()

    # Visualization
    if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
        if config.training.use_ema and ema_model is not None:
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
        reconstruct_images(
            model,
            images[: config.training.num_generated_images],
            fnames[: config.training.num_generated_images],
            accelerator,
            global_step + 1,
            config.experiment.output_dir,
            logger,
            config,
        )
        if config.training.use_ema and ema_model is not None:
            ema_model.restore(model.parameters())

    # Evaluation
    if eval_dataloader and (global_step + 1) % config.experiment.eval_every == 0:
        logger.info(f"Step {global_step + 1}: Starting evaluation...")
        scores = eval_reconstruction(model, eval_dataloader, accelerator, evaluator)
        if accelerator.is_main_process:
            score_str = " | ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
            logger.info(f"Step {global_step + 1} Eval Results: {score_str}")
            accelerator.log({f"eval/{k}": v for k, v in scores.items()}, step=global_step + 1)
        accelerator.wait_for_everyone()


def train_one_epoch_stage1(
    config,
    logger,
    accelerator,
    model,
    ema_model,
    loss_module,
    optimizer,
    discriminator_optimizer,
    lr_scheduler,
    discriminator_lr_scheduler,
    train_dataloader,
    eval_dataloader,
    evaluator,
    global_step,
):
    """Training loop for MUSE Stage 1 (Topology Warmup).

    Monitors gradient orthogonality between Rec and Topo losses.
    """
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()
    model.train()

    if global_step == 0 and accelerator.is_main_process:
        logger.info(">>> [MUSE Theory] Phase 1: Topological Warmup.")
        logger.info(">>> Optimizing Structure (KL-DINO) and Pixel (Rec) ONLY.")

    for i, batch in enumerate(train_dataloader):
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        with accelerator.accumulate([model, loss_module]):
            # Forward
            reconstructed_images, muse_results = model(images)

            # Raw MUSE losses
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_orig_mod"):
                unwrapped_model = unwrapped_model._orig_mod
            raw_muse_losses = unwrapped_model.compute_muse_loss(images, reconstructed_images, muse_results)
            muse_results.update(raw_muse_losses)

            # Weighted total loss
            total_loss, loss_dict, raw_components = loss_module(
                images, reconstructed_images, muse_results, global_step, mode="generator"
            )

            # Gradient Orthogonality Analysis (Rec vs Topo)
            is_log_step = (global_step + 1) % config.experiment.log_every == 0
            ortho_score = None
            if is_log_step and accelerator.is_main_process:
                l_rec = raw_components.get("rec_loss")
                l_topo = raw_components.get("loss_topo")
                if l_rec is not None and l_topo is not None:
                    if l_rec.requires_grad and l_topo.requires_grad:
                        ortho_score = measure_gradient_orthogonality(model, l_rec, l_topo, accelerator)

            # Backward
            accelerator.backward(total_loss)
            if config.training.max_grad_norm and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Discriminator step
            if discriminator_optimizer is not None:
                unwrapped_loss = accelerator.unwrap_model(loss_module)
                if unwrapped_loss.should_discriminator_be_trained(global_step):
                    d_loss, d_loss_dict = loss_module(
                        images, reconstructed_images, muse_results, global_step, mode="discriminator"
                    )
                    accelerator.backward(d_loss)
                    if config.training.max_grad_norm:
                        accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)
                    discriminator_optimizer.step()
                    discriminator_lr_scheduler.step()
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    loss_dict.update(d_loss_dict)

        # Logging & checkpointing
        if accelerator.sync_gradients:
            batch_time_meter._end_time = end
            _common_log_and_save(
                config, logger, accelerator, model, ema_model, loss_dict, global_step,
                batch_time_meter, data_time_meter, lr_scheduler, images, fnames,
                eval_dataloader, evaluator, ortho_score=ortho_score, stage=1,
            )
            end = time.time()
            global_step += 1
            if global_step >= config.training.max_train_steps:
                break

    return global_step


def _train_one_epoch_with_text(
    config,
    logger,
    accelerator,
    model,
    ema_model,
    loss_module,
    optimizer,
    discriminator_optimizer,
    lr_scheduler,
    discriminator_lr_scheduler,
    train_dataloader,
    eval_dataloader,
    evaluator,
    global_step,
    tokenizer,
    text_encoder,
    stage=2,
):
    """Shared training loop for Stages 2 & 3 (with text encoding and ITC loss).

    Args:
        stage: 2 or 3. Affects log prefixes and orthogonality analysis targets.
    """
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()
    model.train()

    if global_step == 0 and accelerator.is_main_process:
        if stage == 2:
            logger.info(">>> [MUSE Theory] Phase 2: Active Semantic Injection.")
            logger.info(">>> Optimizing ITC (Value) and Topo (Attention) SIMULTANEOUSLY.")
        else:
            logger.info(">>> [MUSE Theory] Phase 3: Synergistic Tuning.")
            logger.info(">>> Optimizing Manifold Alignment (Backbone Active).")

    for i, batch in enumerate(train_dataloader):
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        captions = batch.get("text", batch.get("caption", []))
        if not isinstance(captions, (list, tuple)) or len(captions) == 0:
            captions = [""] * images.shape[0]
        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        with accelerator.accumulate([model, loss_module]):
            # Encode text
            with torch.no_grad():
                text_inputs = tokenizer(
                    list(captions), padding="max_length", truncation=True, max_length=77, return_tensors="pt"
                ).to(accelerator.device)
                text_embeddings = text_encoder(text_inputs.input_ids).text_embeds

            # Forward
            reconstructed_images, muse_results = model(images, text_embeddings=text_embeddings)

            # Raw losses
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_orig_mod"):
                unwrapped_model = unwrapped_model._orig_mod
            raw_muse_losses = unwrapped_model.compute_muse_loss(images, reconstructed_images, muse_results)
            muse_results.update(raw_muse_losses)

            # Weighted total loss
            total_loss, loss_dict, raw_components = loss_module(
                images, reconstructed_images, muse_results, global_step, mode="generator"
            )

            # Gradient Orthogonality (ITC vs Topo)
            is_log_step = (global_step + 1) % config.experiment.log_every == 0
            ortho_score = None
            if is_log_step and accelerator.is_main_process:
                l_itc = raw_components.get("loss_itc", muse_results.get("loss_itc"))
                l_topo = raw_components.get("loss_topo", muse_results.get("loss_topo"))
                if l_itc is not None and l_topo is not None:
                    if l_itc.requires_grad and l_topo.requires_grad:
                        ortho_score = measure_gradient_orthogonality(model, l_itc, l_topo, accelerator)

            # Backward
            accelerator.backward(total_loss)
            if config.training.max_grad_norm and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Discriminator
            if discriminator_optimizer is not None:
                unwrapped_loss = accelerator.unwrap_model(loss_module)
                if unwrapped_loss.should_discriminator_be_trained(global_step):
                    d_loss, d_loss_dict = loss_module(
                        images, reconstructed_images, muse_results, global_step, mode="discriminator"
                    )
                    accelerator.backward(d_loss)
                    if config.training.max_grad_norm:
                        accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)
                    discriminator_optimizer.step()
                    discriminator_lr_scheduler.step()
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    loss_dict.update(d_loss_dict)

        # Logging & checkpointing
        if accelerator.sync_gradients:
            batch_time_meter._end_time = end
            _common_log_and_save(
                config, logger, accelerator, model, ema_model, loss_dict, global_step,
                batch_time_meter, data_time_meter, lr_scheduler, images, fnames,
                eval_dataloader, evaluator, ortho_score=ortho_score, stage=stage,
            )
            end = time.time()
            global_step += 1
            if global_step >= config.training.max_train_steps:
                break

    return global_step


def train_one_epoch_stage2(
    config, logger, accelerator,
    model, ema_model, loss_module,
    optimizer, discriminator_optimizer,
    lr_scheduler, discriminator_lr_scheduler,
    train_dataloader, eval_dataloader,
    evaluator, global_step,
    tokenizer, text_encoder,
):
    """Training loop for MUSE Stage 2 (Semantic Injection)."""
    return _train_one_epoch_with_text(
        config, logger, accelerator,
        model, ema_model, loss_module,
        optimizer, discriminator_optimizer,
        lr_scheduler, discriminator_lr_scheduler,
        train_dataloader, eval_dataloader,
        evaluator, global_step,
        tokenizer, text_encoder,
        stage=2,
    )


def train_one_epoch_stage3(
    config, logger, accelerator,
    model, ema_model, loss_module,
    optimizer, discriminator_optimizer,
    lr_scheduler, discriminator_lr_scheduler,
    train_dataloader, eval_dataloader,
    evaluator, global_step,
    tokenizer, text_encoder,
):
    """Training loop for MUSE Stage 3 (Synergistic Tuning)."""
    return _train_one_epoch_with_text(
        config, logger, accelerator,
        model, ema_model, loss_module,
        optimizer, discriminator_optimizer,
        lr_scheduler, discriminator_lr_scheduler,
        train_dataloader, eval_dataloader,
        evaluator, global_step,
        tokenizer, text_encoder,
        stage=3,
    )
