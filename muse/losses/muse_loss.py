"""MUSE Training Loss Module.

Combines:
- Pixel reconstruction loss (L1/L2)
- Perceptual loss (LPIPS + ConvNeXt-S)
- GAN loss (Hinge) with LeCam regularization
- Topology loss (KL divergence to DINO attention)
- ITC loss (Image-Text Contrastive)

Reference:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""

from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..models.perceptual_loss import PerceptualLoss
from ..models.discriminator import NLayerDiscriminator


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam regularization loss."""
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class MUSE_Loss(torch.nn.Module):
    """MUSE Training Loss Module.

    Handles both generator and discriminator forward passes with
    automatic normalization between ImageNet-norm inputs and
    tanh-range reconstructions.

    Args:
        config: OmegaConf configuration with 'losses' section.
    """

    def __init__(self, config):
        super().__init__()
        loss_config = config.losses

        # Discriminator
        self.discriminator = NLayerDiscriminator()
        self.discriminator_iter_start = loss_config.discriminator_start
        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)

        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        # Reconstruction
        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight

        # Perceptual
        self.perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight

        # MUSE-specific
        self.topo_weight = loss_config.get("weight_topology", 2.0)
        self.itc_weight = loss_config.get("weight_itc", 0.1)
        self.kl_weight = loss_config.get("kl_weight", 1e-6)
        self.quantizer_weight = loss_config.get("quantizer_weight", 1.0)

        # Adaptive logvar (optional)
        self.use_adaptive_logvar = loss_config.get("use_adaptive_logvar", False)
        if self.use_adaptive_logvar:
            logvar_init = loss_config.get("logvar_init", 0.0)
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=True)

        self.config = config

    def should_discriminator_be_trained(self, global_step: int) -> bool:
        return global_step >= self.discriminator_iter_start

    @autocast(enabled=False)
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
        mode: str = "generator",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Forward pass for loss computation.

        Args:
            inputs: Original images (ImageNet-normalized).
            reconstructions: Reconstructed images (tanh range [-1, 1]).
            extra_result_dict: Dictionary with MUSE loss components
                (loss_topo, loss_itc, quantizer_loss, etc.).
            global_step: Current training step.
            mode: "generator" or "discriminator".

        Returns:
            total_loss: Scalar loss for backpropagation.
            loss_dict: Dictionary of detached loss components for logging.
            raw_components: (generator only) Dictionary of gradient-preserving
                loss components for orthogonality analysis.
        """
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def _forward_generator(self, inputs, reconstructions, extra_result_dict, global_step):
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # --- 1. Normalization ---
        # ImageNet Norm → [0, 1]
        std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(1, -1, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=inputs.device).view(1, -1, 1, 1)
        inputs = inputs * std + mean
        inputs = torch.clamp(inputs, 0.0, 1.0)

        # Tanh [-1, 1] → [0, 1]
        reconstructions = (reconstructions + 1) / 2.0
        reconstructions = torch.clamp(reconstructions, 0.0, 1.0)

        # --- 2. Pixel Loss ---
        if self.reconstruction_loss == "l1":
            rec_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            rec_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unknown rec loss: {self.reconstruction_loss}")

        if self.use_adaptive_logvar:
            rec_loss_final = rec_loss / torch.exp(self.logvar) + self.logvar
            rec_loss_final *= self.reconstruction_weight
        else:
            rec_loss_final = rec_loss * self.reconstruction_weight

        # --- 3. Perceptual Loss ---
        p_loss = self.perceptual_loss(inputs, reconstructions).mean()
        p_loss_final = p_loss * self.perceptual_weight

        # --- 4. GAN Loss ---
        g_loss = torch.zeros((), device=inputs.device)
        d_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0.0

        if d_factor > 0.0 and self.discriminator_weight > 0.0:
            for p in self.discriminator.parameters():
                p.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)

        g_loss_final = g_loss * self.discriminator_weight * d_factor

        # --- 5. MUSE Losses ---
        l_topo_final = torch.tensor(0.0, device=inputs.device)
        if 'loss_topo' in extra_result_dict:
            l_topo_final = extra_result_dict['loss_topo'] * self.topo_weight

        l_itc_final = torch.tensor(0.0, device=inputs.device)
        if 'loss_itc' in extra_result_dict:
            l_itc_final = extra_result_dict['loss_itc'] * self.itc_weight

        kl_loss_final = torch.tensor(0.0, device=inputs.device)
        if self.kl_weight > 0.0 and 'posteriors' in extra_result_dict:
            kl = extra_result_dict['posteriors'].kl()
            kl = torch.sum(kl) / kl.shape[0]
            kl_loss_final = kl * self.kl_weight

        total_loss = (
            rec_loss_final
            + p_loss_final
            + g_loss_final
            + l_topo_final
            + l_itc_final
            + kl_loss_final
        )

        loss_dict = {
            "total_loss": total_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "perceptual_loss": p_loss_final.detach(),
            "loss_topo": l_topo_final.detach(),
            "loss_itc": l_itc_final.detach(),
            "weighted_gan_loss": g_loss_final.detach(),
            "gan_loss": g_loss.detach(),
            "discriminator_factor": torch.tensor(d_factor),
        }

        if self.use_adaptive_logvar:
            loss_dict["logvar"] = self.logvar.detach()

        # Raw loss components (with gradients) for orthogonality analysis
        raw_components = {
            "rec_loss": rec_loss_final,
            "loss_topo": l_topo_final,
        }

        return total_loss, loss_dict, raw_components

    def _forward_discriminator(self, inputs, reconstructions, global_step):
        """Discriminator training step."""
        # Normalization
        std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(1, -1, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=inputs.device).view(1, -1, 1, 1)

        inputs = inputs * std + mean
        inputs = torch.clamp(inputs, 0.0, 1.0)

        reconstructions = (reconstructions + 1) / 2.0
        reconstructions = torch.clamp(reconstructions, 0.0, 1.0)

        d_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0.0

        for p in self.discriminator.parameters():
            p.requires_grad = True

        inputs = inputs.detach().requires_grad_(True)
        reconstructions = reconstructions.detach()

        logits_real = self.discriminator(inputs)
        logits_fake = self.discriminator(reconstructions)

        d_loss = hinge_d_loss(logits_real, logits_fake)

        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = (
                self.ema_real_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            )
            self.ema_fake_logits_mean = (
                self.ema_fake_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
            )

            d_loss += lecam_loss

        d_loss_final = d_loss * d_factor

        loss_dict = {
            "discriminator_loss": d_loss_final.detach(),
            "logits_real": logits_real.detach().mean(),
            "logits_fake": logits_fake.detach().mean(),
            "lecam_loss": lecam_loss.detach(),
        }

        return d_loss_final, loss_dict
