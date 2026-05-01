"""Perceptual loss module using LPIPS and ConvNeXt-S."""

import torch
import torch.nn.functional as F
from torchvision import models

from .lpips import LPIPS

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class PerceptualLoss(torch.nn.Module):
    """Perceptual loss combining LPIPS and/or ConvNeXt-S features.

    Supports model_name format: "lpips-convnext_s-{w_lpips}-{w_convnext}"
    for weighted combination of both losses.

    Args:
        model_name: Model specification string (e.g., "lpips-convnext_s-1.0-0.1").
    """

    def __init__(self, model_name: str = "convnext_s"):
        super().__init__()
        if ("lpips" not in model_name) and ("convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")

        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        if "lpips" in model_name:
            self.lpips = LPIPS().eval()

        if "convnext_s" in model_name:
            self.convnext = models.convnext_small(
                weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            ).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split('-')[-2:]
            self.loss_weight_lpips = float(loss_config[0])
            self.loss_weight_convnext = float(loss_config[1])

        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute perceptual loss.

        Args:
            input: Predicted images in [0, 1] range [B, 3, H, W].
            target: Target images in [0, 1] range [B, 3, H, W].

        Returns:
            Weighted perceptual loss scalar.
        """
        self.eval()
        loss = 0.
        num_losses = 0.

        if self.lpips is not None:
            lpips_loss = self.lpips(input, target)
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            input_resized = F.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
            target_resized = F.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
            pred_input = self.convnext((input_resized - self.imagenet_mean) / self.imagenet_std)
            pred_target = self.convnext((target_resized - self.imagenet_mean) / self.imagenet_std)
            convnext_loss = F.mse_loss(pred_input, pred_target, reduction="mean")

            if self.loss_weight_convnext is None:
                num_losses += 1
                loss += convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss += self.loss_weight_convnext * convnext_loss

        loss = loss / num_losses
        return loss
