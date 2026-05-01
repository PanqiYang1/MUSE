"""PatchGAN Discriminator with blur resampling.

Reference:
    Mark Weber's improved discriminator architecture.
"""

import functools
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(nn.Conv2d):
    """Conv2D with 'SAME' padding."""

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


class BlurBlock(torch.nn.Module):
    """Anti-aliased downsampling with blur filter."""

    def __init__(self, kernel: Tuple[int] = (1, 3, 3, 1)):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ic, ih, iw = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=4, s=2)
        pad_w = self.calc_same_pad(i=iw, k=4, s=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        weight = self.kernel.expand(ic, -1, -1, -1)
        out = F.conv2d(input=x, weight=weight, stride=2, groups=x.shape[1])
        return out


class NLayerDiscriminator(torch.nn.Module):
    """Multi-scale PatchGAN discriminator with optional blur resampling.

    Args:
        num_channels: Number of input channels (default: 3).
        hidden_channels: Base hidden channel count (default: 128).
        num_stages: Number of downsampling stages (default: 3).
        blur_resample: Whether to use anti-aliased blur downsampling.
        blur_kernel_size: Size of the blur kernel (3, 4, or 5).
    """

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 128,
        num_stages: int = 3,
        blur_resample: bool = True,
        blur_kernel_size: int = 4
    ):
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (3 <= blur_kernel_size <= 5), \
            "Blur kernel size must be in [3,5] when sampling"

        in_channel_mult = (1,) + tuple(map(lambda t: 2 ** t, range(num_stages)))
        init_kernel_size = 5
        activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)

        self.block_in = torch.nn.Sequential(
            Conv2dSame(num_channels, hidden_channels, kernel_size=init_kernel_size),
            activation(),
        )

        BLUR_KERNEL_MAP = {
            3: (1, 2, 1),
            4: (1, 3, 3, 1),
            5: (1, 4, 6, 4, 1),
        }

        discriminator_blocks = []
        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            block = torch.nn.Sequential(
                Conv2dSame(in_channels, out_channels, kernel_size=3),
                torch.nn.AvgPool2d(kernel_size=2, stride=2) if not blur_resample
                else BlurBlock(BLUR_KERNEL_MAP[blur_kernel_size]),
                torch.nn.GroupNorm(32, out_channels),
                activation(),
            )
            discriminator_blocks.append(block)

        self.blocks = torch.nn.ModuleList(discriminator_blocks)
        self.pool = torch.nn.AdaptiveMaxPool2d((16, 16))

        self.to_logits = torch.nn.Sequential(
            Conv2dSame(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Discriminator logits [B, 1, H', W'].
        """
        hidden_states = self.block_in(x)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.pool(hidden_states)
        return self.to_logits(hidden_states)
