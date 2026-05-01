"""MUSE ViT: Multi-granularity Unified Semantic Embedding Vision Transformer.

Core architecture of the MUSE tokenizer featuring:
- InternViT encoder backbone
- Pixel Unshuffle (Space-to-Depth) adapter
- Synergistic Blocks with decoupled topology and semantic streams
- DC-AE pixel decoder
- Optional DINO teacher for topology alignment
- Optional ITC head for semantic contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from omegaconf import OmegaConf
from typing import Optional

from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel
from diffusers.models import AutoencoderDC

from .base_model import BaseModel


# ==============================================================================
# 1. Modern Components (RMSNorm, RoPE, SwiGLU)
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def init_rotary_embeddings(dim, max_seq_len=4096, base=10000):
    """Initialize Rotary Position Embeddings (RoPE)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary_pos_emb(t, cos, sin):
    """Apply rotary position embeddings to queries/keys."""
    d = t.shape[-1] // 2
    x1, x2 = t[..., :d], t[..., d:]
    rotated_t = torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0).to(t.device)
    sin = sin.unsqueeze(0).unsqueeze(0).to(t.device)
    seq_len = t.shape[2]
    return (t * cos[:, :, :seq_len, :]) + (rotated_t * sin[:, :, :seq_len, :])


class SwiGLU(nn.Module):
    """SwiGLU Feedforward Network."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        hidden_features = int(2 * hidden_features / 3)
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ==============================================================================
# 2. Gradient Filter & Utils
# ==============================================================================

class GradientScaler(torch.autograd.Function):
    """Custom autograd function for spectral gradient filtering.
    
    Scales gradients during backpropagation to control the flow
    of gradient information through specific paths.
    """

    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def spectral_filter(x, scale=1.0):
    """Apply spectral gradient filtering to tensor x."""
    return GradientScaler.apply(x, scale)


def resize_attention_map_4d(attn, target_h, target_w):
    """Resize a 2D attention map [B, N_src, N_src] to [B, N_tgt, N_tgt].

    Uses bilinear interpolation on the 4D spatial structure.
    """
    B, N_src, _ = attn.shape
    H_src = int(math.sqrt(N_src))
    W_src = N_src // H_src

    attn_4d = attn.view(B, H_src, W_src, H_src, W_src)
    attn_keys = attn_4d.view(B * N_src, 1, H_src, W_src)
    attn_keys = F.interpolate(attn_keys, size=(target_h, target_w), mode='bilinear', align_corners=False)

    attn_keys = attn_keys.view(B, H_src, W_src, target_h, target_w)
    N_tgt = target_h * target_w

    attn_queries = attn_keys.permute(0, 3, 4, 1, 2).reshape(B * N_tgt, 1, H_src, W_src)
    attn_final = F.interpolate(attn_queries, size=(target_h, target_w), mode='bilinear', align_corners=False)

    attn_final = attn_final.view(B, N_tgt, N_tgt)
    attn_final = attn_final / (attn_final.sum(dim=-1, keepdim=True) + 1e-6)
    return attn_final


# ==============================================================================
# 3. MUSE Synergistic Block (Decoupled Topology + Semantic Streams)
# ==============================================================================

class SynergisticBlock(nn.Module):
    """Core MUSE block with decoupled topology and semantic streams.

    Architecture:
        - Topology stream: q_topo, k_topo → structural attention map
        - Reconstruction value: v_sem → features for pixel decoding
        - Semantic value: v_itc → features for ITC contrastive loss
        - Gradient shield: detached attention map for value aggregation
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. Topology Stream (Structure / Attention)
        self.q_topo = nn.Linear(dim, dim, bias=False)
        self.k_topo = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # 2. Reconstruction Value (for Attention aggregation → Decoder)
        self.v_sem = nn.Linear(dim, dim, bias=False)

        # 3. Semantic Value (for ITC Loss, decoupled from reconstruction)
        self.v_itc = nn.Linear(dim, dim, bias=False)

        # 4. Components
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x, rope_cos, rope_sin):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # --- A. Topology Stream (Structure) ---
        q = self.q_topo(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_topo(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_sem(x).reshape(B, N, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rotary_pos_emb(q, rope_cos, rope_sin)
        k = apply_rotary_pos_emb(k, rope_cos, rope_sin)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_logits.softmax(dim=-1)

        # Gradient Shield: detach attention for value aggregation
        attn_map_for_flow = attn_map.detach()

        # --- B. Synergy Re-integration ---
        x_synergy = (attn_map_for_flow @ v).transpose(1, 2).reshape(B, N, C)
        x_synergy = self.proj(x_synergy)

        x = shortcut + x_synergy
        x = x + self.mlp(self.norm2(x))

        # --- C. Semantic Export (decoupled) ---
        v_raw_itc = self.v_itc(shortcut)

        return x, attn_map, v_raw_itc


# ==============================================================================
# 4. MUSE Model
# ==============================================================================

class MUSE_ViT(BaseModel, PyTorchModelHubMixin):
    """MUSE Vision Transformer for image tokenization.

    Architecture:
        InternViT encoder → Pixel Unshuffle → Adapter MLP →
        MUSE Synergistic Blocks (×N) → Latent Projection → DC-AE Decoder

    Args:
        config: OmegaConf configuration object with model parameters.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        super().__init__()
        self.config = config

        self.latent_dim = config.model.get("latent_dim", 32)
        self.use_dino_topo = config.model.get("use_dino_structure", True)
        self.use_active_itc = config.model.get("use_active_itc", True)
        self.grad_scale = config.model.get("grad_scale", 1.0)
        self.num_muse_layers = config.model.get("muse_layers", 6)

        print(f"\n>>> [MUSE] Decoupled Heads + Pixel Unshuffle Enabled.")
        print(f">>> Layers: {self.num_muse_layers} | Latent: {self.latent_dim} | Grad Scale: {self.grad_scale}")

        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.dino_patch_size = 16

        # 1. Encoder (InternViT)
        student_model = AutoModel.from_pretrained(
            self.config.model.mllm_path, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            trust_remote_code=True, use_flash_attn=False
        )
        self.encoder = student_model.vision_model
        self.encoder.requires_grad_(False)

        if hasattr(student_model.config, "vision_config"):
            vision_dim = student_model.config.vision_config.hidden_size
        else:
            vision_dim = self.encoder.config.hidden_size

        # 2. Adapter (with Space-to-Depth / Pixel Unshuffle)
        # Pixel Unshuffle multiplies channels by 4 (2×2 downsampling)
        self.adapter_mlp = nn.Sequential(
            RMSNorm(vision_dim * 4),
            nn.Linear(vision_dim * 4, 512, bias=False),
            nn.GELU()
        )

        # 3. RoPE
        head_dim = 512 // 8
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        cos, sin = init_rotary_embeddings(head_dim, max_seq_len=4096)
        self.rope_cos = cos
        self.rope_sin = sin

        # 4. Deep MUSE Stack
        self.muse_blocks = nn.ModuleList([
            SynergisticBlock(dim=512, num_heads=8)
            for _ in range(self.num_muse_layers)
        ])

        # 5. Latent Projection
        self.to_latent = nn.Linear(512, self.latent_dim)

        # 6. Heads
        if self.use_active_itc:
            self.semantic_dim = 768
            self.semantic_projector = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.GELU(),
                nn.Linear(512, self.semantic_dim, bias=False)
            )
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        # 7. Pixel Decoder (DC-AE)
        dc_ae = AutoencoderDC.from_pretrained(self.config.model.dc_ae_path, torch_dtype=torch.float32)
        self.decoder = dc_ae.decoder
        self.decoder.requires_grad_(True)
        self.decoder_adapter = nn.Conv2d(self.latent_dim, 32, 1) if self.latent_dim != 32 else nn.Identity()

        # 8. DINO Teacher
        if self.use_dino_topo:
            self.dino_teacher = AutoModel.from_pretrained(
                self.config.model.dinov3_path, trust_remote_code=True, output_attentions=True
            )
            self.dino_teacher.requires_grad_(False)
            self.dino_teacher.eval()
            if hasattr(self.dino_teacher.config, "patch_size"):
                self.dino_patch_size = self.dino_teacher.config.patch_size

        # Initialize trainable modules
        self.adapter_mlp.apply(self._init_weights)
        self.muse_blocks.apply(self._init_weights)
        self.to_latent.apply(self._init_weights)
        if isinstance(self.decoder_adapter, nn.Module):
            self.decoder_adapter.apply(self._init_weights)
        if self.use_active_itc:
            self.semantic_projector.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight.data, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)

    def _preprocess_for_dino(self, x):
        """Preprocess images for DINO teacher (normalize + pad to patch size)."""
        x = (x + 1.0) / 2.0
        x = (x - self.dino_mean.to(x.device)) / self.dino_std.to(x.device)
        _, _, h, w = x.shape
        ph = (self.dino_patch_size - h % self.dino_patch_size) % self.dino_patch_size
        pw = (self.dino_patch_size - w % self.dino_patch_size) % self.dino_patch_size
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))
        return x

    def get_dino_attention(self, x):
        """Extract attention maps from frozen DINO teacher."""
        x_dino = self._preprocess_for_dino(x)
        b, _, h, w = x_dino.shape
        num_spatial = (h // self.dino_patch_size) * (w // self.dino_patch_size)
        with torch.no_grad():
            outputs = self.dino_teacher(x_dino, output_attentions=True)
            attn = outputs.attentions[-1]
            attn_spatial = attn[:, :, -num_spatial:, -num_spatial:]
            attn_spatial = attn_spatial.mean(dim=1)
            attn_spatial = attn_spatial / (attn_spatial.sum(dim=-1, keepdim=True) + 1e-6)
            return attn_spatial

    def pixel_unshuffle(self, x):
        """Space-to-Depth downsampling: [B, N, C] → [B, N/4, 4C]."""
        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = N // H

        x = x.view(B, H, W, C)

        if H % 2 != 0 or W % 2 != 0:
            x = x[:, :H - 1, :W - 1, :]
            H, W = H - 1, W - 1

        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H // 2, W // 2, 4 * C)

        return x.view(B, -1, 4 * C)

    def encode(self, x):
        """Encode images to latent space.

        Args:
            x: Input images [B, 3, H, W], ImageNet-normalized.

        Returns:
            z_map: Latent feature map [B, latent_dim, h, w].
            last_attn_topo: Attention map from last MUSE block [B, heads, N, N].
            last_feat_itc: Semantic features from last MUSE block [B, N, C].
        """
        vit_embeds = self.encoder.embeddings(x)
        for layer in self.encoder.encoder.layers:
            vit_embeds = layer(vit_embeds)
        spatial_tokens = vit_embeds[:, 1:, :].contiguous().float()

        # Space-to-Depth downsampling
        feat = self.pixel_unshuffle(spatial_tokens)

        # Adapter: project 4*C → 512
        feat = self.adapter_mlp(feat)

        # RoPE Prep
        seq_len = feat.shape[1]
        if self.rope_cos.device != feat.device:
            self.rope_cos = self.rope_cos.to(feat.device)
            self.rope_sin = self.rope_sin.to(feat.device)
        cur_cos = self.rope_cos[:seq_len]
        cur_sin = self.rope_sin[:seq_len]

        current_feat = feat
        last_attn_topo = None
        last_feat_itc = None

        for block in self.muse_blocks:
            current_feat, attn_map, v_raw_itc = block(current_feat, cur_cos, cur_sin)
            last_attn_topo = attn_map
            last_feat_itc = v_raw_itc

        # Reconstruction Flow (with gradient scaling)
        feat_for_rec = spectral_filter(current_feat, scale=self.grad_scale)
        z_quantized = self.to_latent(feat_for_rec)

        z_quantized = z_quantized.permute(0, 2, 1).contiguous()
        b, c, hw = z_quantized.shape
        h_latent = int(math.sqrt(hw))
        z_map = z_quantized.view(b, c, h_latent, h_latent)

        return z_map, last_attn_topo, last_feat_itc

    def decode(self, z):
        """Decode latent features to pixel space.

        Args:
            z: Latent feature map [B, latent_dim, h, w].

        Returns:
            Reconstructed image [B, 3, 224, 224] in tanh range [-1, 1].
        """
        z_adapted = self.decoder_adapter(z)
        dec = self.decoder(z_adapted)
        dec = F.interpolate(dec, size=(224, 224), mode='bilinear', align_corners=False)
        return dec

    def forward(self, x, text_embeddings: Optional[torch.Tensor] = None):
        """Full forward pass.

        Args:
            x: Input images [B, 3, H, W], ImageNet-normalized.
            text_embeddings: Optional text embeddings for ITC [B, D_text].

        Returns:
            reconstruction: Reconstructed images [B, 3, 224, 224].
            result_dict: Dictionary with topology/semantic outputs for loss computation.
        """
        z_map, attn_topo, feat_itc = self.encode(x)
        reconstruction = self.decode(z_map)

        if not self.training:
            return reconstruction, {}

        result_dict = {}

        # A. Topology (Structure Alignment)
        if self.use_dino_topo:
            t_attn = self.get_dino_attention(x)
            if t_attn.shape[-1] != attn_topo.shape[-1]:
                N_student = attn_topo.shape[-1]
                target_size = int(math.sqrt(N_student))
                t_attn = resize_attention_map_4d(t_attn, target_size, target_size)

            result_dict['student_attn'] = attn_topo
            result_dict['teacher_attn'] = t_attn.detach()

        # B. Semantics (ITC Contrastive Learning)
        if self.use_active_itc:
            global_sem = feat_itc.mean(dim=1)
            image_embeds = self.semantic_projector(global_sem)
            image_embeds = F.normalize(image_embeds, dim=-1)
            result_dict['image_embeds'] = image_embeds
            result_dict['logit_scale'] = self.logit_scale.exp()
            if text_embeddings is not None:
                result_dict['text_embeds'] = F.normalize(text_embeddings, dim=-1)

        return reconstruction, result_dict

    def compute_muse_loss(self, images, reconstruction, result_dict):
        """Compute raw MUSE loss components (without weighting).

        Args:
            images: Original images.
            reconstruction: Reconstructed images.
            result_dict: Dictionary from forward() containing attention/embedding outputs.

        Returns:
            Dictionary of individual loss components.
        """
        losses = {}

        # 1. Reconstruction Loss
        losses['loss_rec'] = F.mse_loss(reconstruction, images)

        # 2. Topology Loss (KL Divergence)
        if 'student_attn' in result_dict:
            s_attn = result_dict['student_attn'].mean(dim=1)
            t_attn = result_dict['teacher_attn']
            kl_val = F.kl_div(torch.log(s_attn + 1e-8), t_attn, reduction='batchmean')
            num_tokens = s_attn.shape[-1]
            losses['loss_topo'] = kl_val / num_tokens

        # 3. ITC Loss (Cross-Entropy)
        if 'image_embeds' in result_dict and 'text_embeds' in result_dict:
            image_feats = result_dict['image_embeds']
            text_feats = result_dict['text_embeds']
            logit_scale = result_dict['logit_scale']
            logits_per_image = logit_scale * image_feats @ text_feats.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(image_feats.shape[0], device=image_feats.device)
            loss_itc = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
            losses['loss_itc'] = loss_itc

        return losses
