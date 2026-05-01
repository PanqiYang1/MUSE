"""Visualization utilities for reconstruction, generation, and attention analysis."""

import os
import math
import numpy as np

import torch
import torchvision.transforms.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False



def make_viz_from_samples(original_images, reconstructed_images):
    """Generates side-by-side visualization of original vs reconstructed images.

    Creates a grid showing [original | reconstructed | difference] for each image.

    Args:
        original_images: A torch.Tensor of original images in [0, 1] range.
        reconstructed_images: A torch.Tensor of reconstructed images in [0, 1] range.

    Returns:
        A tuple (images_for_saving, images_for_logging):
            - images_for_saving: List of PIL Images.
            - images_for_logging: Tensor of uint8 images.
    """
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    reconstructed_images = reconstructed_images * 255.0
    reconstructed_images = reconstructed_images.cpu()

    original_images = torch.clamp(original_images, 0.0, 1.0)
    original_images *= 255.0
    original_images = original_images.cpu()

    diff_img = torch.abs(original_images - reconstructed_images)
    to_stack = [original_images, reconstructed_images, diff_img]

    images_for_logging = rearrange(
        torch.stack(to_stack),
        "(l1 l2) b c h w -> b c (l1 h) (l2 w)",
        l1=1,
    ).byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging


def make_viz_from_samples_generation(generated_images):
    """Generates a grid visualization from generated images.

    Args:
        generated_images: A torch.Tensor of generated images in [0, 1] range.

    Returns:
        A tuple (image_for_saving, image_for_logging):
            - image_for_saving: A single PIL Image.
            - image_for_logging: Tensor of uint8 image.
    """
    generated = torch.clamp(generated_images, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(
        generated,
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2,
    )

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    return images_for_saving, images_for_logging


def make_viz_from_samples_t2i_generation(generated_images, captions):
    """Generates a grid visualization from text-to-image generated images with captions.

    Args:
        generated_images: A torch.Tensor of generated images in [0, 1] range.
        captions: A list of caption strings.

    Returns:
        A tuple (image_with_captions, image_for_logging):
            - image_with_captions: A PIL Image with captions below.
            - image_for_logging: Tensor of uint8 image (without captions).
    """
    generated = torch.clamp(generated_images, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(
        generated,
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2,
    )

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    # Create a new image with space for captions
    width, height = images_for_saving.size
    caption_height = 20 * len(captions) + 10
    new_height = height + caption_height
    new_image = Image.new("RGB", (width, new_height), "black")
    new_image.paste(images_for_saving, (0, 0))

    # Add captions below the image
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()

    for i, caption in enumerate(captions):
        draw.text((10, height + 10 + i * 20), caption, fill="white", font=font)

    return new_image, images_for_logging


# ==============================================================================
# Attention Map Visualization Utilities
# ==============================================================================

def _require_cv2():
    """Check that OpenCV is available."""
    if not HAS_CV2:
        raise ImportError(
            "OpenCV (cv2) is required for attention visualization. "
            "Install with: pip install opencv-python"
        )


def extract_attention_saliency(
    attn_map: torch.Tensor,
    temperature: float = 0.1,
    remove_diagonal: bool = True,
    top_k_heads: int = 3,
) -> np.ndarray:
    """Extract a 2D saliency map from multi-head attention.

    Core algorithm (from UniLIP):
      1. Optionally remove self-attention (diagonal) to suppress background self-similarity.
      2. Compute per-head "被关注度" (column-mean attention).
      3. Select top-K heads by variance (heads with clear focal points).
      4. Average selected heads, reshape to 2D grid, normalize, and sharpen.

    Args:
        attn_map: Attention tensor of shape [Heads, N, N] for a single sample.
        temperature: Sharpening factor in (0, 1). Lower = sharper.
        remove_diagonal: If True, zero out diagonal (self-attention).
        top_k_heads: Number of highest-variance heads to average.

    Returns:
        A 2D numpy array of shape [grid_h, grid_w] in [0, 1].
    """
    attn = attn_map.detach().cpu().float()
    num_heads, num_tokens, _ = attn.shape

    # Step 1: Remove self-attention (diagonal → 0)
    if remove_diagonal:
        diag_mask = torch.eye(num_tokens, dtype=torch.bool)
        attn.masked_fill_(diag_mask, 0.0)

    # Step 2: Column-mean → "who is looking at me"
    head_saliency = attn.mean(dim=1)  # [Heads, N]

    # Step 3: Select top-K heads by variance (high variance = clear focal point)
    head_variance = head_saliency.var(dim=-1)  # [Heads]
    topk = min(top_k_heads, num_heads)
    _, top_indices = torch.topk(head_variance, topk)

    # Step 4: Average selected heads
    best_saliency = head_saliency[top_indices].mean(dim=0)  # [N]

    # Step 5: Reshape to 2D grid
    grid_size = int(math.sqrt(num_tokens))
    if grid_size ** 2 != num_tokens:
        best_saliency = best_saliency[: grid_size ** 2]

    saliency_2d = best_saliency.view(grid_size, grid_size).numpy()

    # Step 6: Normalize to [0, 1]
    vmin, vmax = saliency_2d.min(), saliency_2d.max()
    saliency_2d = (saliency_2d - vmin) / (vmax - vmin + 1e-8)

    # Step 7: Sharpen
    saliency_2d = saliency_2d ** (1.0 - temperature)

    return saliency_2d


def auto_invert_saliency(saliency_2d: np.ndarray) -> np.ndarray:
    """Auto-invert a saliency map if the background is brighter than the center.

    Compares edge-region mean vs center-region mean. If edges are brighter,
    the map is inverted so the foreground gets the highest values.

    Args:
        saliency_2d: A 2D numpy array in [0, 1].

    Returns:
        Possibly inverted 2D numpy array.
    """
    h, w = saliency_2d.shape
    margin = max(h // 8, 1)

    edge_mean = (
        np.mean(saliency_2d[:margin, :])
        + np.mean(saliency_2d[-margin:, :])
        + np.mean(saliency_2d[:, :margin])
        + np.mean(saliency_2d[:, -margin:])
    ) / 4.0
    center_mean = np.mean(saliency_2d[margin:-margin, margin:-margin])

    if edge_mean > center_mean:
        saliency_2d = 1.0 - saliency_2d

    return saliency_2d


def connected_component_filter(saliency_2d: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """Keep only the hottest connected component via morphological analysis.

    Similar to the PCA-mask approach in DINO visualization:
      1. Binarize the saliency map.
      2. Find connected components.
      3. Keep the component with the highest total attention sum.
      4. Smooth the mask edges.

    Args:
        saliency_2d: A 2D numpy array in [0, 1].
        threshold: Binarization threshold.

    Returns:
        A 2D float mask in [0, 1] retaining only the main foreground island.
    """
    _require_cv2()

    binary = (saliency_2d > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return np.ones_like(saliency_2d)

    best_label, max_heat = -1, -1.0
    for i in range(1, num_labels):
        island_mask = labels == i
        heat = np.sum(saliency_2d[island_mask])
        if heat > max_heat:
            max_heat = heat
            best_label = i

    if best_label == -1:
        return np.ones_like(saliency_2d)

    clean_mask = (labels == best_label).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.GaussianBlur(clean_mask, (0, 0), sigmaX=0.5, sigmaY=0.5)
    return clean_mask


def process_attention_refined(
    attn_map: torch.Tensor,
    temperature: float = 0.1,
    remove_diagonal: bool = True,
    top_k_heads: int = 3,
    use_cc_filter: bool = True,
) -> tuple:
    """Full attention-map post-processing pipeline.

    Pipeline: extract saliency → auto-invert → connected-component filter →
    Gaussian flood-fill → re-normalize inside mask → gamma sharpen.

    Args:
        attn_map: Attention tensor [Heads, N, N] for one sample.
        temperature: Sharpening factor.
        remove_diagonal: Remove self-attention diagonal.
        top_k_heads: Number of top-variance heads to aggregate.
        use_cc_filter: Whether to apply connected-component island filtering.

    Returns:
        (refined_map, mask):
            refined_map: 2D numpy array [grid_h, grid_w] in [0, 1].
            mask: 2D numpy array of the foreground mask (or None if cc_filter disabled).
    """
    # Step 1: Basic saliency extraction
    saliency = extract_attention_saliency(
        attn_map,
        temperature=1.0,  # no sharpening yet; we sharpen later
        remove_diagonal=remove_diagonal,
        top_k_heads=top_k_heads,
    )

    # Step 2: Auto-invert
    saliency = auto_invert_saliency(saliency)

    if not use_cc_filter or not HAS_CV2:
        # Simple path: just sharpen and return
        saliency = saliency ** (1.0 - temperature)
        return saliency, None

    # Step 3: Connected-component filter
    clean_mask = connected_component_filter(saliency, threshold=0.35)

    # Step 4: Gaussian flood-fill (smooth internal cold spots)
    flooded = cv2.GaussianBlur(saliency, (0, 0), sigmaX=2.0, sigmaY=2.0)
    flooded = (flooded - flooded.min()) / (flooded.max() - flooded.min() + 1e-8)

    # Step 5: Fuse with mask
    fused = flooded * clean_mask

    # Step 6: Re-normalize inside mask region
    masked_vals = fused[clean_mask > 0.1]
    if masked_vals.size > 0 and masked_vals.max() > 1e-8:
        fused = fused / masked_vals.max()

    # Step 7: Gamma sharpening
    fused = np.clip(fused, 0, 1) ** (1.0 - 0.4)
    fused = fused * clean_mask

    return fused, clean_mask


# ==============================================================================
# Attention Rendering Utilities
# ==============================================================================

def denormalize_image(
    tensor: torch.Tensor,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """De-normalize an ImageNet-normalized tensor to uint8 BGR (OpenCV format).

    Args:
        tensor: Image tensor [3, H, W] (ImageNet-normalized).
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        uint8 BGR numpy array [H, W, 3].
    """
    _require_cv2()
    m = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    s = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * s + m
    tensor = torch.clamp(tensor, 0, 1)
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img_bgr


def apply_heatmap_overlay(
    img_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha_img: float = 0.45,
    alpha_map: float = 0.55,
    colormap: int = None,
) -> np.ndarray:
    """Overlay a heatmap onto an image using JET colormap.

    Args:
        img_bgr: Original image [H, W, 3] in uint8 BGR.
        heatmap: 2D float array [h, w] in [0, 1].
        alpha_img: Blending weight for original image.
        alpha_map: Blending weight for heatmap.
        colormap: OpenCV colormap ID (default: COLORMAP_JET).

    Returns:
        Blended image [H, W, 3] in uint8 BGR.
    """
    _require_cv2()
    if colormap is None:
        colormap = cv2.COLORMAP_JET

    h, w = img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    blended = cv2.addWeighted(img_bgr, alpha_img, colored, alpha_map, 0)
    return blended


def save_attention_visualization(
    save_dir: str,
    src_img_bgr: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray = None,
) -> None:
    """Save attention visualization outputs to a directory.

    Saves:
      - origin.png: Original image.
      - heatmap.png: Heatmap overlay.
      - mask.png: Foreground mask (if available).
      - combined.png: Side-by-side [origin | heatmap].

    Args:
        save_dir: Directory to save files.
        src_img_bgr: Original image [H, W, 3] uint8 BGR.
        heatmap: 2D float heatmap [h, w] in [0, 1].
        mask: Optional 2D float mask [h, w] in [0, 1].
    """
    _require_cv2()
    os.makedirs(save_dir, exist_ok=True)
    h, w = src_img_bgr.shape[:2]

    # 1. Save original
    cv2.imwrite(os.path.join(save_dir, "origin.png"), src_img_bgr)

    # 2. Save heatmap overlay
    viz_img = apply_heatmap_overlay(src_img_bgr, heatmap)
    cv2.imwrite(os.path.join(save_dir, "heatmap.png"), viz_img)

    # 3. Save mask (if available)
    if mask is not None:
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_vis = (mask_resized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, "mask.png"), mask_vis)

    # 4. Save side-by-side
    combined = np.hstack([src_img_bgr, viz_img])
    cv2.imwrite(os.path.join(save_dir, "combined.png"), combined)


def make_attention_grid_pil(
    original_images: torch.Tensor,
    attn_maps: torch.Tensor,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    temperature: float = 0.1,
    remove_diagonal: bool = True,
    use_cc_filter: bool = True,
) -> Image.Image:
    """Create a PIL grid image: [orig | heatmap] for each sample in a batch.

    Useful for logging to W&B / TensorBoard during training.

    Args:
        original_images: Batch of ImageNet-normalized images [B, 3, H, W].
        attn_maps: Attention maps [B, Heads, N, N].
        mean: Normalization mean.
        std: Normalization std.
        temperature: Sharpening parameter.
        remove_diagonal: Remove self-attention diagonal.
        use_cc_filter: Apply connected-component filtering.

    Returns:
        A PIL Image showing side-by-side original + heatmap for each sample.
    """
    _require_cv2()
    B = original_images.shape[0]
    rows = []

    for i in range(B):
        # De-normalize
        src_bgr = denormalize_image(original_images[i], mean, std)

        # Process attention
        refined, _ = process_attention_refined(
            attn_maps[i],
            temperature=temperature,
            remove_diagonal=remove_diagonal,
            use_cc_filter=use_cc_filter,
        )
        viz_bgr = apply_heatmap_overlay(src_bgr, refined)

        # Combine
        pair = np.hstack([src_bgr, viz_bgr])
        rows.append(pair)

    grid = np.vstack(rows)
    # BGR → RGB for PIL
    grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    return Image.fromarray(grid_rgb)
