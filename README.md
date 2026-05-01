<div align="center">

# MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality

<p>
  <a href="#"><img alt="ICML 2026" src="https://img.shields.io/badge/ICML-2026-blue.svg"/></a>
  <a href="#"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2506.xxxxx-b31b1b.svg"/></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache%202.0-green.svg"/></a>
  <a href="#"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg"/></a>
  <a href="#"><img alt="PyTorch 2.0+" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg"/></a>
</p>

**Breaking the Visual Tokenization Trade-off — Generation ∧ Understanding, Not Generation ∨ Understanding.**

[Paper](#) · [Project Page](#) · [Model Zoo](#model-zoo) · [Quick Start](#quick-start)

</div>

<div align="center">
  <img src="asst/Figure1.pdf" width="95%"/>
</div>

> **TL;DR**: Unified visual tokenizers suffer from *Manifold Misalignment* — pixel gradients and semantic gradients destructively interfere. MUSE resolves this via **Topological Orthogonality**, physically decoupling structure into attention topology and semantics into feature values. Result: **gFID 3.08** (matching generation specialists) + **Linear Probe 85.2%** (surpassing its own teacher InternViT-300M at 82.5%).

---

## Highlights

<table>
<tr>
<td width="50%">

### 🎯 Mutual Reinforcement, Not Trade-off

Unlike prior unified tokenizers trapped in a zero-sum game, MUSE achieves **genuine synergy** — structurally aligned reconstruction *actively refines* semantic perception.

| Metric | MUSE | Best Prior |
|:--|:--:|:--:|
| gFID ↓ | **3.08** | 3.08 (VTP) |
| Zero-Shot Acc ↑ | **77.1%** | 75.7% (UniLIP) |
| Linear Probe ↑ | **85.2%** | 82.5% (Teacher) |
| Seg. mIoU ↑ | **46.5** | 36.8 (UniLIP) |
| MMVP ↑ | **74.8** | 72.7 (UniLIP) |

</td>
<td width="50%">

### 🧠 Key Insight: Gradient Orthogonality

<img src="asst/Figure3_v3.pdf" width="100%"/>

Semantic gradients naturally occupy **W_V** while structural gradients cluster in **W_Q, W_K**. MUSE respects this inductive bias, eliminating destructive interference.

</td>
</tr>
</table>

---

## Method

### Manifold Misalignment & Topological Orthogonality

<div align="center">
  <img src="asst/Figure2.pdf" width="95%"/>
</div>

<br>

The core challenge: pixel reconstruction wants to *unfold* the latent manifold for detail, while semantic alignment wants to *collapse* it for invariance. Naively combining them causes **destructive gradient interference**.

**MUSE** resolves this via the **Synergistic Block**, which physically decouples the two objectives:

- **Topology Stream** (`W_Q`, `W_K`) → structural gradients refine the attention routing graph
- **Semantic Stream** (`W_V`) → semantic gradients update feature content values
- **Stop-Gradient (`//`)** isolates the semantic branch from reconstruction gradients

This transforms interference into **mutual reinforcement** — a single architecture, two orthogonal optimization subspaces.

### Three-Stage Progressive Training

MUSE follows an information-theoretic curriculum: structure first, then semantics, then synergy.

| Stage | Name | What Learns | What's Frozen | Key Objective |
|:-----:|:-----|:------------|:--------------|:--------------|
| **1** | Topology Warmup | Connector (`W_Q`, `W_K`) | Encoder + Semantic Proj. | `L_topo`: align attention topology with DINO teacher |
| **2** | Semantic Injection | Connector (`W_V`) + Proj. | Encoder | `L_ITC`: anchor feature values to CLIP manifold |
| **3** | Synergistic Tuning | Full model | DINO teacher only | All losses: `L_rec` + `L_topo` + `L_ITC` + `L_GAN` |

---

## Results

### Tokenizer Comparison

MUSE breaks the generative-semantic trade-off, establishing a new **Pareto frontier**:

<div align="center">

| Method | Type | rFID ↓ | gFID ↓ | ZS Acc ↑ | LP Acc ↑ | mIoU ↑ |
|:-------|:----:|:------:|:------:|:--------:|:--------:|:------:|
| VQGAN | Gen. | 1.28 | 5.20 | — | — | 15.4 |
| VA-VAE | Gen. | **0.46** | 3.92 | — | — | 18.5 |
| UniLIP | Unified | 0.74 | 3.62 | 75.7 | 83.6 | 36.8 |
| VTP | Unified | 0.73 | **3.08** | 71.2 | 81.4 | 32.1 |
| **MUSE** | **Unified** | 0.62 | **3.08** | **77.1** | **85.2** | **46.5** |

</div>

### Unified Multimodal Model (UMM)

When integrated into a full UMM pipeline, MUSE enables high-quality generation and editing **without** compromising perception:

<div align="center">

| Model | MMB ↑ | MMVP ↑ | GenEval ↑ | WISE ↑ | Edit Bkg. ↑ |
|:------|:-----:|:------:|:---------:|:------:|:-----------:|
| InternVL3 (specialist) | 78.2 | 72.7 | — | — | — |
| FLUX.1-dev (specialist) | — | — | 0.76 | 0.50 | — |
| UniLIP | 72.6 | 72.7 | 0.78 | 0.62 | 0.79 |
| **MUSE** | **73.4** | **74.8** | **0.82** | **0.65** | **0.87** |

</div>

### Qualitative Results

<details>
<summary><b>Attention Maps — MUSE vs. Baselines</b></summary>

<div align="center">
  <img src="asst/Appendix_Figure2.pdf" width="80%"/>
</div>

MUSE faithfully mirrors the precise, ground-truth-like attention patterns of the DINO teacher, while VQGAN scatters across textures and UniLIP produces overly diffuse maps.

</details>

<details>
<summary><b>Text-to-Image Generation</b></summary>

<div align="center">
  <img src="asst/Appendix_T2I.pdf" width="90%"/>
</div>

Complex attribute binding, accurate spatial reasoning, and realistic textures across diverse prompts.

</details>

<details>
<summary><b>Image Editing</b></summary>

<div align="center">
  <img src="asst/Appendix_Figure3.pdf" width="90%"/>
</div>

Localized semantic modifications while strictly maintaining global layout and background consistency.

</details>

---

## Model Zoo

| Model | Backbone | Params | gFID ↓ | LP Acc ↑ | Checkpoint |
|:------|:---------|:------:|:------:|:--------:|:----------:|
| MUSE-1B | InternVL3-1B + SANA-0.6B | 496M | 3.08 | 85.2 | Coming Soon |
| MUSE-3B | InternVL3-2B + SANA-1.6B | — | — | — | Coming Soon |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/MUSE.git
cd MUSE

# Create conda environment (recommended)
conda create -n muse python=3.10 -y
conda activate muse

# Install dependencies
pip install -e .
# Or:
pip install -r requirements.txt
```

### Prerequisites

Download the following pretrained models:

| Model | Role | Source |
|:------|:-----|:-------|
| InternVL3-1B / 2B | Vision backbone encoder | [HuggingFace](https://huggingface.co/OpenGVLab/InternVL3-1B) |
| DC-AE (SANA) | Pixel decoder | [HuggingFace](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) |
| DINOv3-ViT-H+ | Topology teacher (Stage 1) | Custom checkpoint |
| CLIP-ViT-L-14 | Text encoder for ITC (Stage 2+) | [OpenCLIP](https://github.com/mlfoundations/open_clip) |

---

## Quick Start

### Training the Tokenizer

```bash
export CHECKPOINT_DIR=/path/to/pretrained/models
export DATA_DIR=/path/to/datasets

# Stage 1: Topology Warmup — align attention with DINO teacher
bash tools/train_stage1.sh muse_1b

# Stage 2: Semantic Injection — anchor features to CLIP manifold
bash tools/train_stage2.sh muse_1b

# Stage 3: Synergistic Tuning — full co-optimization
bash tools/train_stage3.sh muse_1b
```

### Evaluation

```bash
# Reconstruction metrics (rFID, PSNR, SSIM, LPIPS)
bash tools/evaluate.sh \
    configs/muse_1b/stage3.yaml \
    /path/to/checkpoint.bin

# Linear probe on ImageNet-1K
python scripts/linear_probe.py \
    --config configs/muse_1b/stage3.yaml \
    --checkpoint /path/to/checkpoint.bin

# Zero-shot ImageNet classification
python scripts/zero_shot.py \
    --config configs/muse_1b/stage3.yaml \
    --checkpoint /path/to/checkpoint.bin

# ADE20K segmentation probe (mIoU)
bash tools/segment_probe.sh muse \
    --config configs/muse_1b/stage3.yaml \
    --checkpoint /path/to/checkpoint.bin \
    --train-url "/path/to/ade20k-train-{000000..000020}.tar" \
    --val-url "/path/to/ade20k-validation-{000000..000002}.tar"
```

### Inference

```bash
# Single image reconstruction
python scripts/inference.py \
    --config configs/muse_1b/stage3.yaml \
    --checkpoint /path/to/checkpoint.bin \
    --image_path /path/to/image.jpg \
    --output_dir outputs/inference

# Attention map visualization
bash tools/visualize_attention.sh single \
    --config configs/muse_1b/stage3.yaml \
    --checkpoint /path/to/checkpoint.bin \
    --image /path/to/image.jpg \
    --output outputs/attention_viz
```

---

## Data Format

MUSE uses [WebDataset](https://github.com/webdataset/webdataset) (`.tar`) format for scalable data loading:

```
shard-000000.tar
├── 00000.jpg          # Image
├── 00000.txt          # Caption (Stages 2–3)
├── 00001.jpg
├── 00001.txt
└── ...
```

For segmentation probing, each shard additionally contains `*.seg.png` (ADE20K labels).

---

## Project Structure

```
MUSE/
├── muse/                              # Core library
│   ├── models/
│   │   ├── muse_vit.py                #   MUSE_ViT + SynergisticBlock
│   │   ├── base_model.py              #   Save/load utilities
│   │   ├── ema_model.py               #   EMA model wrapper
│   │   ├── discriminator.py           #   PatchGAN discriminator
│   │   ├── lpips.py                   #   LPIPS perceptual metric
│   │   └── perceptual_loss.py         #   LPIPS + ConvNeXt-S perceptual
│   ├── losses/
│   │   └── muse_loss.py               #   Pixel + Perceptual + GAN + Topo + ITC
│   ├── data/
│   │   └── dataloader.py              #   WebDataset loader
│   ├── evaluation/
│   │   ├── evaluator.py               #   rFID / PSNR / SSIM / LPIPS
│   │   └── inception.py               #   InceptionV3 for FID
│   └── utils/
│       ├── viz_utils.py               #   Attention visualization pipeline
│       ├── train_utils.py             #   Training helpers
│       ├── lr_schedulers.py           #   LR schedule (cosine / constant)
│       └── logger.py                  #   Logging setup
├── scripts/                           # Entry-point scripts
│   ├── train_stage{1,2,3}.py          #   Three-stage training
│   ├── evaluate.py                    #   Batch reconstruction eval
│   ├── inference.py                   #   Single-image reconstruction
│   ├── linear_probe.py               #   ImageNet linear probe
│   ├── zero_shot.py                   #   Zero-shot classification
│   ├── zero_shot_meta.py              #   ImageNet class names + templates
│   ├── segment_probe.py              #   ADE20K segmentation probe
│   └── visualize_attention.py         #   Attention map visualization
├── configs/
│   ├── muse_1b/                       #   MUSE-1B configs (stage1–3)
│   └── muse_3b/                       #   MUSE-3B configs (stage1–3)
├── tools/                             # Shell launch scripts
│   ├── train_stage{1,2,3}.sh
│   ├── evaluate.sh
│   ├── visualize_attention.sh
│   └── segment_probe.sh
├── asst/                              # Paper figures & assets
├── requirements.txt
├── setup.py
└── LICENSE                            # Apache 2.0
```

---

## Citation

If you find MUSE useful in your research, please consider citing:

```bibtex
@inproceedings{muse2026,
  title     = {MUSE: Resolving Manifold Misalignment in Visual Tokenization 
               via Topological Orthogonality},
  author    = {Anonymous},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

MUSE builds upon several excellent open-source projects:

- [InternVL3](https://github.com/OpenGVLab/InternVL) — Vision backbone
- [SANA](https://github.com/NVlabs/Sana) / [DC-AE](https://github.com/mit-han-lab/efficientvit) — Pixel decoder
- [DINOv3](https://github.com/facebookresearch/dinov3) — Structural topology teacher
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — Text encoder for semantic anchoring

## License

This project is licensed under the [Apache License 2.0](LICENSE).
