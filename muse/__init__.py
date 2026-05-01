"""MUSE: Multi-granularity Unified Semantic Embedding Tokenizer.

A novel image tokenizer that jointly optimizes pixel-level reconstruction,
topological structure alignment, and semantic understanding through a
three-stage training pipeline.
"""

__version__ = "1.0.0"

from .models.muse_vit import MUSE_ViT
from .models.base_model import BaseModel
from .models.ema_model import EMAModel
