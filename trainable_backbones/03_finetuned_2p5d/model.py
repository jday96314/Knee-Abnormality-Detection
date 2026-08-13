#!/usr/bin/env python3
"""Architecture 3: partially fine-tuned 2.5D encoder + hybrid pooling.

The plan's primary candidate for a production model. Three changes from architecture 1:

  1. the encoder is unfrozen from the top down, rather than frozen entirely
  2. through-plane context via a Conv1D over adjacent *slice embeddings* (the plan's
     variant B), which preserves the encoder's original single-slice input semantics
     rather than repurposing its RGB channels as neighbouring slices
  3. a hybrid series representation: the fixed [mean, p90, max] statistics *plus* a small
     learned attention pool, so the validated prior is kept as a residual path rather
     than something the learned pooling has to rediscover

The study-level aggregation is a switch rather than a given. The plan says to keep
architecture 2's twelve pathology queries, but measured here those lost to fixed plane
pooling by more than the whole sweep's spread, so both are available and compared.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frozen_backbones"))
import backbones as fb  # noqa: E402

PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE_INDEX = {p: i for i, p in enumerate(PLANES)}


class TrainableMRICore(nn.Module):
    """MRI-CORE ViT-B/16 with the top `unfreeze` blocks trainable.

    Loaded through the same remapping as the frozen probe, so the weights are identical;
    the differences are that gradients are allowed and the module is left in float32,
    since bfloat16 master weights and Adam do not mix well at this scale.

    Partial unfreezing is the point: there are ~600k slices but only ~4,000 independent
    supervisory events, so a large pretrained encoder with a small trainable surface is
    the right ratio. Unfreezing everything would be fitting 86M parameters to 3,479 labels.
    """

    def __init__(self, checkpoint: Path, unfreeze: int = 4, device: str = "cuda"):
        super().__init__()
        import timm

        model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0,
            img_size=224, dynamic_img_size=True,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["teacher"]
        remapped = {}
        for key, value in state.items():
            if not key.startswith("backbone.") or "mask_token" in key:
                continue
            key = key.removeprefix("backbone.")
            key = re.sub(r"^blocks\.\d+\.(\d+)\.", r"blocks.\1.", key)
            remapped[key] = value
        model.load_state_dict(remapped, strict=True)

        for p in model.parameters():
            p.requires_grad_(False)
        blocks = list(model.blocks)
        for block in blocks[len(blocks) - unfreeze:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for p in model.norm.parameters():
            p.requires_grad_(True)

        self.model = model.to(device)
        self.dim = model.embed_dim
        self.unfrozen = unfreeze

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[N, 1, H, W] grayscale -> [N, dim] CLS embeddings."""
        x = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
        return self.model.forward_features(x)[:, 0]


class SliceContext(nn.Module):
    """Conv1D over adjacent slice embeddings within a series (the plan's variant B).

    Local through-plane context without touching the encoder's input semantics: a finding
    that spans three slices becomes visible to the pooling without the encoder ever having
    to interpret channel 0 and channel 2 as different anatomy.
    """

    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):                       # [B, S, D]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)                 # residual: context is additive, not a replacement


class HybridPool(nn.Module):
    """Fixed [mean, p90, max] concatenated with a learned attention pool.

    The fixed path is kept because the frozen sweeps showed it is a strong prior; the
    learned path exists to add whatever that prior misses. Keeping both as a concatenation
    rather than replacing one with the other is the plan's synthesis, and it means the
    learned pooling starts from a position that cannot be worse than fixed pooling alone.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.out_dim = 4 * dim

    def forward(self, x, mask):                 # [B, S, D], [B, S]
        m = mask.unsqueeze(-1).float()
        counts = m.sum(1).clamp(min=1)
        mean = (x * m).sum(1) / counts
        filled = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        mx = filled.max(1).values
        mx = torch.nan_to_num(mx, neginf=0.0)
        q = torch.nan_to_num(
            torch.quantile(x.masked_fill(~mask.unsqueeze(-1), float("nan")).float(),
                           0.9, dim=1, interpolation="linear"), nan=0.0).to(x.dtype)
        weights = self.score(x).masked_fill(~mask.unsqueeze(-1), float("-inf")).softmax(1)
        attended = (x * weights).sum(1)
        return torch.cat([mean, q, mx, attended], dim=-1)


class StudyHead(nn.Module):
    """Series descriptors -> 12 logits, by either fixed plane pooling or query attention."""

    def __init__(self, in_dim: int, mode: str = "plane", d_model: int = 256,
                 n_labels: int = 12, dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        self.project = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model), nn.GELU())
        self.plane_emb = nn.Embedding(len(PLANES) + 1, d_model)
        if mode == "query":
            self.queries = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
            self.attn = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
            self.out = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
            self.bias = nn.Parameter(torch.zeros(n_labels))
        else:
            # Fixed per-plane mean/max, which beat learned attention in architecture 2.
            self.head = nn.Sequential(
                nn.LayerNorm(2 * len(PLANES) * d_model), nn.Dropout(dropout),
                nn.Linear(2 * len(PLANES) * d_model, n_labels),
            )

    def forward(self, series, plane_idx, mask):
        tokens = self.project(series) + self.plane_emb(plane_idx)
        if self.mode == "query":
            q = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            attended, _ = self.attn(q, tokens, tokens, key_padding_mask=~mask)
            return (attended * self.out.unsqueeze(0)).sum(-1) + self.bias
        chunks = []
        for p in range(len(PLANES)):
            sel = mask & (plane_idx == p)
            m = sel.unsqueeze(-1).float()
            chunks.append((tokens * m).sum(1) / m.sum(1).clamp(min=1))
            chunks.append(torch.nan_to_num(
                tokens.masked_fill(~sel.unsqueeze(-1), float("-inf")).max(1).values, neginf=0.0))
        return self.head(torch.cat(chunks, dim=-1))


class Model2p5D(nn.Module):
    def __init__(self, checkpoint: Path, unfreeze: int = 4, mode: str = "plane",
                 d_model: int = 256, dropout: float = 0.1, kernel: int = 3, device="cuda"):
        super().__init__()
        self.encoder = TrainableMRICore(checkpoint, unfreeze, device)
        self.context = SliceContext(self.encoder.dim, kernel)
        self.pool = HybridPool(self.encoder.dim)
        self.head = StudyHead(self.pool.out_dim, mode, d_model, dropout=dropout)

    def forward(self, images, slice_mask, series_of_slice, plane_idx, series_mask):
        """images [B, S, K, 1, H, W] flattened by the caller; see train.py for shapes."""
        raise NotImplementedError("driven by train.py, which batches series explicitly")
