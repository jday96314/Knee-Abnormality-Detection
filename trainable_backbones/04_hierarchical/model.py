#!/usr/bin/env python3
"""Architecture 4: full hierarchical slice-Transformer + series-Transformer.

The plan's most elaborate aggregator: a small Transformer over slice embeddings inside a
series, then a second over series inside a study, with twelve pathology queries at the top.

It is built here on **frozen** features rather than on architecture 3's fine-tuned encoder,
which is a deliberate deviation. The single question this architecture asks is whether
Transformer-based aggregation beats fixed pooling; architectures 1 and 2 answered adjacent
questions on frozen features with the same split and the same augmentations, so building
#4 the same way makes it the third point on one controlled curve instead of a new system
that differs in two ways at once. It is also what allows every slice of every series to be
used — the fine-tuned path can only afford a sample.

Two structural commitments from the plan are kept because they encode findings rather than
guesses:

  * **the fixed-pooling residual path.** `[mean, p90, max]` is computed from the projected
    inputs and concatenated with the Transformer's series token, so the aggregator starts
    from the prior architecture 1 validated instead of having to rediscover it.
  * **slice position.** Slice index within the series is embedded, which is the one piece of
    structure fixed pooling throws away and the reason a slice Transformer could win at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE_INDEX = {p: i for i, p in enumerate(PLANES)}


def masked_stats(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[mean, p90, max] over dim 1, honouring `mask`. The architecture-1 prior."""
    m = mask.unsqueeze(-1)
    mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
    mx = torch.nan_to_num(x.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values,
                          neginf=0.0)
    q = torch.nan_to_num(torch.quantile(
        x.masked_fill(~mask.unsqueeze(-1), float("nan")).float(), 0.9, dim=1), nan=0.0)
    return torch.cat([mean, q.to(x.dtype), mx], dim=-1)


class SliceTransformer(nn.Module):
    """Slices of one series -> one series vector.

    Output is `[CLS | mean | p90 | max]`: the learned token and the fixed prior side by
    side, so a degenerate Transformer costs accuracy only through the extra parameters,
    not by destroying the signal fixed pooling already extracts.
    """

    def __init__(self, in_dim: int, d_model: int = 256, n_layers: int = 1,
                 n_heads: int = 4, dropout: float = 0.1, max_slices: int = 512):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Absolute index, not physical z: slice spacing is not in the cache index, and a
        # learned index embedding already gives the ordering that fixed pooling lacks.
        self.pos = nn.Embedding(max_slices + 1, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=dropout,
                                       batch_first=True, norm_first=True),
            num_layers=n_layers)
        self.out_dim = 4 * d_model

    def forward(self, feats, mask):                 # [N, K, in_dim], [N, K]
        x = self.project(feats)
        idx = torch.arange(1, x.shape[1] + 1, device=x.device).clamp(max=self.pos.num_embeddings - 1)
        x = x + self.pos(idx).unsqueeze(0)
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        pad = torch.cat([torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device),
                         mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~pad)
        return torch.cat([encoded[:, 0], masked_stats(x, mask)], dim=-1)


class HierarchicalModel(nn.Module):
    """Series vectors -> 12 logits, via a series Transformer and a study-level head."""

    def __init__(self, in_dim: int, d_model: int = 256, slice_layers: int = 1,
                 series_layers: int = 1, n_heads: int = 4, dropout: float = 0.1,
                 n_labels: int = 12, mode: str = "plane"):
        super().__init__()
        self.mode = mode
        self.slice_tf = SliceTransformer(in_dim, d_model, slice_layers, n_heads, dropout)
        self.to_series = nn.Sequential(
            nn.LayerNorm(self.slice_tf.out_dim), nn.Linear(self.slice_tf.out_dim, d_model))
        self.plane_emb = nn.Embedding(len(PLANES) + 1, d_model)
        self.fluid_emb = nn.Embedding(2, d_model)
        self.fatsat_emb = nn.Embedding(2, d_model)
        self.series_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=dropout,
                                       batch_first=True, norm_first=True),
            num_layers=series_layers)
        if mode == "query":
            self.queries = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.out = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
            self.bias = nn.Parameter(torch.zeros(n_labels))
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(3 * d_model), nn.Dropout(dropout), nn.Linear(3 * d_model, n_labels))

    def forward(self, feats, slice_mask, plane, fluid, fatsat, series_mask):
        """feats [B, S, K, in_dim]; slice_mask [B, S, K]; series_mask [B, S]."""
        B, S, K, D = feats.shape
        series = self.slice_tf(feats.reshape(B * S, K, D), slice_mask.reshape(B * S, K))
        tokens = (self.to_series(series).reshape(B, S, -1)
                  + self.plane_emb(plane) + self.fluid_emb(fluid) + self.fatsat_emb(fatsat))
        # A study with a single series would give the encoder an all-padding row; the mask
        # guarantees at least one valid token per study, enforced by the batch builder.
        tokens = self.series_tf(tokens, src_key_padding_mask=~series_mask)
        if self.mode == "query":
            q = self.queries.unsqueeze(0).expand(B, -1, -1)
            attended, _ = self.attn(q, tokens, tokens, key_padding_mask=~series_mask)
            return (attended * self.out.unsqueeze(0)).sum(-1) + self.bias
        return self.head(masked_stats(tokens, series_mask))
