#!/usr/bin/env python3
"""Architecture 2: frozen slice pooling -> learned pathology attention over series.

Keeps architecture 1's fixed `[mean, p90, max]` slice pooling — which the frozen sweep
already validated — and replaces only the study-level aggregation. Twelve learned query
tokens, one per finding, cross-attend over the study's 3-15 series.

The reason to change *this* part rather than the slice pooling is statistical. Attention
over ~5 series in the median study is a far easier estimation problem than attention over
~180 individual slices, and it buys the thing fixed pooling structurally cannot do: each
finding can look at a different sequence. An ACL tear wants the sagittal fluid-sensitive
series; a collateral ligament wants the coronal; a Baker's cyst wants the axial. Fixed
pooling has to average that choice away.

Series metadata (plane, fluid sensitivity, fat suppression) is embedded and added to the
series token, so a query can attend by protocol rather than having to infer it from the
image features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE_INDEX = {p: i for i, p in enumerate(PLANES)}


class SeriesEncoder(nn.Module):
    """One series -> one token: fixed pooling, projection, plus protocol embeddings."""

    def __init__(self, feat_dim: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(3 * feat_dim),
            nn.Linear(3 * feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.plane = nn.Embedding(len(PLANES) + 1, d_model)   # +1 for unknown
        self.fluid = nn.Embedding(2, d_model)
        self.fatsat = nn.Embedding(2, d_model)

    def forward(self, pooled, plane_idx, fluid, fatsat):
        return self.project(pooled) + self.plane(plane_idx) + self.fluid(fluid) + self.fatsat(fatsat)


class PathologyAttention(nn.Module):
    """Twelve queries cross-attending over the series tokens of one study.

    Each query owns its output head, so a finding's logit depends only on what that
    finding attended to. A shared head would let one finding's evidence leak into
    another's prediction through the projection, which is exactly the interference this
    architecture exists to avoid.
    """

    def __init__(self, d_model: int = 256, n_labels: int = 12, n_heads: int = 4,
                 dropout: float = 0.1, n_layers: int = 1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "norm_q": nn.LayerNorm(d_model),
                "norm_kv": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(2 * d_model, d_model),
                ),
            })
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Parameter(torch.zeros(n_labels, d_model))
        self.bias = nn.Parameter(torch.zeros(n_labels))
        nn.init.normal_(self.out, std=0.02)

    def forward(self, series_tokens, key_padding_mask, return_attention=False):
        batch = series_tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attn_weights = None
        for layer in self.layers:
            attended, attn_weights = layer["attn"](
                layer["norm_q"](q), layer["norm_kv"](series_tokens),
                layer["norm_kv"](series_tokens),
                key_padding_mask=key_padding_mask, need_weights=return_attention,
            )
            q = q + attended
            q = q + layer["ff"](q)
        q = self.out_norm(q)
        logits = (q * self.out.unsqueeze(0)).sum(-1) + self.bias
        return (logits, attn_weights) if return_attention else logits


class SeriesAttentionModel(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 256, n_labels: int = 12,
                 n_heads: int = 4, dropout: float = 0.1, n_layers: int = 1):
        super().__init__()
        self.encoder = SeriesEncoder(feat_dim, d_model, dropout)
        self.attention = PathologyAttention(d_model, n_labels, n_heads, dropout, n_layers)

    def forward(self, pooled, plane_idx, fluid, fatsat, mask, return_attention=False):
        tokens = self.encoder(pooled, plane_idx, fluid, fatsat)
        return self.attention(tokens, key_padding_mask=~mask, return_attention=return_attention)
