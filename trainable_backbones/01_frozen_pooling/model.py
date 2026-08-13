#!/usr/bin/env python3
"""Architecture 1: frozen encoder -> fixed hierarchical pooling -> shallow 12-label head.

The plan's reference model, and the thing every later architecture has to beat. There is
no learned aggregation anywhere: slices are reduced to a series descriptor by fixed
statistics, series are grouped by imaging plane, and only the final head is trained.

The pooling statistics are `[mean, p90, max]` together rather than one of them, because
the findings genuinely want different things — a joint effusion is diffuse and shows up
in the mean, while a fracture line or a collateral-ligament tear is focal and only shows
up in the extremes. Letting the head choose per finding is nearly free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PLANES = ["Sagittal", "Coronal", "Axial"]
STATS = ["mean", "p90", "max"]


def pool_slices(x: torch.Tensor) -> torch.Tensor:
    """[n_slices, dim] -> [3 * dim] as mean/p90/max concatenated."""
    if x.numel() == 0:
        return torch.zeros(3 * 0)
    mean = x.mean(0)
    p90 = torch.quantile(x.float(), 0.9, dim=0).to(x.dtype)
    mx = x.max(0).values
    return torch.cat([mean, p90, mx])


class StudyDescriptor:
    """Turns one study's slice features into a fixed-length vector.

    Hierarchy: slices -> series (mean/p90/max) -> plane (mean and max over that plane's
    series) -> study (planes concatenated in a fixed order). A plane the study does not
    have contributes zeros, and a mask marks it, so the head can tell "absent" from
    "present but unremarkable".
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.out_dim = len(PLANES) * 2 * 3 * dim + len(PLANES)

    def __call__(self, series_feats: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
        by_plane: dict[str, list[torch.Tensor]] = {p: [] for p in PLANES}
        for plane, feats in series_feats:
            if plane in by_plane and feats.shape[0] > 0:
                by_plane[plane].append(pool_slices(feats))

        chunks, mask = [], []
        for plane in PLANES:
            got = by_plane[plane]
            if got:
                stacked = torch.stack(got)
                chunks += [stacked.mean(0), stacked.max(0).values]
                mask.append(1.0)
            else:
                zeros = torch.zeros(3 * self.dim)
                chunks += [zeros, zeros]
                mask.append(0.0)
        return torch.cat(chunks + [torch.tensor(mask)])


class Head(nn.Module):
    """Shallow MLP to twelve independent logits.

    Deliberately small: there are ~3,500 training studies and the descriptor is already
    high-dimensional, so capacity here buys overfitting rather than accuracy.
    """

    def __init__(self, in_dim: int, n_labels: int = 12, hidden: int = 0, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        if hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, n_labels),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_labels))

    def forward(self, x):
        return self.net(self.norm(x))
