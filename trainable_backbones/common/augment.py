#!/usr/bin/env python3
"""Feature-level augmentation, shared by every architecture that trains on cached features.

Lives in `common/` rather than in one architecture's directory so the others do not have
to import across sibling packages — which additionally caused `model` to resolve to the
wrong file, since both architectures define one.

The plan ranks these structural perturbations above pixel transforms: dropping a series
or shifting which slices are read reproduces real protocol variation, whereas Gaussian
noise on an embedding corresponds to nothing physical.
"""

from __future__ import annotations

import torch


def augment(series_feats, rng, slice_keep=1.0, slice_dropout=0.0,
            series_dropout=0.0, noise=0.0):
    """Perturb one study's list of (plane, slice_features) pairs.

    A series is dropped only if one survives — losing the whole study is not a variation
    the model should be asked to handle. Slice dropout removes a contiguous block rather
    than scattered slices, because real acquisitions lose coverage in runs.
    """
    out = []
    for plane, feats in series_feats:
        n = feats.shape[0]
        if n == 0:
            continue
        if slice_keep < 1.0:
            k = max(3, int(round(n * slice_keep)))
            start = int(rng.integers(0, max(1, n - k + 1)))
            feats = feats[start:start + k]
            n = feats.shape[0]
        if slice_dropout > 0 and n > 6:
            drop = int(round(n * slice_dropout))
            if drop:
                start = int(rng.integers(0, n - drop))
                feats = torch.cat([feats[:start], feats[start + drop:]])
        out.append((plane, feats))

    if series_dropout > 0 and len(out) > 1:
        keep = [s for s in out if rng.random() > series_dropout]
        out = keep if keep else [out[int(rng.integers(0, len(out)))]]

    if noise > 0:
        out = [(p, f + torch.randn_like(f) * noise) for p, f in out]
    return out
