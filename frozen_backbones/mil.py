#!/usr/bin/env python3
"""Attention-based MIL over the cached per-slice features, with orientation-aware heads.

Every pooling strategy so far applies the same fixed reduction to every study and
every finding. The hierarchical sweep showed that the *right* reduction is
finding-dependent -- focal labels want a soft-max within series, diffuse ones want
an average -- which is precisely what a learned attention pooling can express and
a fixed one cannot.

Gated attention follows Ilse et al. (2018): a scalar weight per slice from a
tanh/sigmoid gate, softmax-normalized over the bag, used to average the slices.
The model is deliberately tiny (a 1024->64 projection and a 64->32 gate, ~70k
parameters) because a training fold here is 46 bags. It is trained per label, in
the same folds as the linear probes, so the numbers are directly comparable.

Orientation-aware variants condition the attention on where a slice came from:

  plain        one attention over the whole bag; orientation ignored
  plane_embed  a learned embedding per plane added to each slice feature, so the
               gate can score a coronal slice differently from a sagittal one
  plane_attn   a separate attention pooling per plane, concatenated; each plane
               gets its own notion of which slice matters, and the classifier
               sees the three summaries separately
  orient       the series' true slice-normal direction cosines projected and
               added, which keeps obliquity that the three-way plane label bins
               away (this cohort runs to 38 degrees off-axis)
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

import data
import probe

RESULTS = probe.FEATURES.parent / "results"
PLANES = ("Sagittal", "Coronal", "Axial")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Bag assembly
# ---------------------------------------------------------------------------


class Bags:
    """Per-study slice features padded into one [S, N, D] tensor on the GPU.

    Padding rather than ragged batching: the whole cohort is 58 x 572 x 1024
    floats (~136 MB), so the simplest layout fits and every fold is one
    full-batch step.
    """

    def __init__(self, backbone: str, size: int, head: str, central: bool = True,
                 level: str = "slice", inner: str = "p90"):
        heads, meta = probe.load_features(backbone, size)
        features = heads[head].astype(np.float32)
        studies, _ = data.cohort()
        self.ids = studies.StudyInstanceUID.tolist()
        self.labels = studies[data.LABELS]

        orientation = data.series_orientation().set_index("SeriesInstanceUID")
        normals = orientation[["normal0", "normal1", "normal2"]]

        keep = np.ones(len(meta), bool)
        if central:
            # The same trim the fixed poolings benefited from; the ends of a knee
            # stack are off-joint and the attention should not have to learn that
            # from 46 bags.
            keep = (meta["pos"].to_numpy() > 0.25) & (meta["pos"].to_numpy() < 0.75)

        study_of = meta["study"].to_numpy()
        plane_of = meta["plane"].to_numpy()
        series_of = meta["series"].to_numpy()
        plane_index = {p: i for i, p in enumerate(PLANES)}

        if level == "series":
            # Collapse each series first, exactly as the hierarchical sweep found
            # best, so the bag is ~6 series descriptors instead of ~180 slices.
            # 46 bags cannot supervise an attention over 180 items; they can
            # plausibly supervise one over 6.
            collapsed_x, collapsed_meta = [], []
            for uid in self.ids:
                for series_uid in pd.unique(series_of[keep & (study_of == uid)]):
                    rows = np.flatnonzero(keep & (series_of == series_uid))
                    collapsed_x.append(probe._reduce(features[rows], inner))
                    collapsed_meta.append((uid, series_uid, plane_of[rows[0]]))
            features = np.stack(collapsed_x)
            study_of = np.array([m[0] for m in collapsed_meta])
            series_of = np.array([m[1] for m in collapsed_meta])
            plane_of = np.array([m[2] for m in collapsed_meta])
            keep = np.ones(len(features), bool)

        per_study = [np.flatnonzero(keep & (study_of == uid)) for uid in self.ids]
        width = max(len(rows) for rows in per_study)

        self.x = torch.zeros(len(self.ids), width, features.shape[1])
        self.mask = torch.zeros(len(self.ids), width, dtype=torch.bool)
        self.plane = torch.zeros(len(self.ids), width, dtype=torch.long)
        self.normal = torch.zeros(len(self.ids), width, 3)

        for i, rows in enumerate(per_study):
            n = len(rows)
            self.x[i, :n] = torch.from_numpy(features[rows])
            self.mask[i, :n] = True
            self.plane[i, :n] = torch.tensor([plane_index.get(p, 0) for p in plane_of[rows]])
            self.normal[i, :n] = torch.tensor(
                normals.loc[series_of[rows]].to_numpy(dtype=np.float32))

        self.x = self.x.to(DEVICE)
        self.mask = self.mask.to(DEVICE)
        self.plane = self.plane.to(DEVICE)
        self.normal = self.normal.to(DEVICE)
        self.dim = features.shape[1]

    def standardized(self, train_idx: np.ndarray, pca_dim: int = 0) -> torch.Tensor:
        """Z-score, and optionally PCA-reduce, using only training studies.

        The reduction is the difference between a workable model and an
        unworkable one here: a 1024->64 input projection is 65k parameters
        fitted on 46 bags. Reducing to `pca_dim` first cuts that by 16x. PCA is
        unsupervised and fitted on training rows only, so it leaks nothing.
        """
        rows = self.x[train_idx][self.mask[train_idx]]
        mean, std = rows.mean(0), rows.std(0).clamp_min(1e-6)
        x = (self.x - mean) / std
        if not pca_dim:
            return x
        rows = (rows - mean) / std
        _, _, v = torch.pca_lowrank(rows, q=min(pca_dim, rows.shape[0] - 1, rows.shape[1]))
        return x @ v


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the slice axis, with all-empty rows returning all zeros.

    An empty row happens whenever a study has no series in some plane, which
    `plane_attn` hits constantly. Left to itself softmax returns NaN there.
    """
    logits = logits.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(logits, dim=1)
    return torch.nan_to_num(weights, nan=0.0)


class AttentionMIL(nn.Module):
    def __init__(self, dim: int, variant: str = "plain", d_proj: int = 64,
                 d_att: int = 32, dropout: float = 0.3):
        super().__init__()
        self.variant = variant
        self.proj = nn.Sequential(nn.Linear(dim, d_proj), nn.ReLU(), nn.Dropout(dropout))
        if variant == "plane_embed":
            self.plane_emb = nn.Embedding(len(PLANES), d_proj)
        if variant == "orient":
            self.orient = nn.Linear(3, d_proj)

        n_pool = len(PLANES) if variant == "plane_attn" else 1
        self.gate_v = nn.ModuleList(nn.Linear(d_proj, d_att) for _ in range(n_pool))
        self.gate_u = nn.ModuleList(nn.Linear(d_proj, d_att) for _ in range(n_pool))
        self.gate_w = nn.ModuleList(nn.Linear(d_att, 1) for _ in range(n_pool))
        self.head = nn.Linear(d_proj * n_pool, 1)

    def _pool(self, h, mask, index):
        scores = self.gate_w[index](
            torch.tanh(self.gate_v[index](h)) * torch.sigmoid(self.gate_u[index](h)))
        weights = masked_softmax(scores, mask.unsqueeze(-1))
        return (weights * h).sum(1), weights

    def forward(self, x, mask, plane, normal):
        h = self.proj(x)
        if self.variant == "plane_embed":
            h = h + self.plane_emb(plane)
        elif self.variant == "orient":
            h = h + self.orient(normal)

        if self.variant == "plane_attn":
            pooled = [self._pool(h, mask & (plane == p), p)[0] for p in range(len(PLANES))]
            z = torch.cat(pooled, dim=-1)
        else:
            z, _ = self._pool(h, mask, 0)
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_predict(bags: Bags, x: torch.Tensor, train_idx, test_idx, target,
                variant: str, seed: int, max_steps: int = 300, patience: int = 40,
                lr: float = 1e-3, weight_decay: float = 1e-2,
                d_proj: int = 64, d_att: int = 32, dropout: float = 0.3) -> np.ndarray:
    """Train one MIL on a training fold, early-stopping on an inner split."""
    y_train = target[train_idx]
    if y_train.sum() < 2 or y_train.sum() == len(y_train):
        return np.full(len(test_idx), float(y_train.mean()))

    inner_train, inner_val = train_test_split(
        np.arange(len(train_idx)), test_size=0.25, stratify=y_train, random_state=seed)
    fit_idx = train_idx[inner_train]
    val_idx = train_idx[inner_val]

    torch.manual_seed(seed)
    model = AttentionMIL(x.shape[-1], variant, d_proj, d_att, dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_fit = torch.tensor(target[fit_idx], dtype=torch.float32, device=DEVICE)
    n_pos = max(1.0, float(y_fit.sum()))
    pos_weight = torch.tensor((len(y_fit) - n_pos) / n_pos, device=DEVICE)

    best_state, best_auc, waited = None, -np.inf, 0
    for step in range(max_steps):
        model.train()
        optimizer.zero_grad()
        logits = model(x[fit_idx], bags.mask[fit_idx], bags.plane[fit_idx],
                       bags.normal[fit_idx])
        loss = F.binary_cross_entropy_with_logits(logits, y_fit, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            model.eval()
            with torch.no_grad():
                scores = model(x[val_idx], bags.mask[val_idx], bags.plane[val_idx],
                               bags.normal[val_idx]).cpu().numpy()
            y_val = target[val_idx]
            auc = roc_auc_score(y_val, scores) if len(np.unique(y_val)) > 1 else 0.5
            if auc > best_auc:
                best_auc, waited = auc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                waited += 5
                if waited >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x[test_idx], bags.mask[test_idx], bags.plane[test_idx],
                     bags.normal[test_idx]).cpu().numpy()


def evaluate(bags: Bags, variant: str, seed: int = 0, n_splits: int = 5,
             pca_dim: int = 0, **fit_kwargs) -> dict:
    per_label, oof_all = {}, {}
    for label in data.LABELS:
        target = bags.labels[label].to_numpy()
        oof = np.zeros(len(target))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(np.zeros(len(target)), target):
            x = bags.standardized(train_idx, pca_dim)
            oof[test_idx] = fit_predict(bags, x, train_idx, test_idx, target, variant,
                                        seed, **fit_kwargs)
        per_label[label] = roc_auc_score(target, oof)
        oof_all[label] = oof
    return {"macro_auc": float(np.mean(list(per_label.values()))),
            "per_label": per_label, "oof": oof_all}


SOURCES = [
    ("orthofoundation", 224, "cls"),
    ("orthofoundation", 224, "patch_std"),
    ("mri_core", 224, "cls"),
]
VARIANTS = ["plain", "plane_embed", "plane_attn", "orient"]


# Chosen by a 32-point grid over level/pca/d_proj/weight_decay/dropout on seeds
# 0-2 (see README). Series-level bags win: 46 training bags cannot supervise an
# attention over ~180 slices, but they can over ~6 series descriptors.
TUNED = dict(level="series", pca_dim=64, d_proj=32, d_att=16,
             weight_decay=1e-1, dropout=0.5)


def main(seeds: tuple[int, ...], out: str) -> None:
    rows = []
    fit_kwargs = {k: TUNED[k] for k in ("d_proj", "d_att", "weight_decay", "dropout")}
    for backbone, size, head in SOURCES:
        bags = Bags(backbone, size, head, level=TUNED["level"])
        for variant in VARIANTS:
            runs = [evaluate(bags, variant, seed=s, pca_dim=TUNED["pca_dim"], **fit_kwargs)
                    for s in seeds]
            macros = [r["macro_auc"] for r in runs]
            per_label = {l: float(np.mean([r["per_label"][l] for r in runs]))
                         for l in data.LABELS}
            rows.append({"backbone": backbone, "head": head, "variant": variant,
                         "macro_auc": float(np.mean(macros)), "std": float(np.std(macros)),
                         **{f"auc/{k}": v for k, v in per_label.items()}})
            print(f"{backbone:16s} {head:10s} {variant:12s} "
                  f"macro {np.mean(macros):.4f} +/- {np.std(macros):.4f}", flush=True)
        del bags
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    frame.to_csv(RESULTS / out, index=False)
    print("\nRanked:")
    print(frame[["backbone", "head", "variant", "macro_auc", "std"]]
          .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out", default="mil.csv")
    args = parser.parse_args()
    main(tuple(range(args.first_seed, args.first_seed + args.n_seeds)), args.out)
