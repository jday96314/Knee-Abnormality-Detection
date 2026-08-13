#!/usr/bin/env python3
"""The evaluation protocol every architecture in this sweep must follow.

Fixed here rather than per-architecture so that numbers from different subdirectories are
comparable, and so the protocol cannot quietly drift as the sweep progresses.

    train      80% of the pseudo-labelled studies
    validate   the remaining 20%, scored by soft binary cross-entropy
    report     ROC AUC over the 58 gold studies, as an auxiliary number only

The split is deterministic given the seed, so every architecture sees exactly the same
studies. The gold studies are *excluded from both* train and validation: they are a
locked measurement, never a tuning signal. Selecting on them would both overfit 58 cases
and defeat the point of having an independent check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dataset import LABELS

SPLIT_SEED = 20260813
VAL_FRACTION = 0.2


def make_split(studies: pd.DataFrame, seed: int = SPLIT_SEED) -> dict[str, np.ndarray]:
    """Deterministic 80/20 split over the pseudo-labelled studies.

    Gold studies sit outside both halves. There are only 58 of them against 4,349
    pseudo-labelled ones, so excluding them costs the training set ~1.3% and buys an
    evaluation set that no architecture choice has ever touched.
    """
    rng = np.random.default_rng(seed)
    pseudo = np.flatnonzero(~studies["is_gold"].to_numpy())
    order = rng.permutation(len(pseudo))
    cut = int(round(len(pseudo) * (1 - VAL_FRACTION)))
    return {
        "train": np.sort(pseudo[order[:cut]]),
        "val": np.sort(pseudo[order[cut:]]),
        "gold": np.flatnonzero(studies["is_gold"].to_numpy()),
    }


def soft_targets(studies: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    return studies.iloc[rows][LABELS].to_numpy(np.float32)


def gold_targets(studies: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    return studies.iloc[rows][[f"{c}__gold" for c in LABELS]].to_numpy(np.float32)


def soft_bce(target: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-finding soft BCE. The optimisation objective and the selection metric.

    Soft targets are kept as probabilities rather than thresholded: a teacher value of
    0.84 carries information that a hard 1 discards, which is the entire reason for
    training on 4,349 studies instead of 58.
    """
    p = np.clip(pred, eps, 1 - eps)
    return -(target * np.log(p) + (1 - target) * np.log(1 - p)).mean(axis=0)


def gold_auc(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-finding ROC AUC on the gold studies. Reported, never optimised against."""
    out = np.full(target.shape[1], np.nan)
    for j in range(target.shape[1]):
        y = target[:, j]
        if len(np.unique(y)) > 1 and len(np.unique(pred[:, j])) > 1:
            out[j] = roc_auc_score(y, pred[:, j])
    return out


def report(name: str, val_target, val_pred, gold_target, gold_pred, extra: dict | None = None) -> dict:
    """One result row, in the shape every architecture in the sweep reports."""
    bce = soft_bce(val_target, val_pred)
    auc = gold_auc(gold_target, gold_pred)
    row = {
        "model": name,
        "val_soft_bce": float(bce.mean()),
        "gold_auc": float(np.nanmean(auc)),
        **{f"bce__{lab}": float(b) for lab, b in zip(LABELS, bce)},
        **{f"auc__{lab}": float(a) for lab, a in zip(LABELS, auc)},
    }
    row.update(extra or {})
    return row


def baseline_bce(studies: pd.DataFrame, split: dict[str, np.ndarray]) -> float:
    """Predict the training-set mean for every study. The bar a model must clear."""
    train_mean = soft_targets(studies, split["train"]).mean(axis=0, keepdims=True)
    val = soft_targets(studies, split["val"])
    return float(soft_bce(val, np.repeat(train_mean, len(val), axis=0)).mean())
