#!/usr/bin/env python3
"""Pool per-slice frozen features to study level and linear-probe them.

A study is a bag of slices drawn from several series in several planes, and the
twelve findings live at different places in that bag: an effusion is visible on
most fluid-sensitive slices, a fracture on one or two. Pooling is therefore the
part of this pipeline that decides what the classifier can possibly see, which
is why it is the swept axis and the backbone read-out is swept alongside it.

Scoring is macro-averaged ROC AUC over pooled out-of-fold predictions from
5-fold stratified CV. The regularization strength is chosen inside each training
fold, never on the out-of-fold predictions being scored.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data

FEATURES = Path(__file__).resolve().parent / "features"
PLANES = ("Sagittal", "Coronal", "Axial")

# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _reduce(block: np.ndarray, how: str) -> np.ndarray:
    """Collapse a [n_slices, D] block to a fixed-length descriptor."""
    if how == "mean":
        return block.mean(0)
    if how == "max":
        return block.max(0)
    if how == "std":
        return block.std(0)
    if how == "p90":
        return np.percentile(block, 90, axis=0)
    if how == "meanmax":
        return np.concatenate([block.mean(0), block.max(0)])
    if how == "meanstd":
        return np.concatenate([block.mean(0), block.std(0)])
    if how == "meanmaxstd":
        return np.concatenate([block.mean(0), block.max(0), block.std(0)])
    if how == "gem3":
        # Generalized mean: between mean and max, so a finding on a minority of
        # slices survives averaging without the noise sensitivity of a hard max.
        shifted = block - block.min(0, keepdims=True) + 1e-6
        return (shifted ** 3).mean(0) ** (1 / 3)
    raise ValueError(how)


@dataclass(frozen=True)
class Pooling:
    """How one study's slices become one vector.

    `group` splits the bag before reducing and concatenates the results, so the
    classifier sees which plane a response came from instead of averaging a
    sagittal and an axial slice together. `central` drops the ends of each stack,
    which on a knee series are off-joint.

    `inner` makes the pooling hierarchical: each series is first collapsed on its
    own with `inner`, and `reduce` then runs across the resulting one-vector-per-
    series bag. This is what lets a focal finding survive. A fracture visible on
    two slices of a 40-slice series is a 5% perturbation of a flat mean over the
    study, but `inner="max"` promotes it to that series' descriptor, where the
    outer reduction sees it at full strength. It also decouples the two levels:
    a max within series says "the strongest evidence in this acquisition", while
    a mean across series says "how much of the protocol agrees".

    `inner="mean"` is the special case that just re-weights series equally, so a
    320-slice acquisition cannot outvote a 20-slice one; it is named `bal` for
    continuity with the first sweep, which only had that one variant.
    """

    reduce: str = "mean"
    group: str = "all"          # all | plane | fluid | plane_fluid
    inner: str = ""             # collapse each series first, then `reduce` across series
    across: str = ""            # reduce across groups instead of concatenating them
    central: bool = False       # keep only the central half of each stack
    l2: bool = False            # unit-normalize each slice vector before pooling

    def name(self) -> str:
        bits = [self.reduce, self.group]
        if self.inner:
            bits.append("bal" if self.inner == "mean" else f"in{self.inner}")
        if self.across:
            bits.append(f"ax{self.across}")
        if self.central:
            bits.append("ctr")
        if self.l2:
            bits.append("l2")
        return "-".join(bits)

    # Group keys are strings, not tuples: comparing an object array of tuples
    # against a tuple makes numpy broadcast the tuple's elements, which silently
    # yields an all-False mask for 1-tuples and raises for longer ones.
    def groups(self) -> list[str]:
        if self.group == "all":
            return [""]
        if self.group in ("plane", "orient"):
            return list(PLANES)
        if self.group == "fluid":
            return ["0", "1"]
        if self.group == "plane_fluid":
            return [f"{p}|{f}" for p in PLANES for f in (0, 1)]
        raise ValueError(self.group)

    def key_of(self, meta: pd.DataFrame) -> np.ndarray | None:
        if self.group in ("all", "orient"):
            return None
        if self.group == "plane":
            return meta["plane"].to_numpy().astype(str)
        if self.group == "fluid":
            return meta["fluid"].astype(str).to_numpy()
        return (meta["plane"].astype(str) + "|" + meta["fluid"].astype(str)).to_numpy()


def orientation_weights(meta: pd.DataFrame) -> np.ndarray:
    """Soft membership of each slice in the three cardinal orientations.

    `Anatomical_Plane` hard-assigns every series to one of three bins, but this
    cohort's series run up to 38 degrees off-axis, and a coronal-oblique
    acquisition genuinely carries some of what an axial would show. The squared
    direction cosines of the slice normal give exactly that: a partition of unity
    over the three axes (they sum to 1 for a unit vector), so a true coronal
    contributes 1.0 to the coronal block and an oblique one splits itself.

    Returns [n_slices, 3] in PLANES order (Sagittal, Coronal, Axial), matching
    the L-R / A-P / S-I patient axes that define those planes.
    """
    normals = data.series_orientation().set_index("SeriesInstanceUID")
    columns = normals[["normal0", "normal1", "normal2"]]
    per_slice = columns.reindex(meta["series"].to_numpy()).to_numpy(dtype=np.float64)
    weights = per_slice ** 2
    total = weights.sum(axis=1, keepdims=True)
    # A series whose orientation is missing falls back to uniform membership
    # rather than dropping out of every block.
    return np.where(total > 1e-6, weights / np.maximum(total, 1e-12), 1 / 3)


def _soft_block(features: np.ndarray, meta: pd.DataFrame, mask: np.ndarray,
                weight: np.ndarray, pooling: Pooling) -> np.ndarray:
    """One orientation block: a weighted mean, optionally over series descriptors.

    Only mean-like reductions are weighted here, because a weighted maximum is
    not well defined -- scaling a feature by its membership would confound "this
    slice is oblique" with "this slice responded weakly". The focal-preserving
    work is done by `inner` instead: each series is collapsed by, say, p90 first,
    and the orientation weights then combine those series descriptors. That
    composes the hierarchical result with soft orientation rather than forcing a
    choice between them.
    """
    if pooling.reduce not in ("mean", "meanstd"):
        raise ValueError(f"group='orient' supports mean/meanstd, not {pooling.reduce!r}")

    width = features.shape[1]
    size = width * (2 if pooling.reduce == "meanstd" else 1)
    if not mask.any():
        return np.zeros(size)

    block, w = features[mask], weight[mask]
    if pooling.inner:
        series = meta.loc[mask, "series"].to_numpy()
        unique = pd.unique(series)
        block = np.stack([_reduce(block[series == s], pooling.inner) for s in unique])
        # Orientation is a property of the series, so one weight per descriptor.
        w = np.array([w[series == s][0] for s in unique])

    total = w.sum()
    if total < 1e-8:
        return np.zeros(size)
    mean = (w[:, None] * block).sum(0) / total
    if pooling.reduce == "mean":
        return mean
    variance = (w[:, None] * (block - mean) ** 2).sum(0) / total
    return np.concatenate([mean, np.sqrt(np.maximum(variance, 0.0))])


def pool_studies(features: np.ndarray, meta: pd.DataFrame, studies: list[str],
                 pooling: Pooling) -> np.ndarray:
    """Build the [n_studies, D'] design matrix for one pooling strategy.

    Groups absent from a study (an axial-free protocol, say) contribute zeros.
    That is a deliberate choice over dropping the study: with 58 studies the
    cohort cannot afford listwise deletion, and a zero block is a constant the
    scaler and the L2 penalty can absorb.
    """
    if pooling.l2:
        # Removes per-slice magnitude before averaging, so a bright or
        # high-contrast slice cannot dominate the study descriptor purely by
        # having a larger activation norm than its neighbours.
        features = features / np.maximum(
            np.linalg.norm(features, axis=1, keepdims=True), 1e-8)

    keys = pooling.key_of(meta)
    study_of = meta["study"].to_numpy()
    position = meta["pos"].to_numpy()
    soft = orientation_weights(meta) if pooling.group == "orient" else None
    rows = []

    width = features.shape[1]
    for study in studies:
        in_study = study_of == study
        blocks = []
        for axis, group in enumerate(pooling.groups()):
            mask = in_study if keys is None else (in_study & (keys == group))
            if pooling.central:
                mask = mask & (position > 0.25) & (position < 0.75)

            if soft is not None:
                blocks.append(_soft_block(features, meta, mask, soft[:, axis], pooling))
                continue

            if not mask.any():
                # When groups are reduced across rather than concatenated, a
                # missing plane is skipped entirely: zero-filling would make an
                # absent acquisition compete in the max.
                if pooling.across:
                    continue
                probe = _reduce(np.zeros((1, width), np.float32), pooling.reduce)
                blocks.append(np.zeros_like(probe))
                continue

            block = features[mask]
            if pooling.inner:
                # One descriptor per series, then the outer reduction across them.
                # `inner` must be a single-statistic reducer so every series
                # contributes the same D columns.
                series = meta.loc[mask, "series"].to_numpy()
                block = np.stack([_reduce(block[series == s], pooling.inner)
                                  for s in pd.unique(series)])
            blocks.append(_reduce(block, pooling.reduce))

        if not pooling.across:
            rows.append(np.concatenate(blocks))
        elif blocks:
            rows.append(_reduce(np.stack(blocks), pooling.across))
        else:
            rows.append(np.zeros(len(_reduce(np.zeros((1, width), np.float32),
                                             pooling.reduce))))

    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class RowSpaceSVD(BaseEstimator, TransformerMixin):
    """Rotate onto the training row space. Exact, not approximate.

    For an L2-penalized linear model the solution lies in the span of the
    training rows, so any component of a test point orthogonal to that span is
    multiplied by zero. Rotating both onto the training right-singular vectors
    therefore leaves every prediction unchanged while shrinking the design from
    d columns (up to 12288 here) to at most n_train (46). The penalty is
    preserved too, because the rotation is orthonormal.

    This is purely a speedup -- it makes the sweep roughly two orders of
    magnitude cheaper -- and `tests/test_probe.py` asserts the equivalence
    against an unreduced fit.
    """

    def fit(self, x, y=None):
        centered = x - x.mean(axis=0, keepdims=True)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        keep = singular > singular[0] * 1e-10 if singular[0] > 0 else singular > 0
        self.components_ = vt[keep]
        return self

    def transform(self, x):
        return x @ self.components_.T


def _probe_pipeline(c: float):
    """The probe as a single estimator. Used by the equivalence test."""
    return make_pipeline(
        StandardScaler(),
        RowSpaceSVD(),
        LogisticRegression(C=c, max_iter=2000, class_weight="balanced"),
    )


def _logit(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=2000, class_weight="balanced")


def _reduce_design(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize on the training rows, then rotate both onto their row space."""
    scaler = StandardScaler().fit(x_train)
    rotation = RowSpaceSVD().fit(scaler.transform(x_train))
    return (rotation.transform(scaler.transform(x_train)),
            rotation.transform(scaler.transform(x_test)))

# Spans under- to over-fitting for these designs. The upper end matters: with
# d >> n the penalty has to be weak before the decision function stops being a
# rescaled class-mean difference, and a grid topping out at C=1 leaves real AUC
# on the table. The grid is threaded through every call rather than read from
# this global, which would not reach joblib's worker processes.
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)


def _fit_predict(x_train, y_train, x_test, c_grid=C_GRID, inner_splits=4,
                 inner_repeats=3, seed=0):
    """Select C inside the training fold, then predict the held-out fold.

    The inner selection is what keeps the reported AUC honest: choosing C
    against the out-of-fold predictions would tune on the same numbers the
    score is computed from.
    """
    positives = int(y_train.sum())
    if positives < 2 or positives == len(y_train):
        return np.full(len(x_test), float(y_train.mean()))

    n_splits = max(2, min(inner_splits, positives, len(y_train) - positives))
    # Repeated rather than single inner CV. With ~46 training studies and as few
    # as 7 positives, one 4-fold estimate of a candidate C is noisy enough that
    # a wider grid scores *worse* -- the extra candidates just give noise more
    # chances to win. Averaging over repeats is what makes the grid safe to widen.
    inner = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=inner_repeats,
                                    random_state=seed)
    splits = [(tr, te) for tr, te in inner.split(x_train, y_train)
              if len(np.unique(y_train[tr])) > 1 and len(np.unique(y_train[te])) > 1]

    # Scale and rotate once per split, then sweep C on the reduced design. The
    # scaler and the rotation do not depend on C, and refitting them inside the
    # C loop is what made the row-space reduction cost as much as it saved.
    reduced = []
    for tr, te in splits:
        z_tr, z_te = _reduce_design(x_train[tr], x_train[te])
        reduced.append((z_tr, y_train[tr], z_te, y_train[te]))

    best_c, best_score = c_grid[0], -np.inf
    for c in c_grid:
        scores = [
            roc_auc_score(y_te, _logit(c).fit(z_tr, y_tr).predict_proba(z_te)[:, 1])
            for z_tr, y_tr, z_te, y_te in reduced
        ]
        if scores and np.mean(scores) > best_score:
            best_c, best_score = c, float(np.mean(scores))

    z_train, z_test = _reduce_design(x_train, x_test)
    return _logit(best_c).fit(z_train, y_train).predict_proba(z_test)[:, 1]


def _oof_for_label(x: np.ndarray, target: np.ndarray, seed: int, n_splits: int,
                   c_grid) -> np.ndarray:
    oof = np.zeros(len(target))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(x, target):
        oof[test_idx] = _fit_predict(x[train_idx], target[train_idx], x[test_idx],
                                     c_grid=c_grid, seed=seed)
    return oof


def evaluate(x: np.ndarray, y: pd.DataFrame, seed: int = 0, n_splits: int = 5,
             n_jobs: int = 12, c_grid=C_GRID) -> dict:
    """Pooled out-of-fold ROC AUC per label, plus the macro average.

    Each label gets its own stratified split. With MCL at 9 positives in 58
    studies, a split shared across all twelve labels cannot keep every label
    balanced, and an unstratified fold would sometimes hold no positives at all.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        oofs = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_oof_for_label)(x, y[label].to_numpy(), seed, n_splits, c_grid)
            for label in data.LABELS
        )

    per_label = {label: roc_auc_score(y[label].to_numpy(), oof)
                 for label, oof in zip(data.LABELS, oofs)}
    return {
        "macro_auc": float(np.mean(list(per_label.values()))),
        "per_label": per_label,
        "oof": dict(zip(data.LABELS, oofs)),
    }


def load_features(backbone: str, size: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    payload = np.load(FEATURES / f"{backbone}_{size}.npz")
    heads = {k: payload[k].astype(np.float32) for k in payload.files}
    meta = pd.DataFrame(json.loads((FEATURES / f"{backbone}_{size}.index.json").read_text()))
    return heads, meta
