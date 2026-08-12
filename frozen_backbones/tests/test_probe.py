#!/usr/bin/env python3
"""Correctness checks for the probe's pooling and its row-space speedup."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe  # noqa: E402


def synthetic(n_studies=20, n_slices=7, dim=300, seed=0):
    rng = np.random.default_rng(seed)
    rows, features = [], []
    planes = ["Sagittal", "Coronal", "Axial"]
    for s in range(n_studies):
        for k in range(n_slices):
            rows.append({
                "study": f"s{s}",
                "series": f"s{s}_ser{k % 2}",
                "plane": planes[k % 3],
                "fluid": k % 2,
                "fatsat": k % 2,
                "pos": (k + 0.5) / n_slices,
            })
            features.append(rng.normal(size=dim))
    return np.asarray(features), pd.DataFrame(rows), [f"s{s}" for s in range(n_studies)]


@pytest.mark.parametrize("c", [1e-2, 1.0, 1e2])
@pytest.mark.parametrize("dim", [80, 300])
def test_rowspace_svd_matches_unreduced_fit(c, dim):
    """The speedup must not change a single prediction.

    Both fits are run to a much tighter tolerance than the probe's default. The
    equivalence is exact in the optimum, so the only thing that can separate the
    two parameterizations is how far lbfgs stopped short of it -- at C=1e2 the
    default tol=1e-4 leaves them ~1e-2 apart in probability, which is a
    convergence artefact rather than a difference in the solution.
    """
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(40, dim))
    y_train = (rng.random(40) < 0.4).astype(int)
    x_test = rng.normal(size=(18, dim))

    settings = dict(C=c, max_iter=200000, tol=1e-12, class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reference = make_pipeline(
            StandardScaler(), LogisticRegression(**settings),
        ).fit(x_train, y_train).predict_proba(x_test)[:, 1]
        reduced = probe._probe_pipeline(c)
        reduced.set_params(**{f"logisticregression__{k}": v for k, v in settings.items()})
        reduced = reduced.fit(x_train, y_train).predict_proba(x_test)[:, 1]

    assert np.allclose(reference, reduced, atol=1e-4), np.abs(reference - reduced).max()


def test_pooling_group_masks_are_not_empty():
    """Grouped pooling must actually select slices, not silently produce zeros."""
    features, meta, studies = synthetic()
    for group in ("all", "plane", "fluid", "plane_fluid"):
        pooling = probe.Pooling("mean", group)
        x = probe.pool_studies(features, meta, studies, pooling)
        n_groups = len(pooling.groups())
        assert x.shape == (len(studies), features.shape[1] * n_groups)
        # Every block must vary across studies; an all-False mask would leave a
        # constant zero block, which is the bug this guards.
        for block in range(n_groups):
            columns = x[:, block * features.shape[1]:(block + 1) * features.shape[1]]
            assert columns.std(axis=0).max() > 0, f"{group} block {block} is constant"


def test_reduce_dimensions():
    block = np.random.default_rng(0).normal(size=(9, 25))
    assert probe._reduce(block, "mean").shape == (25,)
    assert probe._reduce(block, "meanmax").shape == (50,)
    assert probe._reduce(block, "meanmaxstd").shape == (75,)
    assert np.allclose(probe._reduce(block, "max"), block.max(0))


def test_l2_pooling_normalizes_slices():
    features, meta, studies = synthetic(dim=40)
    scaled = features * np.linspace(0.1, 10, len(features))[:, None]
    plain = probe.Pooling("mean", l2=False)
    unit = probe.Pooling("mean", l2=True)
    # Rescaling individual slices changes an unnormalized mean but not a
    # normalized one.
    assert not np.allclose(probe.pool_studies(features, meta, studies, plain),
                           probe.pool_studies(scaled, meta, studies, plain))
    assert np.allclose(probe.pool_studies(features, meta, studies, unit),
                       probe.pool_studies(scaled, meta, studies, unit))


def test_balance_weights_series_equally():
    """A long series must not outvote a short one when balance=True."""
    rng = np.random.default_rng(3)
    rows, features = [], []
    for k in range(30):  # 30 slices in series A, 2 in series B
        rows.append({"study": "s0", "series": "A", "plane": "Sagittal",
                     "fluid": 1, "fatsat": 1, "pos": 0.5})
        features.append(np.zeros(4))
    for k in range(2):
        rows.append({"study": "s0", "series": "B", "plane": "Sagittal",
                     "fluid": 1, "fatsat": 1, "pos": 0.5})
        features.append(np.ones(4))
    features, meta = np.asarray(features), pd.DataFrame(rows)

    plain = probe.pool_studies(features, meta, ["s0"], probe.Pooling("mean"))
    balanced = probe.pool_studies(features, meta, ["s0"], probe.Pooling("mean", inner="mean"))
    assert np.allclose(plain, 2 / 32)
    assert np.allclose(balanced, 0.5)


def test_inner_max_promotes_a_focal_finding():
    """A finding on 2 of 40 slices survives inner=max and is diluted by a mean.

    This is the whole point of hierarchical pooling: series A is 40 slices of
    which 2 carry the finding, series B is 20 clean slices. A flat mean over the
    study sees the finding at 2/60; collapsing each series by max first makes it
    a full-strength component of one of the two series descriptors.
    """
    rows, features = [], []
    for k in range(40):
        rows.append({"study": "s0", "series": "A", "plane": "Sagittal",
                     "fluid": 1, "fatsat": 1, "pos": (k + 0.5) / 40})
        features.append(np.ones(4) if k in (18, 19) else np.zeros(4))
    for k in range(20):
        rows.append({"study": "s0", "series": "B", "plane": "Sagittal",
                     "fluid": 1, "fatsat": 1, "pos": (k + 0.5) / 20})
        features.append(np.zeros(4))
    features, meta = np.asarray(features, float), pd.DataFrame(rows)

    flat = probe.pool_studies(features, meta, ["s0"], probe.Pooling("mean"))
    hier = probe.pool_studies(features, meta, ["s0"], probe.Pooling("mean", inner="max"))
    assert np.allclose(flat, 2 / 60)
    assert np.allclose(hier, 0.5)  # (max over A = 1, max over B = 0) -> mean


def test_pooling_name_roundtrips():
    from stage2 import parse_pooling
    for pooling in [
        probe.Pooling("mean"),
        probe.Pooling("meanmax", "plane"),
        probe.Pooling("mean", "all", "mean"),          # legacy "bal" spelling
        probe.Pooling("meanmax", "plane", "max", central=True, l2=True),
        probe.Pooling("mean", "fluid", "p90", central=True),
        probe.Pooling("max", "plane", "max", "mean"),
        probe.Pooling("meanmax", "plane", across="max"),
    ]:
        assert parse_pooling(pooling.name()) == pooling, pooling.name()


def test_across_reduces_groups_instead_of_concatenating():
    """`across` keeps the descriptor at D columns and skips absent planes."""
    rows, features = [], []
    for plane, value in [("Sagittal", 1.0), ("Coronal", 3.0)]:  # no axial series
        for k in range(4):
            rows.append({"study": "s0", "series": f"{plane}_a", "plane": plane,
                         "fluid": 1, "fatsat": 1, "pos": (k + 0.5) / 4})
            features.append(np.full(5, value))
    features, meta = np.asarray(features), pd.DataFrame(rows)

    concat = probe.pool_studies(features, meta, ["s0"], probe.Pooling("mean", "plane"))
    assert concat.shape == (1, 15)
    assert np.allclose(concat[0, 10:], 0.0)  # missing axial block is zero-filled

    # Reducing across planes must ignore the absent one, not average a zero in.
    assert np.allclose(
        probe.pool_studies(features, meta, ["s0"], probe.Pooling("mean", "plane", across="mean")),
        np.full((1, 5), 2.0))
    assert np.allclose(
        probe.pool_studies(features, meta, ["s0"], probe.Pooling("mean", "plane", across="max")),
        np.full((1, 5), 3.0))
