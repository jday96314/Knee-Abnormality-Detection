#!/usr/bin/env python3
"""Sweep backbone x resolution x read-out head x pooling strategy.

Writes every configuration's macro AUC to results/sweep.csv so the comparison
can be re-read without recomputing. The sweep is deliberately wide on pooling
and narrow on everything else: the pooling axis is the question, the rest is
context for it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import data
import probe

RESULTS = Path(__file__).resolve().parent / "results"

POOLINGS = [
    # Reducer sweep on the whole bag of slices.
    probe.Pooling("mean"), probe.Pooling("max"), probe.Pooling("std"),
    probe.Pooling("p90"), probe.Pooling("gem3"),
    probe.Pooling("meanmax"), probe.Pooling("meanstd"), probe.Pooling("meanmaxstd"),
    # Does telling the classifier which plane a response came from help?
    probe.Pooling("mean", "plane"), probe.Pooling("max", "plane"),
    probe.Pooling("meanmax", "plane"), probe.Pooling("meanmaxstd", "plane"),
    # Fluid-sensitive vs T1-like carry different findings.
    probe.Pooling("mean", "fluid"), probe.Pooling("meanmax", "fluid"),
    probe.Pooling("mean", "plane_fluid"), probe.Pooling("meanmax", "plane_fluid"),
    # Weighting and slice-range variants of the two strongest plain reducers.
    probe.Pooling("mean", balance=True), probe.Pooling("meanmax", balance=True),
    probe.Pooling("mean", "plane", balance=True),
    probe.Pooling("mean", central=True), probe.Pooling("meanmax", central=True),
    probe.Pooling("mean", "plane", central=True),
    # Unit-normalizing each slice before pooling, across the same shapes.
    probe.Pooling("mean", l2=True), probe.Pooling("max", l2=True),
    probe.Pooling("meanmax", l2=True), probe.Pooling("meanmaxstd", l2=True),
    probe.Pooling("mean", "plane", l2=True), probe.Pooling("meanmax", "plane", l2=True),
    probe.Pooling("mean", balance=True, l2=True),
    probe.Pooling("meanmax", central=True, l2=True),
]


def run(configs, out_name: str, seed: int = 0) -> pd.DataFrame:
    studies, _ = data.cohort()
    study_ids = studies.StudyInstanceUID.tolist()
    labels = studies[data.LABELS]

    rows = []
    cache: dict[tuple, tuple] = {}

    for backbone, size, head, pooling in configs:
        if (backbone, size) not in cache:
            cache.clear()
            cache[(backbone, size)] = probe.load_features(backbone, size)
        heads, meta = cache[(backbone, size)]

        x = probe.pool_studies(heads[head], meta, study_ids, pooling)
        if not np.isfinite(x).all() or x.std(axis=0).max() == 0:
            raise ValueError(f"degenerate design matrix for {backbone}/{head}/{pooling.name()}")
        result = probe.evaluate(x, labels, seed=seed)

        rows.append({
            "backbone": backbone, "size": size, "head": head,
            "pooling": pooling.name(), "dim": x.shape[1],
            "macro_auc": result["macro_auc"],
            **{f"auc/{k}": v for k, v in result["per_label"].items()},
        })
        print(f"{backbone:16s} {size} {head:11s} {pooling.name():22s} "
              f"d={x.shape[1]:5d}  macro AUC {result['macro_auc']:.4f}", flush=True)

    frame = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    RESULTS.mkdir(exist_ok=True)
    frame.to_csv(RESULTS / out_name, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=["mri_core", "orthofoundation"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[224, 448])
    parser.add_argument("--heads", nargs="+",
                        default=["cls", "patch_mean", "patch_max", "patch_std"])
    parser.add_argument("--out", default="sweep.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = [(b, s, h, p) for b, s in itertools.product(args.backbones, args.sizes)
               for h in args.heads for p in POOLINGS]
    print(f"{len(configs)} configurations\n")
    frame = run(configs, args.out, seed=args.seed)
    print("\nTop 15:")
    print(frame.head(15)[["backbone", "size", "head", "pooling", "dim", "macro_auc"]]
          .to_string(index=False))
