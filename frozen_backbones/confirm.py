#!/usr/bin/env python3
"""Re-score the leading configurations on CV seeds never used for selection.

The sweep ranked 480 configurations on seed 0 and stage2 re-ranked the leaders
on seeds 0-4, so both of those numbers have had the same data used to choose
what to report. This script fixes the shortlist and evaluates it on a disjoint
block of seeds, which is the number worth quoting.

It also reports a permutation baseline: the same pipeline with the labels
shuffled, which is what "chance" actually looks like once fold-wise model
selection on 58 studies is in the loop.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import data
import probe
from stage2 import Pool, parse_pooling

RESULTS = Pool  # re-exported for symmetry; the path lives in stage2/probe

ORTHO_CLS = [("orthofoundation", 224, "cls")]
ORTHO_STD = [("orthofoundation", 224, "patch_std")]
MRI_CLS = [("mri_core", 224, "cls")]

SHORTLIST = [
    # Hierarchical leaders from hier.py.
    ("orthofoundation/224/cls", ORTHO_CLS, "meanmax-plane-inp90-axmax-ctr"),
    ("orthofoundation/224/cls", ORTHO_CLS, "max-all-inp90-ctr"),
    ("orthofoundation/224/cls", ORTHO_CLS, "meanmax-all-inp90-ctr"),
    ("orthofoundation/224/patch_std", ORTHO_STD, "meanmax-plane-axmean-ctr"),
    # Matched controls: identical in every respect except the series level, so
    # the difference is attributable to hierarchy alone rather than to the
    # central trim or the plane grouping that the leaders also use.
    ("orthofoundation/224/cls", ORTHO_CLS, "meanmax-plane-axmax-ctr"),
    ("orthofoundation/224/cls", ORTHO_CLS, "meanmax-all-ctr"),
    # Flat-bag leaders from the first sweep.
    ("orthofoundation/224/patch_std", ORTHO_STD, "mean-plane-ctr-l2"),
    ("mri_core/224/cls", MRI_CLS, "meanmax-all-bal"),
    # A deliberately plain reference point: the simplest thing anyone would try.
    ("mri_core/224/cls", MRI_CLS, "mean-all"),
    ("orthofoundation/224/cls", ORTHO_CLS, "mean-all"),
]


def permutation_baseline(pool: Pool, sources, pooling, seeds) -> float:
    """Macro AUC when the labels are shuffled, holding everything else fixed."""
    x = pool.matrix(sources, pooling)
    scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = pool.labels.apply(lambda col: rng.permutation(col.to_numpy()))
        scores.append(probe.evaluate(x, shuffled, seed=seed)["macro_auc"])
    return float(np.mean(scores)), float(np.std(scores))


def main(first_seed: int, n_seeds: int) -> None:
    seeds = tuple(range(first_seed, first_seed + n_seeds))
    pool = Pool()
    print(f"seeds {seeds[0]}-{seeds[-1]} ({n_seeds} repeats), 58 studies\n")

    rows = []
    for name, sources, pooling_name in SHORTLIST:
        pooling = parse_pooling(pooling_name)
        mean, std, per_label = pool.score(sources, pooling, seeds)
        rows.append({"config": name, "pooling": pooling_name,
                     "macro_auc": mean, "std": std, **per_label})
        print(f"{name:30s} {pooling_name:20s} macro {mean:.4f} +/- {std:.4f}", flush=True)

    frame = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    frame.to_csv(probe.FEATURES.parent / "results" / "confirm.csv", index=False)

    best = frame.iloc[0]
    print(f"\nleading: {best['config']} / {best['pooling']}")
    per_label = pd.DataFrame({
        "positives": [int(pool.studies[l].sum()) for l in data.LABELS],
        "auc": [best[l] for l in data.LABELS],
    }, index=data.LABELS).sort_values("auc", ascending=False)
    print(per_label.to_string(float_format=lambda v: f"{v:.4f}"))

    name, sources, pooling_name = SHORTLIST[0]
    chance, chance_std = permutation_baseline(
        pool, sources, parse_pooling(pooling_name), seeds[:5])
    print(f"\npermutation baseline (labels shuffled): {chance:.4f} +/- {chance_std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    main(args.first_seed, args.n_seeds)
