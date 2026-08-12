#!/usr/bin/env python3
"""Ensemble backbones at the prediction level rather than the feature level.

`stage2.py` already showed that *concatenating* two backbones' features scores
below the better one alone: at n=58 the extra columns cost more than the extra
signal pays. Averaging predictions is the other way to combine them, and it has
the opposite scaling -- each member is fitted in its own, smaller feature space,
so nothing is paid for the union's dimensionality.

The members' folds are identical by construction: `StratifiedKFold.split` uses
only the length of X and the label vector, so for a fixed seed and label every
member produces out-of-fold predictions for the same study in the same fold.
That is what makes averaging them legitimate rather than a leak.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import data
import probe
from stage2 import Pool, parse_pooling

RESULTS = probe.FEATURES.parent / "results"

# One member per distinct backbone/read-out combination, each with a pooling
# that performed well for it. Deliberately diverse: two backbones, three heads.
MEMBERS = {
    "ortho/cls": ([("orthofoundation", 224, "cls")], "max-all-inp90-ctr"),
    "ortho/patch_std": ([("orthofoundation", 224, "patch_std")], "meanmax-plane-axmean-ctr"),
    "mri/cls": ([("mri_core", 224, "cls")], "meanmax-plane-bal-ctr"),
    "ortho/patch_max": ([("orthofoundation", 224, "patch_max")], "p90-all"),
    "mri/patch_max": ([("mri_core", 224, "patch_max")], "mean-fluid"),
}


def member_oof(pool: Pool, name: str, seed: int) -> dict[str, np.ndarray]:
    sources, pooling = MEMBERS[name]
    x = pool.matrix(sources, parse_pooling(pooling))
    return probe.evaluate(x, pool.labels, seed=seed)["oof"]


def combine(oofs: list[dict[str, np.ndarray]], how: str) -> dict[str, np.ndarray]:
    """Average member predictions per label.

    Rank averaging is the safer default for an AUC target: members are separate
    logistic regressions whose probabilities are calibrated differently, and a
    plain mean lets the most confident member dominate for reasons unrelated to
    being the most correct one.
    """
    out = {}
    for label in data.LABELS:
        stack = np.stack([o[label] for o in oofs])
        if how == "rank":
            stack = np.stack([rankdata(row) / len(row) for row in stack])
        out[label] = stack.mean(0)
    return out


def macro(oof: dict[str, np.ndarray], labels: pd.DataFrame) -> float:
    return float(np.mean([roc_auc_score(labels[l].to_numpy(), oof[l]) for l in data.LABELS]))


def main(seeds: tuple[int, ...]) -> None:
    pool = Pool()
    names = list(MEMBERS)
    print(f"seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} repeats)\n")

    # Cache every member's OOF once per seed; ensembles are then free.
    cached: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        cached[seed] = {n: member_oof(pool, n, seed) for n in names}

    rows = []

    def record(label: str, subset: list[str], how: str):
        scores = [macro(combine([cached[s][n] for n in subset], how), pool.labels)
                  for s in seeds]
        rows.append({"ensemble": label, "n_members": len(subset), "combine": how,
                     "macro_auc": float(np.mean(scores)), "std": float(np.std(scores)),
                     "members": "+".join(subset)})
        print(f"{label:44s} {how:5s} macro {np.mean(scores):.4f} +/- {np.std(scores):.4f}",
              flush=True)

    print("== single members ==")
    for name in names:
        record(name, [name], "prob")

    print("\n== pairs ==")
    for pair in itertools.combinations(names, 2):
        for how in ("prob", "rank"):
            record("+".join(pair), list(pair), how)

    print("\n== larger ensembles ==")
    for size in (3, 4, 5):
        for subset in itertools.combinations(names, size):
            for how in ("prob", "rank"):
                record(f"{size} members: " + "+".join(s.split('/')[0][:5] for s in subset),
                       list(subset), how)

    frame = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    frame.to_csv(RESULTS / "ensemble.csv", index=False)
    print("\nTop 12:")
    print(frame.head(12)[["members", "combine", "macro_auc", "std"]]
          .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    main(tuple(range(args.first_seed, args.first_seed + args.n_seeds)))
