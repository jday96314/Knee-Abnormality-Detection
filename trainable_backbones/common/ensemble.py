#!/usr/bin/env python3
"""Evaluate ensembles of saved architecture predictions.

The plan is explicit that the volumetric/video branch should be judged "largely on ensemble
gain, not solo AUC", and that prediction-level ensembling beat feature-level fusion in the
earlier frozen work. That only becomes measurable if each architecture writes out its
validation and gold predictions, which `--save-preds` on each trainer does.

    python ensemble.py                       # every saved member, plus all subsets
    python ensemble.py --members 01 03       # a specific combination

Averaging is in logit space. Probability averaging is the other obvious choice, but the
selection metric here is BCE, and averaging probabilities systematically pulls confident
agreeing members towards the middle in a way that costs calibrated log-loss.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

import protocol

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Only the architecture directories of this ladder. A bare `0*` glob would also pick up
# sibling workstreams whose predictions may not share this split, silently corrupting the
# comparison rather than failing.
ARCH_DIRS = ["01_frozen_pooling", "02_series_attention", "03_finetuned_2p5d",
             "04_hierarchical", "05_volumetric"]


def discover() -> dict[str, dict]:
    """Every `results/preds_*.npz` under an architecture directory, keyed by a short name."""
    found = {}
    for path in sorted(p for d in ARCH_DIRS for p in (ROOT / d).glob("results/preds_*.npz")):
        data = np.load(path)
        name = f"{path.parts[-3].split('_')[0]}:{path.stem.removeprefix('preds_')}"
        found[name] = {"val": data["val"], "gold": data["gold"], "path": path}
    return found


def score(members, y_val, y_gold):
    mean_val = sigmoid(np.mean([logit(m["val"]) for m in members], axis=0))
    mean_gold = sigmoid(np.mean([logit(m["gold"]) for m in members], axis=0))
    return (float(protocol.soft_bce(y_val, mean_val).mean()),
            float(np.nanmean(protocol.gold_auc(y_gold, mean_gold))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--members", nargs="*", default=None)
    p.add_argument("--max-size", type=int, default=4)
    args = p.parse_args()

    import dataset as ds
    studies, _ = ds.all_studies()
    import features as featlib
    _, index, _ = featlib.load("mri_core", 224)
    studies = studies[studies.StudyInstanceUID.isin(set(index["study"]))].reset_index(drop=True)
    split = protocol.make_split(studies)
    y_val = protocol.soft_targets(studies, split["val"])
    y_gold = protocol.gold_targets(studies, split["gold"])

    found = discover()
    if args.members:
        found = {k: v for k, v in found.items()
                 if any(k.startswith(m) or m in k for m in args.members)}
    if not found:
        raise SystemExit("no saved predictions found; run each trainer with --save-preds")

    rows = []
    names = list(found)
    for size in range(1, min(args.max_size, len(names)) + 1):
        for combo in itertools.combinations(names, size):
            bce, auc = score([found[n] for n in combo], y_val, y_gold)
            rows.append((size, "+".join(combo), bce, auc))

    rows.sort(key=lambda r: r[2])
    print(f"{'n':>2}  {'members':<52} {'val_soft_bce':>12} {'gold_auc':>9}")
    for size, combo, bce, auc in rows:
        print(f"{size:>2}  {combo:<52} {bce:>12.4f} {auc:>9.4f}")

    solo = {r[1]: r[2] for r in rows if r[0] == 1}
    best_solo = min(solo.values())
    best_all = rows[0]
    print(f"\nbest single member: {best_solo:.4f}")
    print(f"best ensemble:      {best_all[2]:.4f}  ({best_all[1]})")
    print(f"ensemble gain:      {best_solo - best_all[2]:+.4f} BCE")


if __name__ == "__main__":
    main()
