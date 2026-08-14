#!/usr/bin/env python3
"""Combine arms the way the public notebooks combine them, and score the result.

Their submission is not a model, it is a rank mean over twenty-odd members. Rank pooling
is the right operation for a metric that reads only order, and it is what makes their
score higher than any member of it — so a fair reading of "how well does this approach
work" has to measure the combination as well as the parts.

Two combinations are scored here, because they answer different questions:

    rank mean      the notebooks' own operation. Ordering only, so soft BCE is
                   undefined for it and only gold AUC is reported.
    probability    the arithmetic mean of the predicted probabilities, which stays a
                   probability and therefore still has a BCE.

Reads the `preds_*.npz` files `train.py` writes, so nothing is refitted here.

    python blend.py "radc: reference 130mm" "dinov2-224 x30"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import protocol  # noqa: E402

RESULTS = HERE / "results"


def find(name):
    hits = sorted(RESULTS.glob(f"preds_*{name}*.npz"))
    if not hits:
        raise FileNotFoundError(f"no saved predictions matching {name!r}")
    if len(hits) > 1:
        raise ValueError(f"{name!r} matches {len(hits)}: {[h.name for h in hits]}")
    return hits[0]


def load(name):
    d = np.load(find(name), allow_pickle=True)
    # Predictions are stored in the order the folds ran, so two members are only
    # comparable row-by-row once both are put back into study order.
    order = np.argsort(d["val_rows"])
    return {"rows": d["val_rows"][order], "p_val": d["p_val"][order],
            "y_val": d["y_val"][order], "p_gold": d["p_gold"], "y_gold": d["y_gold"]}


def ranks(p):
    return pd.DataFrame(p).rank(pct=True).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("members", nargs="+")
    args = ap.parse_args()

    loaded = [load(m) for m in args.members]
    base = loaded[0]
    for other in loaded[1:]:
        if not np.array_equal(base["rows"], other["rows"]):
            raise ValueError("members were scored on different studies; they cannot be "
                             "blended without refitting one of them on the other's folds")

    rows = []
    for name, m in zip(args.members, loaded):
        rows.append({"model": name,
                     "val_soft_bce": float(protocol.soft_bce(m["y_val"], m["p_val"]).mean()),
                     "gold_auc": float(np.nanmean(protocol.gold_auc(m["y_gold"], m["p_gold"])))})

    prob_val = np.mean([m["p_val"] for m in loaded], axis=0)
    prob_gold = np.mean([m["p_gold"] for m in loaded], axis=0)
    rows.append({"model": f"probability mean of {len(loaded)}",
                 "val_soft_bce": float(protocol.soft_bce(base["y_val"], prob_val).mean()),
                 "gold_auc": float(np.nanmean(protocol.gold_auc(base["y_gold"], prob_gold)))})

    rank_val = np.mean([ranks(m["p_val"]) for m in loaded], axis=0)
    rank_gold = np.mean([ranks(m["p_gold"]) for m in loaded], axis=0)
    rows.append({"model": f"rank mean of {len(loaded)} (the notebooks' operation)",
                 "val_soft_bce": float("nan"),
                 "gold_auc": float(np.nanmean(protocol.gold_auc(base["y_gold"], rank_gold)))})
    # A rank is not a probability, so its BCE would be a number about the transform
    # rather than about the model. Reported as absent instead of as a large value.
    _ = rank_val

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    frame.to_csv(RESULTS / "blend.csv", index=False)


if __name__ == "__main__":
    main()
