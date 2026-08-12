#!/usr/bin/env python3
"""Final ensemble, including an attention-MIL member.

Adds two things to `ensemble.py`: the stronger linear members found by the
orientation sweep, and the MIL model as a member in its own right.

MIL is individually the weakest thing built here (~0.59 against ~0.67 for a
linear probe), which is normally a reason to drop it. It is included anyway
because ensemble gain comes from decorrelation, not from member strength: MIL
learns a per-finding weighting over series descriptors, so its errors have no
particular reason to coincide with those of a fixed pooling followed by logistic
regression. Whether that outweighs its weakness is the question -- a member that
is both weak and correlated should hurt, and the answer is reported either way.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

import data
import mil as mil_module
import probe
from ensemble import combine, macro
from stage2 import Pool, parse_pooling

RESULTS = probe.FEATURES.parent / "results"

LINEAR_MEMBERS = {
    # Best pooling per source from the orientation sweep, which was scored on
    # seeds 100-109 -- hence the fresh seed block below.
    "ortho/cls": ([("orthofoundation", 224, "cls")], "mean-plane-inp90-axmean-ctr"),
    "ortho/patch_std": ([("orthofoundation", 224, "patch_std")], "mean-plane-inp90-axmean-ctr"),
    "mri/cls": ([("mri_core", 224, "cls")], "meanstd-plane-axmean-ctr"),
    "ortho/patch_max": ([("orthofoundation", 224, "patch_max")], "p90-all"),
}
MIL_MEMBERS = {"mil:ortho/cls": (("orthofoundation", 224, "cls"), "plane_embed")}


def main(seeds: tuple[int, ...]) -> None:
    pool = Pool()
    cached: dict[int, dict[str, dict]] = {s: {} for s in seeds}

    for name, (sources, pooling) in LINEAR_MEMBERS.items():
        x = pool.matrix(sources, parse_pooling(pooling))
        for seed in seeds:
            cached[seed][name] = probe.evaluate(x, pool.labels, seed=seed)["oof"]
        print(f"cached linear member {name}", flush=True)

    for name, (source, variant) in MIL_MEMBERS.items():
        bags = mil_module.Bags(*source, level=mil_module.TUNED["level"])
        fit_kwargs = {k: mil_module.TUNED[k]
                      for k in ("d_proj", "d_att", "weight_decay", "dropout")}
        for seed in seeds:
            cached[seed][name] = mil_module.evaluate(
                bags, variant, seed=seed, pca_dim=mil_module.TUNED["pca_dim"],
                **fit_kwargs)["oof"]
        print(f"cached MIL member {name}", flush=True)

    names = list(LINEAR_MEMBERS) + list(MIL_MEMBERS)
    rows = []

    def record(subset):
        scores = [macro(combine([cached[s][n] for n in subset], "rank"), pool.labels)
                  for s in seeds]
        rows.append({"members": "+".join(subset), "n": len(subset),
                     "has_mil": any(n.startswith("mil:") for n in subset),
                     "macro_auc": float(np.mean(scores)), "std": float(np.std(scores))})

    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            record(list(subset))

    frame = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    frame.to_csv(RESULTS / "ensemble2.csv", index=False)

    print(f"\nseeds {seeds[0]}-{seeds[-1]}, rank averaging\n")
    print("Top 10:")
    print(frame.head(10)[["members", "macro_auc", "std"]].to_string(index=False))

    # Does adding the MIL member to an otherwise-identical ensemble help?
    print("\nEffect of adding the MIL member (matched subsets):")
    linear_only = frame[~frame.has_mil].set_index("members")
    for members, row in linear_only.iterrows():
        with_mil = "+".join(members.split("+") + list(MIL_MEMBERS))
        match = frame[frame.members == with_mil]
        if len(match):
            delta = float(match.iloc[0].macro_auc) - float(row.macro_auc)
            print(f"  {members:52s} {row.macro_auc:.4f} -> "
                  f"{float(match.iloc[0].macro_auc):.4f}  ({delta:+.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Members were selected on seeds 100-119, so the ensemble is scored on a
    # disjoint block; otherwise the selection would leak into the headline.
    parser.add_argument("--first-seed", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()
    main(tuple(range(args.first_seed, args.first_seed + args.n_seeds)))
