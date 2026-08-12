#!/usr/bin/env python3
"""Hierarchical pooling: collapse each series first, then pool across series/planes.

The first sweep treated a study as one flat bag of slices, with a single
exception (`bal`, which averaged within series before averaging across them).
That leaves the interesting half of the design space untested. A study is really
a three-level object -- slices inside a series, series inside a plane, planes
inside the study -- and the reduction used at each level can differ.

The motivating asymmetry: a mean over 180 slices dilutes anything focal. A
fracture on two slices of a 40-slice sagittal series moves a study-level mean by
about 1%, but if that series is collapsed by max first, the finding arrives at
the outer reduction at full strength. Diffuse findings (effusion, OA) should
prefer the opposite. This sweep asks which level wants which reduction.

Axes:
  inner   -- how each series is collapsed  ("" = no series level, flat bag)
  reduce  -- how series descriptors combine within a group
  group   -- whether planes are kept apart
  across  -- how plane descriptors combine ("" = concatenate, as before)
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

import probe
from sweep import RESULTS, run

# 224px only, and only the read-out heads that led the first sweep: resolution
# and head were settled there and re-testing them here would just widen the
# multiple-comparisons problem without informing the pooling question.
SOURCES = [
    ("mri_core", 224, "cls"),
    ("orthofoundation", 224, "patch_std"),
    ("orthofoundation", 224, "cls"),
]

INNER = ["", "mean", "max", "p90"]      # "" reproduces the flat-bag baseline
OUTER = ["mean", "max", "meanmax"]
CENTRAL = [False, True]


def poolings() -> list[probe.Pooling]:
    out = []
    for inner, reduce_, central in itertools.product(INNER, OUTER, CENTRAL):
        out.append(probe.Pooling(reduce_, "all", inner, "", central))
        out.append(probe.Pooling(reduce_, "plane", inner, "", central))
        for across in ("mean", "max"):
            out.append(probe.Pooling(reduce_, "plane", inner, across, central))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="hier.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = [(b, s, h, p) for b, s, h in SOURCES for p in poolings()]
    print(f"{len(configs)} configurations ({len(poolings())} poolings "
          f"x {len(SOURCES)} sources)\n")
    frame = run(configs, args.out, seed=args.seed)
    print("\nTop 20:")
    print(frame.head(20)[["backbone", "head", "pooling", "dim", "macro_auc"]]
          .to_string(index=False))
