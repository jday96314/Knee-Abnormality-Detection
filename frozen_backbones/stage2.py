#!/usr/bin/env python3
"""Follow-ups to the main sweep: slice L2, head fusion, backbone fusion, stability.

The sweep picks a winner out of hundreds of configurations scored on the same 58
studies, so its top line is optimistically biased by construction. This script
does two things about that: it tests a few extensions the sweep could not reach
(concatenating read-out heads, concatenating backbones), and it re-scores the
leading configurations over many CV seeds so the reported number carries a
spread rather than a single lucky partition.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import data
import probe

RESULTS = Path(__file__).resolve().parent / "results"

Source = tuple[str, int, str]  # (backbone, size, head)


class Pool:
    """Caches feature files and pooled design matrices across many evaluations."""

    def __init__(self):
        self._features: dict[tuple[str, int], tuple] = {}
        self._pooled: dict[tuple, np.ndarray] = {}
        studies, _ = data.cohort()
        self.studies = studies
        self.ids = studies.StudyInstanceUID.tolist()
        self.labels = studies[data.LABELS]

    def matrix(self, sources: list[Source], pooling: probe.Pooling) -> np.ndarray:
        blocks = []
        for backbone, size, head in sources:
            key = (backbone, size, head, pooling)
            if key not in self._pooled:
                if (backbone, size) not in self._features:
                    self._features[(backbone, size)] = probe.load_features(backbone, size)
                heads, meta = self._features[(backbone, size)]
                self._pooled[key] = probe.pool_studies(heads[head], meta, self.ids, pooling)
            blocks.append(self._pooled[key])
        if len(blocks) == 1:
            return blocks[0]
        # Each source is z-scored before concatenation so a 1024-d block does not
        # simply outweigh a 768-d one through scale; StandardScaler inside the CV
        # fold then re-standardizes on training rows only.
        return np.hstack([(b - b.mean(0)) / np.maximum(b.std(0), 1e-8) for b in blocks])

    def score(self, sources: list[Source], pooling: probe.Pooling,
              seeds=(0,)) -> tuple[float, float, dict]:
        x = self.matrix(sources, pooling)
        runs = [probe.evaluate(x, self.labels, seed=s) for s in seeds]
        macros = [r["macro_auc"] for r in runs]
        per_label = {label: float(np.mean([r["per_label"][label] for r in runs]))
                     for label in data.LABELS}
        return float(np.mean(macros)), float(np.std(macros)), per_label


def label_row(name: str, sources, pooling, mean, std, per_label, dim) -> dict:
    return {
        "config": name,
        "sources": "+".join(f"{b}/{s}/{h}" for b, s, h in sources),
        "pooling": pooling.name(), "dim": dim,
        "macro_auc": mean, "macro_std": std,
        **{f"auc/{k}": v for k, v in per_label.items()},
    }


def main(seeds: tuple[int, ...], top_k: int) -> None:
    pool = Pool()
    sweep = pd.read_csv(RESULTS / "sweep.csv")
    rows = []

    def run(name, sources, pooling):
        x = pool.matrix(sources, pooling)
        mean, std, per_label = pool.score(sources, pooling, seeds)
        rows.append(label_row(name, sources, pooling, mean, std, per_label, x.shape[1]))
        print(f"{name:34s} {pooling.name():24s} d={x.shape[1]:6d} "
              f"macro {mean:.4f} +/- {std:.4f}", flush=True)

    # 1. Does unit-normalizing each slice before pooling help the leaders?
    print("== slice L2 normalization ==")
    leaders = sweep.head(top_k)
    seen = set()
    for row in leaders.itertuples():
        source = (row.backbone, int(row.size), row.head)
        base = parse_pooling(row.pooling)
        for l2 in (False, True):
            pooling = probe.Pooling(base.reduce, base.group, base.inner, base.across, base.central, l2)
            key = (source, pooling)
            if key in seen:
                continue
            seen.add(key)
            run(f"{row.backbone[:6]}/{row.size}/{row.head}", [source], pooling)

    # 2. Fuse read-out heads within a backbone, then fuse the two backbones.
    print("\n== head and backbone fusion ==")
    best = sweep.iloc[0]
    pooling = parse_pooling(best.pooling)
    per_backbone = {}
    for backbone in ("mri_core", "orthofoundation"):
        size = int(sweep[sweep.backbone == backbone].iloc[0]["size"])
        per_backbone[backbone] = size
        for heads in (["cls", "patch_mean"], ["cls", "patch_mean", "patch_max"]):
            sources = [(backbone, size, h) for h in heads]
            run(f"{backbone[:6]}/{size}/{'+'.join(h[:4] for h in heads)}", sources, pooling)

    for heads in (["cls"], ["cls", "patch_mean"]):
        sources = [(b, per_backbone[b], h) for b in per_backbone for h in heads]
        run(f"both/{'+'.join(h[:4] for h in heads)}", sources, pooling)

    frame = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    frame.to_csv(RESULTS / "stage2.csv", index=False)
    print("\nTop 12:")
    print(frame.head(12)[["config", "pooling", "dim", "macro_auc", "macro_std"]]
          .to_string(index=False))


def parse_pooling(name: str) -> probe.Pooling:
    """Inverse of Pooling.name(). "bal" is the legacy spelling of inner="mean"."""
    bits = name.split("-")
    inner = "mean" if "bal" in bits else ""
    across = ""
    for bit in bits:
        if bit.startswith("in"):
            inner = bit[2:]
        elif bit.startswith("ax"):
            across = bit[2:]
    return probe.Pooling(bits[0], bits[1], inner, across, "ctr" in bits, "l2" in bits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    main(tuple(range(args.seeds)), args.top_k)
