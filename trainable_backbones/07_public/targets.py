#!/usr/bin/env python3
"""Training targets, from this repository's blend or from the public notebooks' teachers.

The evaluation is fixed — soft BCE against `llm_classifiers/blend` on the held-out fifth,
ROC AUC over the 58 gold studies — but what a model is *trained* on is a free variable,
and the public notebooks make a different choice: three independently produced readings
of the same reports, averaged, with a per-cell weight that falls when the three disagree
or when the average sits near a half.

Keeping the measurement fixed while varying the teacher is the only way to tell whether
the notebooks' image models are better than this repository's or merely better taught.

Every source returns `(Y, W)` aligned to the rows of the `studies` frame it is given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

FILES = {
    "report_v2": "report_labels_v2.csv",
    "llm_v2": "llm_labels_v2.csv",
    "gpt56": "labels_llm_gpt56sol.csv",
    "v4_blend": "llm_labels_v4_blend.csv",
    "report_gpt56": "report_labels_gpt56sol.csv",
}
# The three the notebooks average. `report_labels_v2` also carries per-cell confidences;
# the other two do not, which is why agreement between the three has to stand in for it.
PUBLIC_TEACHERS = ["report_v2", "llm_v2", "gpt56"]


def _read(name, studies):
    frame = pd.read_csv(ASSETS / FILES[name])
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{name}: duplicate study")
    aligned = studies[["StudyInstanceUID"]].merge(
        frame[["StudyInstanceUID"] + TARGETS], on="StudyInstanceUID", how="left")
    return aligned[TARGETS].to_numpy(np.float64)


def repo_blend(studies):
    """This repository's own soft labels — the column the evaluation also reads."""
    return studies[TARGETS].to_numpy(np.float32), np.ones((len(studies), len(TARGETS)), np.float32)


def single(name, studies):
    """One public teacher on its own, unweighted."""
    y = _read(name, studies)
    w = np.isfinite(y).astype(np.float32)
    return np.nan_to_num(y, nan=0.5).astype(np.float32), w


def public_mean(studies):
    """The notebooks' three-teacher average, with their agreement/certainty weight.

    A cell where the three readings disagree is one where the reports are ambiguous or
    one reader misparsed, and a cell near 0.5 is one where nobody committed. Down-weighting
    both is cheaper than adjudicating them, and it is what the notebooks do.
    """
    cube = np.stack([_read(t, studies) for t in PUBLIC_TEACHERS])
    available = np.isfinite(cube).sum(0)
    y = np.nanmean(cube, axis=0)
    disagreement = np.nanmean(np.abs(cube - y[None]), axis=0)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0, 1)
    certainty = np.clip(2.0 * np.abs(y - .5), 0, 1)
    w = (.15 + .85 * (.65 * agreement + .35 * certainty)).astype(np.float32)
    # A study no teacher covers gets a half and no weight rather than a NaN gradient.
    w[available == 0] = 0.0
    return np.nan_to_num(y, nan=0.5).astype(np.float32), w


def public_mean_unweighted(studies):
    y, _ = public_mean(studies)
    return y, np.ones_like(y)


SOURCES = {
    "repo_blend": repo_blend,
    "public_mean": public_mean,
    "public_mean_flat": public_mean_unweighted,
    "report_v2": lambda s: single("report_v2", s),
    "llm_v2": lambda s: single("llm_v2", s),
    "gpt56": lambda s: single("gpt56", s),
    "v4_blend": lambda s: single("v4_blend", s),
}


def build(name, studies):
    return SOURCES[name](studies)
