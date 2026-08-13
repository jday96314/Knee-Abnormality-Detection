#!/usr/bin/env python3
"""Data layer for training vision models on the pseudo-labelled studies.

Extends `frozen_backbones/data.py`, which is scoped to the 58 gold studies, to the full
4,407-study training set. Slice decoding, ordering and rendering are reused unchanged so
the two code paths cannot drift apart.

The supervision here is the blended soft targets from `llm_classifiers/blend`, not the
gold labels. The 58 gold studies are held out of tuning entirely and used only as a
locked auxiliary measurement, per the plan.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frozen_backbones"))
import data as fb  # noqa: E402  (slice ordering/rendering, reused verbatim)

DATA = ROOT / "data" / "from_host"

# The slice cache lives on local NVMe, not beside the code. The repository sits on a CIFS
# share that is both slower to read and near capacity, and CIFS cannot hold a symlink to
# redirect it — so the location is an explicit setting rather than a link.
CACHE = Path(os.environ.get("KNEE_SLICE_CACHE", "/mnt/data01/knee_cache"))
LABELS = fb.LABELS

SOFT_TARGETS = ROOT / "llm_classifiers" / "blend" / "predictions" / "blended_predictions.csv"


def all_studies() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every training study that is fully on disk, with its series metadata.

    Returns (studies, series). `studies` carries the twelve soft targets plus an
    `is_gold` flag; gold rows additionally carry the hard labels under `<label>__gold`.
    """
    train = pd.read_csv(DATA / "train.csv")
    series = pd.read_csv(DATA / "train_series.csv")
    soft = pd.read_csv(SOFT_TARGETS)

    expected = series.groupby("StudyInstanceUID")["SeriesInstanceUID"].apply(set)
    on_disk = {p.name for p in (DATA / "train_series").iterdir() if p.is_dir()}
    complete = [
        u for u in train.StudyInstanceUID
        if u in on_disk
        and {p.name for p in (DATA / "train_series" / u).iterdir() if p.is_dir()} == expected.get(u, set())
    ]

    gold = train.dropna(subset=LABELS)[["StudyInstanceUID", *LABELS]].copy()
    gold.columns = ["StudyInstanceUID"] + [f"{c}__gold" for c in LABELS]

    studies = (
        soft[soft.StudyInstanceUID.isin(complete)]
        .merge(gold, on="StudyInstanceUID", how="left")
        .reset_index(drop=True)
    )
    studies["is_gold"] = studies[f"{LABELS[0]}__gold"].notna()
    series = series[series.StudyInstanceUID.isin(studies.StudyInstanceUID)].reset_index(drop=True)
    return studies, series


def _decode(job):
    study, series_uid, size = job
    out = CACHE / f"slices{size}" / study / f"{series_uid}.npy"
    if out.exists():
        try:
            return study, series_uid, int(np.load(out, mmap_mode="r").shape[0])
        except Exception:
            out.unlink(missing_ok=True)  # truncated by an interrupted run; redo it
    paths = fb.ordered_slice_paths(DATA / "train_series" / study / series_uid)
    # At 4,407 studies the archive contains a few truncated DICOMs ("pixel data is less
    # than expected"). One of them must not be able to kill a multi-hour decode, so a
    # slice that will not read is dropped and counted rather than raised. Losing one
    # slice from a 30-slice series is immaterial to pooled features; losing the run is not.
    rendered, bad = [], 0
    for path in paths:
        try:
            rendered.append(fb.render(path, size))
        except Exception:  # noqa: BLE001 - corrupt pixel data, truncated file, odd transfer syntax
            bad += 1
    if bad:
        print(f"  skipped {bad}/{len(paths)} unreadable slices in {study[-8:]}/{series_uid[-8:]}",
              flush=True)
    stack = np.stack(rendered) if rendered else np.zeros((0, size, size), np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, stack)
    os.replace(tmp, out)
    return study, series_uid, len(stack)


def build_cache(size: int = 224, workers: int = 14) -> pd.DataFrame:
    """Decode every slice of every series into `cache/slices<size>/`.

    Resumable: a series already on disk is skipped, so an interrupted run costs only
    the series it was mid-way through.
    """
    _, series = all_studies()
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID, size) for r in series.itertuples()]
    counts = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (study, uid, n) in enumerate(pool.map(_decode, jobs, chunksize=4), 1):
            counts[(study, uid)] = n
            if i % 250 == 0 or i == len(jobs):
                print(f"  decoded {i}/{len(jobs)} series", flush=True)
    series = series.copy()
    series["n_slices"] = [counts[(r.StudyInstanceUID, r.SeriesInstanceUID)] for r in series.itertuples()]
    (CACHE / f"slices{size}" / "index.json").write_text(json.dumps(series.to_dict("records")))
    return series


def load_series(study: str, series_uid: str, size: int = 224) -> np.ndarray:
    return np.load(CACHE / f"slices{size}" / study / f"{series_uid}.npy", mmap_mode="r")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 224
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    s = build_cache(size, workers)
    print(f"cached {len(s)} series, {s.n_slices.sum():,} slices at {size}px")
