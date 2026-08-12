#!/usr/bin/env python3
"""Cohort assembly and DICOM decoding for the frozen-backbone probe.

Decoding is the expensive half of this experiment (the backbones themselves run
at a few ms per slice), so every slice of every series is decoded exactly once
into a uint8 cache and reused across backbones, resolutions and pooling
strategies. The cohort is only 10.5k slices, which is small enough that nothing
needs to be subsampled at this stage -- subsampling is a pooling decision, and
deferring it keeps that axis free.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/from_host"
CACHE = Path(__file__).resolve().parent / "cache"

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 58 fully-labeled studies and their series metadata."""
    train = pd.read_csv(DATA / "train.csv")
    series = pd.read_csv(DATA / "train_series.csv")

    labeled = train.dropna(subset=LABELS).copy()
    labeled[LABELS] = labeled[LABELS].astype(int)
    labeled = labeled.sort_values("StudyInstanceUID").reset_index(drop=True)

    missing = [u for u in labeled.StudyInstanceUID
               if not (DATA / "train_series" / u).is_dir()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} labeled studies are not on disk")

    series = series[series.StudyInstanceUID.isin(labeled.StudyInstanceUID)]
    series = series.sort_values(["StudyInstanceUID", "SeriesInstanceUID"]).reset_index(drop=True)
    return labeled, series


# ---------------------------------------------------------------------------
# Slice decoding
# ---------------------------------------------------------------------------


def ordered_slice_paths(series_dir: Path) -> list[Path]:
    """Slice paths in through-plane order.

    InstanceNumber is trusted only when it is present and unique across the
    series; otherwise slices are ordered by their position projected onto the
    slice normal, which is well-defined even when instance numbering restarts or
    interleaves. Falling back to filename order would interleave a two-echo
    acquisition, so the caller is told when neither key is usable.
    """
    paths = sorted(p for p in series_dir.iterdir() if p.suffix == ".dcm")
    numbers, positions = [], []
    for path in paths:
        header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        try:
            numbers.append(float(header.InstanceNumber))
        except (AttributeError, TypeError, ValueError):
            numbers.append(float("nan"))
        orientation = getattr(header, "ImageOrientationPatient", None)
        position = getattr(header, "ImagePositionPatient", None)
        if orientation is not None and position is not None:
            normal = np.cross(np.array(orientation[:3], float), np.array(orientation[3:], float))
            positions.append(float(np.dot(np.array(position, float), normal)))
        else:
            positions.append(float("nan"))

    for key in (numbers, positions):
        if not any(np.isnan(key)) and len(set(key)) == len(key):
            return [p for _, p in sorted(zip(key, paths))]
    return paths


def render(path: Path, size: int) -> np.ndarray:
    """Decode one slice to a square uint8 array.

    Intensity is windowed to the 0.5-99.5 percentile rather than to the raw
    min/max that MRI-CORE's own dataloader uses: MR magnitude images routinely
    carry a few very bright voxels (flow artefact, fat at the coil), and a plain
    min-max lets one of them compress the entire joint into a narrow band.

    The image is letterboxed rather than stretched, so a rectangular field of
    view does not change the apparent aspect of cartilage and menisci.
    """
    dataset = pydicom.dcmread(path, force=True)
    pixels = dataset.pixel_array
    if pixels.ndim == 3:  # rare multi-frame or colour slice
        pixels = pixels[pixels.shape[0] // 2] if pixels.shape[-1] > 4 else pixels[..., 0]

    array = pixels.astype(np.float32)
    array = array * float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    array += float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)

    low, high = np.percentile(array, [0.5, 99.5])
    array = np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)
    if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        array = 1.0 - array

    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    scale = size / max(image.size)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.BILINEAR,
    )
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _decode_series(job: tuple[str, str, int]) -> tuple[str, str, int]:
    study, series_uid, size = job
    out = CACHE / f"slices{size}" / study / f"{series_uid}.npy"
    if out.exists():
        return study, series_uid, int(np.load(out, mmap_mode="r").shape[0])

    paths = ordered_slice_paths(DATA / "train_series" / study / series_uid)
    stack = np.stack([render(p, size) for p in paths]) if paths else np.zeros((0, size, size), np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, stack)
    os.replace(tmp, out)
    return study, series_uid, len(stack)


def build_cache(size: int = 448, workers: int = 16) -> pd.DataFrame:
    """Decode every slice of every series in the cohort into `cache/slices<size>/`.

    Returns the series index with an added `n_slices` column, which is also
    written alongside the cache so downstream steps never re-derive it.
    """
    _, series = cohort()
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID, size) for r in series.itertuples()]

    counts: dict[tuple[str, str], int] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (study, series_uid, n) in enumerate(pool.map(_decode_series, jobs, chunksize=1), 1):
            counts[(study, series_uid)] = n
            if i % 25 == 0 or i == len(jobs):
                print(f"  decoded {i}/{len(jobs)} series", flush=True)

    series = series.copy()
    series["n_slices"] = [counts[(r.StudyInstanceUID, r.SeriesInstanceUID)]
                          for r in series.itertuples()]
    index = CACHE / f"slices{size}" / "index.json"
    index.write_text(json.dumps(series.to_dict("records"), indent=1))
    return series


def load_series(study: str, series_uid: str, size: int = 448) -> np.ndarray:
    return np.load(CACHE / f"slices{size}" / study / f"{series_uid}.npy", mmap_mode="r")


if __name__ == "__main__":
    import sys

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 448
    studies, _ = cohort()
    print(f"cohort: {len(studies)} labeled studies")
    frame = build_cache(size)
    print(f"cached {frame.n_slices.sum()} slices across {len(frame)} series at {size}px")
