#!/usr/bin/env python3
"""Extract frozen-backbone features for every slice of every training study.

The frozen architectures (#1 and #2 in the plan) never touch a GPU during training —
they consume these features. Extracting once and pooling many times is what makes a wide
architecture sweep affordable, and it is the reason the earlier `frozen_backbones`
experiments could try 480 pooling configurations.

Storage is one memory-mappable float16 array per backbone plus a row index, so a training
run loads only the rows it needs rather than 100+ GB of slices.

    python common/features.py mri_core --size 224
    python common/features.py orthofoundation --size 224
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frozen_backbones"))
import backbones  # noqa: E402

import dataset  # noqa: E402

FEATURES = Path(__file__).resolve().parents[1] / "features"


@torch.no_grad()
def extract(name: str, size: int, batch_size: int, heads: tuple[str, ...], device="cuda"):
    """One row per slice, tagged with where it came from so pooling can be redone cheaply."""
    FEATURES.mkdir(parents=True, exist_ok=True)
    out_npy = FEATURES / f"{name}_{size}.f16.npy"
    out_index = FEATURES / f"{name}_{size}.index.json"

    _, series = dataset.all_studies()
    index_path = dataset.CACHE / f"slices{size}" / "index.json"
    if index_path.exists():
        series = dataset.pd.DataFrame(json.loads(index_path.read_text()))

    model = backbones.load_backbone(name, device=device, input_size=size)
    mean = torch.tensor(backbones.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(backbones.IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # Two passes: size the output array, then fill it. Writing straight into a memmap
    # keeps peak RAM at one series regardless of how many slices the cohort has.
    # The cache index already records n_slices. Re-deriving it by opening all 24k arrays
    # costs tens of thousands of filesystem round trips and buys nothing.
    rows, total = [], 0
    for r in series.itertuples():
        n = int(getattr(r, "n_slices", 0) or 0)
        for i in range(n):
            rows.append({
                "study": r.StudyInstanceUID, "series": r.SeriesInstanceUID,
                "plane": r.Anatomical_Plane, "fluid": int(r.Fluid_Sensitive),
                "fatsat": int(r.Fat_Suppression), "pos": (i + 0.5) / max(n, 1),
            })
        total += n
    print(f"{name}: {len(series)} series, {total:,} slices", flush=True)

    dim = None
    store = None
    written = 0
    for s_i, r in enumerate(series.itertuples(), 1):
        try:
            stack = np.asarray(dataset.load_series(r.StudyInstanceUID, r.SeriesInstanceUID, size))
        except FileNotFoundError:
            continue
        if len(stack) == 0:
            continue
        for start in range(0, len(stack), batch_size):
            chunk = stack[start:start + batch_size]
            x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device).float().div_(255)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1).sub_(mean).div_(std)
            feats = model(x)
            vec = torch.cat([feats[h] for h in heads], dim=1).float().cpu().numpy()
            if store is None:
                dim = vec.shape[1]
                store = np.lib.format.open_memmap(
                    out_npy, mode="w+", dtype=np.float16, shape=(total, dim))
            store[written:written + len(vec)] = vec.astype(np.float16)
            written += len(vec)
        if s_i % 500 == 0:
            print(f"  {s_i}/{len(series)} series, {written:,}/{total:,} slices", flush=True)

    store.flush()
    out_index.write_text(json.dumps({"heads": list(heads), "dim": dim, "rows": rows}))
    print(f"wrote {out_npy} ({written:,} x {dim}) and {out_index.name}")


def load(name: str, size: int):
    """Memory-mapped features plus the per-slice index, as a DataFrame."""
    meta = json.loads((FEATURES / f"{name}_{size}.index.json").read_text())
    array = np.load(FEATURES / f"{name}_{size}.f16.npy", mmap_mode="r")
    return array, dataset.pd.DataFrame(meta["rows"]), meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backbone")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--heads", default="cls,patch_mean,patch_max,patch_std")
    a = p.parse_args()
    extract(a.backbone, a.size, a.batch_size, tuple(a.heads.split(",")))
