#!/usr/bin/env python3
"""Run a frozen backbone over every cached slice and store per-slice features.

Nothing is pooled here. The output is one row per slice, tagged with the study,
series and through-plane position it came from, so that pooling -- the axis this
experiment is actually about -- can be re-run without touching a GPU.

Each backbone emits several read-out heads per slice (CLS, patch mean/max/std);
they are stored side by side and selected downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import backbones
import data

FEATURES = Path(__file__).resolve().parent / "features"


@torch.no_grad()
def extract(backbone_name: str, size: int, cache_size: int = 448,
            batch_size: int = 32, device: str = "cuda") -> Path:
    model = backbones.load_backbone(backbone_name, device=device, input_size=size)
    index = json.loads((data.CACHE / f"slices{cache_size}" / "index.json").read_text())

    mean = torch.tensor(backbones.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(backbones.IMAGENET_STD, device=device).view(1, 3, 1, 1)

    heads: dict[str, list[np.ndarray]] = {}
    rows: list[dict] = []

    for n_done, record in enumerate(index, 1):
        study, series_uid = record["StudyInstanceUID"], record["SeriesInstanceUID"]
        stack = np.asarray(data.load_series(study, series_uid, cache_size))
        if len(stack) == 0:
            continue

        for start in range(0, len(stack), batch_size):
            chunk = torch.from_numpy(stack[start:start + batch_size]).to(device)
            images = chunk.float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
            if images.shape[-1] != size:
                images = torch.nn.functional.interpolate(
                    images, size=(size, size), mode="bilinear", align_corners=False)
            images = (images - mean) / std

            for name, value in model(images).items():
                heads.setdefault(name, []).append(value.half().cpu().numpy())

        # Position is stored normalized so that pooling rules can refer to "the
        # middle of the stack" across series of 11 and 320 slices alike.
        n = len(stack)
        rows += [{
            "study": study,
            "series": series_uid,
            "plane": record["Anatomical_Plane"],
            "fluid": int(record["Fluid_Sensitive"]),
            "fatsat": int(record["Fat_Suppression"]),
            "pos": (i + 0.5) / n,
        } for i in range(n)]

        if n_done % 50 == 0 or n_done == len(index):
            print(f"  {backbone_name}@{size}: {n_done}/{len(index)} series", flush=True)

    payload = {name: np.concatenate(parts) for name, parts in heads.items()}

    # A backbone run in the wrong dtype returns all-NaN without raising, and the
    # failure only surfaces much later as a scaler error. Fail here instead.
    for name, value in payload.items():
        if not np.isfinite(value).all():
            n_bad = int((~np.isfinite(value)).any(axis=1).sum())
            raise FloatingPointError(
                f"{backbone_name}@{size} head '{name}': {n_bad}/{len(value)} "
                f"slices are non-finite")

    out = FEATURES / f"{backbone_name}_{size}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    (out.with_suffix(".index.json")).write_text(json.dumps(rows))

    shapes = {k: v.shape for k, v in payload.items()}
    print(f"saved {out.name}: {shapes}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["mri_core", "orthofoundation"])
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    extract(args.backbone, args.size, batch_size=args.batch_size)
