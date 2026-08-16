#!/usr/bin/env python3
"""Slice sampling and parameterised pixel augmentation for architecture 3.

The original run used one hard-coded augmentation bundle, which meant its contribution was
never measured. Here every transform has a strength dial, grouped into six interpretable
knobs so a search has something tractable to tune rather than twenty raw magnitudes:

    geom       rotation / translation / scale, synchronised within a series
    intensity  gamma, brightness, contrast
    noise      additive Gaussian (Rician-ish at these amplitudes)
    bias       smooth low-frequency multiplicative field — the classic MRI artefact
    blur       resolution degradation via downsample-and-restore
    erase      random rectangular occlusion (cutout)

plus two structural dials that act on *which* images are drawn rather than their pixels:
`phase` (slice-position jitter) and `series_dropout`.

Setting every dial to 0.0 disables augmentation entirely, which is the control the original
four configurations never ran.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import dataset as ds
import protocol
from model import PLANE_INDEX


@dataclass
class AugmentConfig:
    """All strengths are 0..1 multipliers on a per-transform maximum."""

    geom: float = 0.0
    intensity: float = 0.0
    noise: float = 0.0
    bias: float = 0.0
    blur: float = 0.0
    erase: float = 0.0
    phase: float = 0.0
    series_dropout: float = 0.0

    @staticmethod
    def preset(name: str) -> "AugmentConfig":
        return {
            "none": AugmentConfig(),
            # The bundle the original four configurations used, expressed in these units:
            # +-6 deg, +-5% shift, 0.95-1.05 scale, gamma 0.8-1.25, 30% light noise.
            "original": AugmentConfig(geom=0.4, intensity=0.5, noise=0.3, phase=0.33),
            "light": AugmentConfig(geom=0.25, intensity=0.3, noise=0.2, phase=0.25),
            "medium": AugmentConfig(geom=0.5, intensity=0.5, noise=0.4, bias=0.3,
                                    blur=0.2, erase=0.2, phase=0.5, series_dropout=0.1),
            "heavy": AugmentConfig(geom=0.85, intensity=0.8, noise=0.7, bias=0.6,
                                   blur=0.5, erase=0.5, phase=0.8, series_dropout=0.25),
        }[name]


def augment_series(arr: torch.Tensor, rng, cfg: AugmentConfig) -> torch.Tensor:
    """[K, H, W] slices of ONE series -> augmented, with geometry shared across K.

    Geometry is synchronised deliberately: the slices are one acquisition of one knee, so
    rotating them independently would fabricate anatomy that cannot occur. Flips are absent
    at every strength — a left-right flip on a sagittal knee exchanges anterior for
    posterior, and the VLM experiments measured flip TTA costing 0.061 AUC on ACL.
    """
    n, h, w = arr.shape

    if cfg.geom > 0:
        angle = float(rng.uniform(-15, 15) * cfg.geom) * np.pi / 180
        scale = 1.0 + float(rng.uniform(-0.12, 0.12) * cfg.geom)
        tx, ty = rng.uniform(-0.12, 0.12, 2) * cfg.geom
        cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0).expand(n, -1, -1), (n, 1, h, w),
                             align_corners=False)
        arr = F.grid_sample(arr.unsqueeze(1), grid, align_corners=False,
                            padding_mode="zeros").squeeze(1)

    if cfg.intensity > 0:
        gamma = float(np.exp(rng.uniform(-0.5, 0.5) * cfg.intensity))
        arr = arr.clamp(min=0).pow(gamma)
        arr = arr * (1 + float(rng.uniform(-0.25, 0.25) * cfg.intensity))
        arr = arr + float(rng.uniform(-0.1, 0.1) * cfg.intensity)

    if cfg.bias > 0 and rng.random() < 0.5:
        # A smooth multiplicative field built by upsampling a 4x4 random grid: this is what
        # coil sensitivity inhomogeneity looks like, and it is the one MRI-specific artefact
        # that a windowed 8-bit slice still carries.
        low = torch.from_numpy(rng.normal(0, 0.5 * cfg.bias, (1, 1, 4, 4)).astype(np.float32))
        field = F.interpolate(low, size=(h, w), mode="bicubic", align_corners=False)
        arr = arr * field.squeeze(0).exp()

    if cfg.blur > 0 and rng.random() < 0.4:
        factor = 1 + rng.random() * 2 * cfg.blur          # 1x .. 3x downsample
        small = max(8, int(h / (1 + factor)))
        arr = F.interpolate(F.interpolate(arr.unsqueeze(1), size=(small, small),
                                          mode="area"),
                            size=(h, w), mode="bilinear", align_corners=False).squeeze(1)

    if cfg.noise > 0 and rng.random() < 0.2 + 0.6 * cfg.noise:
        arr = arr + torch.randn_like(arr) * float(rng.uniform(0.005, 0.06) * cfg.noise)

    if cfg.erase > 0 and rng.random() < cfg.erase:
        eh, ew = (int(h * rng.uniform(0.08, 0.3) * cfg.erase ** 0.5),
                  int(w * rng.uniform(0.08, 0.3) * cfg.erase ** 0.5))
        if eh > 0 and ew > 0:
            top, left = int(rng.integers(0, h - eh + 1)), int(rng.integers(0, w - ew + 1))
            arr[:, top:top + eh, left:left + ew] = float(rng.random())

    return arr.clamp(0, 1)


class StudySlices(Dataset):
    """A stratified, re-drawn sample of slices per study.

    A full pass over every slice of every study is ~626k encoder calls per epoch. Sampling
    `n_series x n_slices` is both the compute budget and the strongest augmentation
    available, since a different sample is drawn every epoch.
    """

    def __init__(self, studies, series_index, rows, size=224, n_series=6, n_slices=4,
                 train=False, seed=0, aug: AugmentConfig | None = None):
        self.uids = studies.iloc[rows]["StudyInstanceUID"].to_numpy()
        self.targets = protocol.soft_targets(studies, rows)
        self.size, self.n_series, self.n_slices, self.train = size, n_series, n_slices, train
        self.aug = aug or AugmentConfig()
        self.rng = np.random.default_rng(seed)
        self.by_study: dict[str, list] = {}
        for r in series_index.itertuples():
            self.by_study.setdefault(r.StudyInstanceUID, []).append(
                (r.SeriesInstanceUID, r.Anatomical_Plane, int(r.n_slices)))

    def __len__(self):
        return len(self.uids)

    def _pick_series(self, entries, rng):
        """One series per plane first, then extras — a study's only axial beats its 4th sagittal."""
        by_plane: dict[str, list] = {}
        for e in entries:
            by_plane.setdefault(e[1], []).append(e)
        picked, spare = [], []
        for plane in sorted(by_plane):
            group = by_plane[plane]
            lead = int(rng.integers(0, len(group))) if self.train else 0
            picked.append(group[lead])
            spare += [g for i, g in enumerate(group) if i != lead]
        if self.train:
            spare = [spare[i] for i in rng.permutation(len(spare))]
            if self.aug.series_dropout > 0 and len(picked) > 1:
                keep = [p for p in picked if rng.random() >= self.aug.series_dropout]
                picked = keep or picked[:1]
        return (picked + spare)[: self.n_series]

    def __getitem__(self, i):
        rng = (np.random.default_rng(self.rng.integers(1 << 31)) if self.train
               else np.random.default_rng(i))
        uid = self.uids[i]
        entries = [e for e in self.by_study.get(uid, []) if e[2] > 0]
        chosen = self._pick_series(entries, rng) if entries else []

        images = torch.zeros(self.n_series, self.n_slices, self.size, self.size)
        plane = torch.zeros(self.n_series, dtype=torch.long)
        mask = torch.zeros(self.n_series, dtype=torch.bool)

        for j, (series_uid, plane_name, _) in enumerate(chosen):
            stack = ds.load_series(uid, series_uid, self.size)
            if len(stack) == 0:
                continue
            centres = np.linspace(0.15, 0.85, self.n_slices)
            if self.train and self.aug.phase > 0:
                centres = np.clip(centres + rng.normal(0, 0.15 * self.aug.phase,
                                                       self.n_slices), 0.02, 0.98)
            idx = np.clip((centres * len(stack)).astype(int), 0, len(stack) - 1)
            arr = torch.from_numpy(np.asarray(stack[idx], dtype=np.float32) / 255.0)
            if self.train:
                arr = augment_series(arr, rng, self.aug)
            images[j] = arr
            plane[j] = PLANE_INDEX.get(plane_name, len(PLANE_INDEX))
            mask[j] = True

        if not mask.any():
            mask[0] = True
        return images, plane, mask, torch.from_numpy(self.targets[i])


def load_cohort(size: int = 224):
    """Studies and per-series slice counts, filtered exactly as architectures 1-2 filter."""
    import json

    studies, _ = ds.all_studies()
    series_index = ds.pd.DataFrame(json.loads(
        (ds.CACHE / f"slices{size}" / "index.json").read_text()))
    series_index = series_index[series_index.n_slices > 0]
    studies = studies[studies.StudyInstanceUID.isin(
        set(series_index["StudyInstanceUID"]))].reset_index(drop=True)
    return studies, series_index
