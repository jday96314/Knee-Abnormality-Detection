#!/usr/bin/env python3
"""Train the volumetric family (architectures 5 and 6).

    python train.py --encoder medicalnet --epochs 15
    python train.py --compare               # both encoders, frozen and partially unfrozen

Each series becomes one fixed-size volume: `depth` slices sampled stratified through the
stack and bilinearly resized to `size`. Depth is resampled rather than cropped because knee
series are strongly anisotropic — throwing away slices to reach a fixed depth would discard
a different fraction of the knee in every study.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import dataset as ds  # noqa: E402
import protocol  # noqa: E402
from model import PLANE_INDEX, VolumetricModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StudyVolumes(Dataset):
    def __init__(self, studies, series_index, rows, size=112, depth=16, n_series=6,
                 train=False, seed=0):
        self.uids = studies.iloc[rows]["StudyInstanceUID"].to_numpy()
        self.targets = protocol.soft_targets(studies, rows)
        self.size, self.depth, self.n_series, self.train = size, depth, n_series, train
        self.rng = np.random.default_rng(seed)
        self.by_study: dict[str, list] = {}
        for r in series_index.itertuples():
            self.by_study.setdefault(r.StudyInstanceUID, []).append(
                (r.SeriesInstanceUID, r.Anatomical_Plane, int(r.Fluid_Sensitive),
                 int(r.Fat_Suppression), int(r.n_slices)))

    def __len__(self):
        return len(self.uids)

    def _pick(self, entries, rng):
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
        return (picked + spare)[:self.n_series]

    def __getitem__(self, i):
        rng = np.random.default_rng(self.rng.integers(1 << 31)) if self.train else np.random.default_rng(i)
        uid = self.uids[i]
        entries = [e for e in self.by_study.get(uid, []) if e[4] > 0]
        chosen = self._pick(entries, rng) if entries else []

        vols = torch.zeros(self.n_series, self.depth, self.size, self.size)
        plane = torch.zeros(self.n_series, dtype=torch.long)
        fluid = torch.zeros(self.n_series, dtype=torch.long)
        fatsat = torch.zeros(self.n_series, dtype=torch.long)
        mask = torch.zeros(self.n_series, dtype=torch.bool)

        for j, (series_uid, plane_name, fl, fs, _) in enumerate(chosen):
            stack = ds.load_series(uid, series_uid, 224)
            if len(stack) == 0:
                continue
            pos = np.linspace(0, len(stack) - 1, self.depth)
            if self.train:
                pos = np.clip(pos + rng.normal(0, 0.4, self.depth), 0, len(stack) - 1)
            arr = torch.from_numpy(
                np.asarray(stack[pos.astype(int)], dtype=np.float32) / 255.0)
            arr = F.interpolate(arr.unsqueeze(1), size=(self.size, self.size),
                                mode="bilinear", align_corners=False).squeeze(1)
            if self.train:
                arr = augment(arr, rng)
            vols[j] = arr
            plane[j] = PLANE_INDEX.get(plane_name, len(PLANE_INDEX))
            fluid[j], fatsat[j] = fl, fs
            mask[j] = True

        if not mask.any():
            mask[0] = True
        return vols, plane, fluid, fatsat, mask, torch.from_numpy(self.targets[i])


def augment(arr, rng):
    """Mild in-plane affine plus intensity, synchronised across the volume. No flips."""
    n = arr.shape[0]
    angle = float(rng.uniform(-6, 6)) * np.pi / 180
    scale = float(rng.uniform(0.95, 1.05))
    tx, ty = rng.uniform(-0.05, 0.05, 2)
    cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
    theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)
    grid = F.affine_grid(theta.unsqueeze(0).expand(n, -1, -1), (n, 1, *arr.shape[1:]),
                         align_corners=False)
    arr = F.grid_sample(arr.unsqueeze(1), grid, align_corners=False).squeeze(1)
    arr = arr.clamp(min=0).pow(float(rng.uniform(0.8, 1.25))) * float(rng.uniform(0.9, 1.1))
    if rng.random() < 0.3:
        arr = (arr + torch.randn_like(arr) * float(rng.uniform(0.01, 0.04))).clamp(0, 1)
    return arr


def forward(net, batch):
    vols, plane, fluid, fatsat, mask = [t.to(DEVICE, non_blocking=True) for t in batch[:5]]
    return net(vols, plane, fluid, fatsat, mask)


def predict(net, loader, n):
    net.eval()
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            out.append(torch.sigmoid(forward(net, batch)).float().cpu().numpy())
    return np.concatenate(out)[:n]


def run(config, studies, series_index, split, args):
    torch.manual_seed(0)
    net = VolumetricModel(config["encoder"], config["unfreeze"], dropout=config["dropout"]).to(DEVICE)
    enc = [p for p in net.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc}
    new = [p for p in net.parameters() if p.requires_grad and id(p) not in enc_ids]
    groups = [{"params": new, "lr": config["lr"]}]
    if enc:
        groups.append({"params": enc, "lr": config["lr"] * config["enc_lr_scale"]})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    common = dict(size=args.size, depth=args.depth, n_series=config["n_series"])
    dl = lambda rows, tr: DataLoader(
        StudyVolumes(studies, series_index, rows, train=tr, **common),
        batch_size=config["batch"], shuffle=tr, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)
    tr_dl, va_dl, go_dl = dl(split["train"], True), dl(split["val"], False), dl(split["gold"], False)
    y_val = protocol.soft_targets(studies, split["val"])
    y_gold = protocol.gold_targets(studies, split["gold"])

    best = {"bce": np.inf, "auc": float("nan"), "epoch": -1, "val": None, "gold": None}
    for epoch in range(args.epochs):
        net.train()
        for batch in tr_dl:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(forward(net, batch).float(), batch[5].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step()
        sched.step()
        p_val = predict(net, va_dl, len(y_val))
        p_gold = predict(net, go_dl, len(y_gold))
        bce = float(protocol.soft_bce(y_val, p_val).mean())
        auc = float(np.nanmean(protocol.gold_auc(y_gold, p_gold)))
        print(f"    epoch {epoch:2d}  bce {bce:.4f}  goldAUC {auc:.4f}", flush=True)
        if bce < best["bce"]:
            best = {"bce": bce, "auc": auc, "epoch": epoch,
                    "val": p_val, "gold": p_gold}
    return best


COMPARE = [
    {"name": "MedicalNet 3D-R18, frozen", "encoder": "medicalnet", "unfreeze": 0},
    {"name": "MedicalNet 3D-R18, unfreeze 1", "encoder": "medicalnet", "unfreeze": 1},
    {"name": "Kinetics R3D-18, frozen", "encoder": "r3d18", "unfreeze": 0},
    {"name": "Kinetics R3D-18, unfreeze 1", "encoder": "r3d18", "unfreeze": 1},
]
DEFAULTS = {"lr": 1e-3, "enc_lr_scale": 0.1, "dropout": 0.1, "n_series": 6, "batch": 8}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--size", type=int, default=112)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--subset", help="1-based indices into COMPARE, e.g. 2,3,4")
    p.add_argument("--save-preds", metavar="TAG",
                   help="write results/preds_<TAG>.npz for common/ensemble.py (single config only)")
    p.add_argument("--encoder", default="medicalnet", choices=["medicalnet", "r3d18"])
    p.add_argument("--unfreeze", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    args = p.parse_args()

    studies, _ = ds.all_studies()
    series_index = ds.pd.DataFrame(json.loads(
        (ds.CACHE / "slices224" / "index.json").read_text()))
    series_index = series_index[series_index.n_slices > 0]
    studies = studies[studies.StudyInstanceUID.isin(
        set(series_index["StudyInstanceUID"]))].reset_index(drop=True)
    split = protocol.make_split(studies)
    print(f"train {len(split['train'])}  val {len(split['val'])}  gold {len(split['gold'])}")
    print(f"baseline val soft-BCE: {protocol.baseline_bce(studies, split):.4f}", flush=True)

    chosen = ([COMPARE[int(i) - 1] for i in args.subset.split(",")]
              if args.subset else COMPARE)
    configs = ([{**DEFAULTS, **c} for c in chosen] if args.compare or args.subset else
               [{**DEFAULTS, "name": args.encoder, "encoder": args.encoder,
                 "unfreeze": args.unfreeze}])
    rows = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}", flush=True)
        t0 = time.perf_counter()
        best = run(config, studies, series_index, split, args)
        rows.append({"name": config["name"], "encoder": config["encoder"],
                     "unfreeze": config["unfreeze"], "val_soft_bce": best["bce"],
                     "gold_auc": best["auc"], "best_epoch": best["epoch"],
                     "minutes": round((time.perf_counter() - t0) / 60, 1)})
        print(f"  -> BCE {best['bce']:.4f}  goldAUC {best['auc']:.4f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.save_preds and len(configs) == 1:
        np.savez(out / f"preds_{args.save_preds}.npz", val=best["val"], gold=best["gold"])
    frame = ds.pd.DataFrame(rows).sort_values("val_soft_bce")
    # A single --save-preds run must not overwrite the full comparison's results.
    name = "compare.csv" if len(configs) > 1 else f"single_{configs[0]['name']}.csv"
    frame.to_csv(out / name, index=False)
    print("\n" + frame.to_string(index=False))


if __name__ == "__main__":
    main()
