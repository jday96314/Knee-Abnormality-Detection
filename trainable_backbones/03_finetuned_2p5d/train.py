#!/usr/bin/env python3
"""Train architecture 3: partially fine-tuned 2.5D encoder with hybrid pooling.

Unlike architectures 1 and 2 this reads pixels, so the cost model is different: a full
pass over every slice of every study is ~626k forward-backward passes per epoch, which is
hours. Instead each study contributes a fixed, stratified, re-drawn sample of slices per
epoch — `n_series` series x `n_slices` slices. That is both the compute budget and the
strongest augmentation available, since a different sample is drawn every epoch.

    python train.py --unfreeze 4 --mode plane --epochs 12
    python train.py --compare        # the small pre-specified config set
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
from model import PLANE_INDEX, Model2p5D  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = Path(__file__).resolve().parents[2] / "frozen_backbones" / "mri_foundation" / "pretrained_weights" / "MRI_CORE_vitb.pth"


class StudySlices(Dataset):
    """Stratified slice sample per study, redrawn every epoch when training.

    Series are chosen preferring distinct planes, because a study's redundant fourth
    sagittal series adds less than its only axial one — the same reasoning the plan gives
    for preferring to drop redundant series during augmentation.
    """

    def __init__(self, studies, series_index, rows, size=224, n_series=6, n_slices=4,
                 train=False, seed=0):
        self.uids = studies.iloc[rows]["StudyInstanceUID"].to_numpy()
        self.targets = protocol.soft_targets(studies, rows)
        self.size, self.n_series, self.n_slices, self.train = size, n_series, n_slices, train
        self.rng = np.random.default_rng(seed)
        self.by_study = {}
        for r in series_index.itertuples():
            self.by_study.setdefault(r.StudyInstanceUID, []).append(
                (r.SeriesInstanceUID, r.Anatomical_Plane, int(r.n_slices)))

    def __len__(self):
        return len(self.uids)

    def _pick_series(self, entries, rng):
        by_plane = {}
        for e in entries:
            by_plane.setdefault(e[1], []).append(e)
        picked, spare = [], []
        for plane in sorted(by_plane):                      # one of each plane first
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
        entries = [e for e in self.by_study.get(uid, []) if e[2] > 0]
        chosen = self._pick_series(entries, rng) if entries else []

        images = torch.zeros(self.n_series, self.n_slices, self.size, self.size)
        plane = torch.zeros(self.n_series, dtype=torch.long)
        mask = torch.zeros(self.n_series, dtype=torch.bool)

        for j, (series_uid, plane_name, n) in enumerate(chosen):
            stack = ds.load_series(uid, series_uid, self.size)
            if len(stack) == 0:
                continue
            # Stratified through the central portion, jittered when training.
            centres = np.linspace(0.15, 0.85, self.n_slices)
            if self.train:
                centres = np.clip(centres + rng.normal(0, 0.05, self.n_slices), 0.02, 0.98)
            idx = np.clip((centres * len(stack)).astype(int), 0, len(stack) - 1)
            arr = torch.from_numpy(np.asarray(stack[idx], dtype=np.float32) / 255.0)
            if self.train:
                arr = augment_series(arr, rng)
            images[j] = arr
            plane[j] = PLANE_INDEX.get(plane_name, len(PLANE_INDEX))
            mask[j] = True

        if not mask.any():
            mask[0] = True
        return images, plane, mask, torch.from_numpy(self.targets[i])


def augment_series(arr: torch.Tensor, rng) -> torch.Tensor:
    """Pixel augmentation, synchronised across the slices of one series.

    Geometry is deliberately mild and shared within a series: the slices are one
    acquisition of one knee, so rotating them independently would fabricate anatomy that
    cannot occur. Flips are excluded entirely — the VLM experiments measured flip TTA
    costing 0.061 AUC on ACL, because a left-right flip on sagittal images exchanges
    anterior for posterior.
    """
    n = arr.shape[0]
    angle = float(rng.uniform(-6, 6)) * np.pi / 180
    scale = float(rng.uniform(0.95, 1.05))
    tx, ty = rng.uniform(-0.05, 0.05, 2)
    cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
    theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)
    grid = F.affine_grid(theta.unsqueeze(0).expand(n, -1, -1), (n, 1, *arr.shape[1:]),
                         align_corners=False)
    arr = F.grid_sample(arr.unsqueeze(1), grid, align_corners=False, padding_mode="zeros").squeeze(1)

    arr = arr.clamp(min=0).pow(float(rng.uniform(0.8, 1.25)))          # gamma
    arr = arr * float(rng.uniform(0.9, 1.1))                            # contrast
    if rng.random() < 0.3:                                              # mild Rician-like noise
        arr = (arr + torch.randn_like(arr) * float(rng.uniform(0.01, 0.04))).clamp(0, 1)
    return arr


def evaluate(net, loader, targets):
    net.eval()
    preds = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for images, plane, mask, _ in loader:
            preds.append(torch.sigmoid(forward(net, images, plane, mask)).float().cpu().numpy())
    return np.concatenate(preds)[: len(targets)]


def forward(net, images, plane, mask):
    """images [B, S, K, H, W] -> logits [B, 12]."""
    B, S, K, H, W = images.shape
    flat = images.reshape(B * S * K, 1, H, W).to(DEVICE, non_blocking=True)
    emb = net.encoder(flat).reshape(B * S, K, -1)
    emb = net.context(emb)
    slice_mask = torch.ones(B * S, K, dtype=torch.bool, device=DEVICE)
    series = net.pool(emb, slice_mask).reshape(B, S, -1)
    return net.head(series, plane.to(DEVICE), mask.to(DEVICE))


def run(config, studies, series_index, split, args):
    torch.manual_seed(0)
    net = Model2p5D(CHECKPOINT, unfreeze=config["unfreeze"], mode=config["mode"],
                    d_model=config["d_model"], dropout=config["dropout"],
                    kernel=config["kernel"], device=DEVICE).to(DEVICE)
    trainable = [p for p in net.parameters() if p.requires_grad]
    # The pretrained encoder gets a much smaller step than the newly initialised head;
    # one shared learning rate would either wreck the features or starve the head.
    enc_params = [p for p in net.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc_params}
    new_params = [p for p in trainable if id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": config["lr"] * config["enc_lr_scale"]},
         {"params": new_params, "lr": config["lr"]}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()   # bf16 autocast needs no gradient scaler

    common = dict(size=args.size, n_series=config["n_series"], n_slices=config["n_slices"])
    tr = StudySlices(studies, series_index, split["train"], train=True, **common)
    va = StudySlices(studies, series_index, split["val"], **common)
    go = StudySlices(studies, series_index, split["gold"], **common)
    dl = lambda d, sh: DataLoader(d, batch_size=config["batch"], shuffle=sh,
                                  num_workers=args.workers, pin_memory=True, drop_last=False,
                                  persistent_workers=args.workers > 0)
    tr_dl, va_dl, go_dl = dl(tr, True), dl(va, False), dl(go, False)
    y_val = protocol.soft_targets(studies, split["val"])
    y_gold = protocol.gold_targets(studies, split["gold"])

    best = {"bce": np.inf, "auc": float("nan"), "epoch": -1, "val": None, "gold": None}
    for epoch in range(args.epochs):
        net.train()
        for images, plane, mask, target in tr_dl:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(forward(net, images, plane, mask).float(), target.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        sched.step()

        p_val, p_gold = evaluate(net, va_dl, y_val), evaluate(net, go_dl, y_gold)
        bce = float(protocol.soft_bce(y_val, p_val).mean())
        auc = float(np.nanmean(protocol.gold_auc(y_gold, p_gold)))
        print(f"    epoch {epoch:2d}  bce {bce:.4f}  goldAUC {auc:.4f}", flush=True)
        if bce < best["bce"]:
            best = {"bce": bce, "auc": auc, "epoch": epoch,
                    "val": p_val, "gold": p_gold}
    return best


COMPARE = [
    # The pre-specified set. `mode` is the contested axis: the plan says to keep
    # architecture 2's pathology queries, but those measurably lost here, so both run.
    {"name": "plane-head, unfreeze 4", "mode": "plane", "unfreeze": 4},
    {"name": "query-head, unfreeze 4", "mode": "query", "unfreeze": 4},
    {"name": "plane-head, frozen encoder", "mode": "plane", "unfreeze": 0},
    {"name": "plane-head, unfreeze 8", "mode": "plane", "unfreeze": 8},
]
DEFAULTS = {"lr": 3e-4, "enc_lr_scale": 0.1, "d_model": 256, "dropout": 0.1,
            "kernel": 3, "n_series": 6, "n_slices": 4, "batch": 8}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--subset", help="1-based indices into COMPARE, e.g. 2,3,4")
    p.add_argument("--save-preds", metavar="TAG",
                   help="write results/preds_<TAG>.npz for common/ensemble.py (single config only)")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    p.add_argument("--mode", default="plane", choices=["plane", "query"])
    p.add_argument("--unfreeze", type=int, default=4)
    args = p.parse_args()

    studies, _ = ds.all_studies()
    # n_slices comes from the cache index rather than being re-derived by opening 24,371
    # files, and the study set is filtered the same way as architectures 1-2 so that
    # `make_split` reproduces exactly the same train/val/gold rows.
    series_index = ds.pd.DataFrame(json.loads(
        (ds.CACHE / f"slices{args.size}" / "index.json").read_text()))
    series_index = series_index[series_index.n_slices > 0]
    keep = set(series_index["StudyInstanceUID"])
    studies = studies[studies.StudyInstanceUID.isin(keep)].reset_index(drop=True)
    split = protocol.make_split(studies)
    print(f"train {len(split['train'])}  val {len(split['val'])}  gold {len(split['gold'])}")
    print(f"baseline val soft-BCE: {protocol.baseline_bce(studies, split):.4f}", flush=True)

    chosen = ([COMPARE[int(i) - 1] for i in args.subset.split(",")]
              if args.subset else COMPARE)
    configs = ([{**DEFAULTS, **c} for c in chosen] if args.compare or args.subset
               else [{**DEFAULTS, "name": "single", "mode": args.mode, "unfreeze": args.unfreeze}])
    rows = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}", flush=True)
        t0 = time.perf_counter()
        best = run(config, studies, series_index, split, args)
        rows.append({"name": config["name"], "mode": config["mode"],
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
