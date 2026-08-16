#!/usr/bin/env python3
"""One Optuna study per (backbone, head) pair for architecture 3.

    python tune.py --backbone mri_core --head plane --trials 20
    python tune.py --list                      # what has been run so far

Each study searches optimisation and augmentation jointly, because they interact: a heavier
augmentation policy wants a longer schedule and tolerates a larger learning rate, so tuning
them separately would understate both. `unfreeze` is in the search because the earlier
four-configuration comparison found it the single largest effect and never reached its top.

Studies persist in a SQLite file on local disk (`KNEE_OPTUNA_DIR`, default
`/mnt/data01/knee_optuna`), so a run can be interrupted and resumed, and
several pairs can share one storage file without clobbering each other.

Pruning is `MedianPruner` on epoch-level validation BCE. That matters here: a trial costs
5-30 minutes, and roughly half of them are visibly hopeless within three epochs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import protocol  # noqa: E402
from encoders import N_UNITS  # noqa: E402
from model import Model2p5D  # noqa: E402
from slices import AugmentConfig, StudySlices, load_cohort  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
# The repository lives on a CIFS share, where SQLite's file locking does not work and every
# write raises "database is locked". The study database therefore lives on local disk; only
# the human-readable summaries are written back into the repository.
DB_DIR = Path(os.environ.get("KNEE_OPTUNA_DIR", "/mnt/data01/knee_optuna"))
STORAGE = f"sqlite:///{DB_DIR / 'optuna.db'}"
HEADS = ["plane", "query", "slot"]


def forward(net, images, plane, mask):
    """images [B, S, K, H, W] -> logits [B, 12]."""
    B, S, K, H, W = images.shape
    emb = net.encoder(images.reshape(B * S * K, 1, H, W).to(DEVICE, non_blocking=True))
    emb = net.context(emb.reshape(B * S, K, -1))
    slice_mask = torch.ones(B * S, K, dtype=torch.bool, device=DEVICE)
    series = net.pool(emb, slice_mask).reshape(B, S, -1)
    return net.head(series, plane.to(DEVICE), mask.to(DEVICE))


def predict(net, loader, n):
    net.eval()
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for images, plane, mask, _ in loader:
            out.append(torch.sigmoid(forward(net, images, plane, mask)).float().cpu().numpy())
    return np.concatenate(out)[:n]


def build_scheduler(kind, opt, epochs, steps_per_epoch, lrs):
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lrs, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=0.25), True                      # stepped per batch
    if kind == "cosine_warm":
        warm = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=2)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 2))
        return torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[2]), False
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.3), False
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs), False


def train_once(params, cohort, split, args, trial=None):
    """Train one configuration; return (best val soft BCE, gold AUC at that epoch)."""
    studies, series_index = cohort
    torch.manual_seed(args.seed)
    net = Model2p5D(backbone=args.backbone, unfreeze=params["unfreeze"], mode=args.head,
                    d_model=params["d_model"], dropout=params["dropout"],
                    kernel=params["kernel"], n_slots=params["n_series"],
                    device=DEVICE).to(DEVICE)

    enc = [p for p in net.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc}
    new = [p for p in net.parameters() if p.requires_grad and id(p) not in enc_ids]
    groups, lrs = [{"params": new, "lr": params["lr"]}], [params["lr"]]
    if enc:
        groups.append({"params": enc, "lr": params["lr"] * params["enc_lr_scale"]})
        lrs.append(params["lr"] * params["enc_lr_scale"])
    opt = torch.optim.AdamW(groups, weight_decay=params["wd"])

    aug = AugmentConfig(**{k: params[k] for k in
                           ("geom", "intensity", "noise", "bias", "blur", "erase",
                            "phase", "series_dropout")})
    common = dict(size=args.size, n_series=params["n_series"], n_slices=params["n_slices"])
    def loader(rows, train):
        return DataLoader(
            StudySlices(studies, series_index, rows, train=train, seed=args.seed,
                        aug=aug if train else None, **common),
            batch_size=params["batch"], shuffle=train, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0)

    tr, va, go = loader(split["train"], True), loader(split["val"], False), loader(split["gold"], False)
    y_val = protocol.soft_targets(studies, split["val"])
    y_gold = protocol.gold_targets(studies, split["gold"])
    sched, per_batch = build_scheduler(params["schedule"], opt, params["epochs"], len(tr), lrs)
    loss_fn = nn.BCEWithLogitsLoss()
    trainable = [p for p in net.parameters() if p.requires_grad]

    best_bce, best_auc = np.inf, float("nan")
    for epoch in range(params["epochs"]):
        net.train()
        for images, plane, mask, target in tr:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(forward(net, images, plane, mask).float(), target.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if per_batch:
                sched.step()
        if not per_batch:
            sched.step()

        bce = float(protocol.soft_bce(y_val, predict(net, va, len(y_val))).mean())
        if bce < best_bce:
            best_bce = bce
            best_auc = float(np.nanmean(protocol.gold_auc(
                y_gold, predict(net, go, len(y_gold)))))
        if trial is not None:
            trial.report(bce, epoch)
            if trial.should_prune():
                trial.set_user_attr("gold_auc", best_auc)
                trial.set_user_attr("pruned_at", epoch)
                raise optuna.TrialPruned()
    if trial is not None:
        trial.set_user_attr("gold_auc", best_auc)
    return best_bce, best_auc


# Measured forward+backward throughput at 224 px on one 4090, mid-unfreeze (slices/s):
# dinov3s 2762, dinov2s 2660, radimagenet 2279, mri_core 1278, ortho 272. The epoch range
# is narrowed for the expensive encoders so a trial costs roughly the same wall-clock
# everywhere — otherwise one OrthoFoundation trial would consume a whole small-backbone study.
EPOCH_RANGE = {"ortho": (6, 12), "mri_core": (8, 16)}
# batch x n_series x n_slices slices are in flight at once, so batch is the main
# activation-memory driver. ViT-L at 24 unfrozen blocks will not fit 16 studies.
BATCH_CHOICES = {"ortho": [2, 4], "mri_core": [4, 8]}


def suggest(trial, args):
    """The search space. Identical across backbones except the cost-adjusted epoch range."""
    n_units = N_UNITS[args.backbone]
    lo, hi = EPOCH_RANGE.get(args.backbone, (8, 16))
    return {
        # How much of the encoder adapts. The earlier comparison stopped at 8/12 while
        # still improving, so the top of this range is deliberately "everything".
        "unfreeze": trial.suggest_int("unfreeze", 0, n_units),
        "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        "enc_lr_scale": trial.suggest_float("enc_lr_scale", 0.01, 1.0, log=True),
        "wd": trial.suggest_float("wd", 1e-4, 1e-1, log=True),
        "batch": trial.suggest_categorical("batch", BATCH_CHOICES.get(args.backbone, [4, 8, 16])),
        "epochs": trial.suggest_int("epochs", lo, hi),
        "schedule": trial.suggest_categorical(
            "schedule", ["cosine", "onecycle", "cosine_warm", "step"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        # kernel=1 makes SliceContext an identity-plus-projection, so this doubles as the
        # ablation of the 2.5D module that the original four configurations never ran.
        "kernel": trial.suggest_categorical("kernel", [1, 3, 5]),
        # Fixed, not searched: these set the cost of a trial (slices per study per epoch),
        # and letting the sampler roam over them spends the budget on speed rather than on
        # the axes actually under study. 6x4 is also the original run's budget, which keeps
        # these numbers comparable to the four-configuration comparison.
        "d_model": 256, "n_series": 6, "n_slices": 4,
        # Augmentation: six pixel dials plus two structural ones, all 0 = disabled.
        "geom": trial.suggest_float("geom", 0.0, 1.0),
        "intensity": trial.suggest_float("intensity", 0.0, 1.0),
        "noise": trial.suggest_float("noise", 0.0, 1.0),
        "bias": trial.suggest_float("bias", 0.0, 1.0),
        "blur": trial.suggest_float("blur", 0.0, 1.0),
        "erase": trial.suggest_float("erase", 0.0, 1.0),
        "phase": trial.suggest_float("phase", 0.0, 1.0),
        "series_dropout": trial.suggest_float("series_dropout", 0.0, 0.3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="mri_core",
                   choices=["mri_core", "ortho", "radimagenet", "dinov2s", "dinov3s"])
    p.add_argument("--head", default="plane", choices=HEADS)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--timeout-hours", type=float, default=None)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if args.list:
        for name in optuna.study.get_all_study_names(storage=STORAGE):
            s = optuna.load_study(study_name=name, storage=STORAGE)
            done = [t for t in s.trials if t.value is not None]
            if not done:
                print(f"{name:28s} {len(s.trials)} trials, none complete")
                continue
            b = s.best_trial
            print(f"{name:28s} {len(done):3d}/{len(s.trials)} complete  "
                  f"best BCE {b.value:.4f}  goldAUC {b.user_attrs.get('gold_auc', float('nan')):.4f}")
        return

    cohort = load_cohort(args.size)
    split = protocol.make_split(cohort[0])
    name = f"{args.backbone}__{args.head}"
    study = optuna.create_study(
        study_name=name, storage=STORAGE, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2))
    print(f"study {name}: {len(study.trials)} existing trials", flush=True)

    def objective(trial):
        params = suggest(trial, args)
        t0 = time.perf_counter()
        try:
            bce, auc = train_once(params, cohort, split, args, trial)
        finally:
            # Runs on OOM and on pruning too, so a failed trial cannot strand its model on
            # the GPU and cascade into every trial after it.
            gc.collect()
            torch.cuda.empty_cache()
        print(f"  trial {trial.number}: BCE {bce:.4f}  goldAUC {auc:.4f}  "
              f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
        return bce

    study.optimize(objective, n_trials=args.trials,
                   timeout=args.timeout_hours * 3600 if args.timeout_hours else None,
                   catch=(torch.cuda.OutOfMemoryError,))

    done = [t for t in study.trials if t.value is not None]
    if done:
        b = study.best_trial
        print(f"\nbest of {len(done)}: BCE {b.value:.4f}  "
              f"goldAUC {b.user_attrs.get('gold_auc', float('nan')):.4f}")
        print(json.dumps(b.params, indent=2, sort_keys=True))
        (RESULTS / f"best_{name}.json").write_text(json.dumps(
            {"backbone": args.backbone, "head": args.head, "val_soft_bce": b.value,
             "gold_auc": b.user_attrs.get("gold_auc"), "params": b.params}, indent=2))


if __name__ == "__main__":
    main()
