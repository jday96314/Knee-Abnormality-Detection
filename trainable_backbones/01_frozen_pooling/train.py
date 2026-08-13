#!/usr/bin/env python3
"""Train and sweep architecture 1: frozen features, fixed hierarchical pooling.

Objective is soft BCE on a held-out 20% of the pseudo-labelled studies. Gold ROC AUC over
the 58 labelled studies is computed for reporting and never used to select anything.

Because the encoder is frozen and the pooling is fixed, every augmentation here acts on
cached features rather than pixels — which is the whole point of caching. The plan ranks
these structural augmentations above pixel transforms anyway: dropping a series or
jittering which slices are kept mimics real protocol variation, while adding noise to an
embedding does not correspond to anything physical.

    python train.py --sweep
    python train.py --hidden 256 --slice-dropout 0.1 --series-dropout 0.15
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import dataset  # noqa: E402
import features as featlib  # noqa: E402
import protocol  # noqa: E402
from model import Head, StudyDescriptor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StudyBank:
    """Per-study slice features, grouped by series, held in RAM as float32 tensors.

    Slice-level access is required because the augmentations resample slices, so the
    descriptor cannot be precomputed. Roughly 780k slices x 768 dims x 4 bytes = 2.4 GB,
    which fits comfortably and avoids re-reading the memmap every epoch.
    """

    def __init__(self, array, index, studies, rows):
        wanted = set(studies.iloc[rows]["StudyInstanceUID"])
        keep = index["study"].isin(wanted).to_numpy()
        idx = np.flatnonzero(keep)
        feats = torch.from_numpy(np.asarray(array[idx], dtype=np.float32))
        sub = index.iloc[idx].reset_index(drop=True)

        self.by_study: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for (study, series), group in sub.groupby(["study", "series"], sort=False):
            plane = group["plane"].iloc[0]
            self.by_study.setdefault(study, []).append((plane, feats[group.index.to_numpy()]))
        self.dim = feats.shape[1]


def augment(series_feats, rng, slice_keep, slice_dropout, series_dropout, noise):
    """Structural augmentation, applied per study, per epoch.

    A series is only dropped if at least one remains — losing the entire study is not a
    variation the model should be asked to handle. Slice dropout is contiguous rather
    than scattered, because real acquisitions lose blocks of coverage, not random slices.
    """
    out = []
    for plane, feats in series_feats:
        n = feats.shape[0]
        if n == 0:
            continue
        if slice_keep < 1.0:  # jitter which part of the stack is seen
            k = max(3, int(round(n * slice_keep)))
            start = rng.integers(0, max(1, n - k + 1))
            feats = feats[start:start + k]
            n = feats.shape[0]
        if slice_dropout > 0 and n > 6:
            drop = int(round(n * slice_dropout))
            if drop:
                start = rng.integers(0, n - drop)
                feats = torch.cat([feats[:start], feats[start + drop:]])
        out.append((plane, feats))

    if series_dropout > 0 and len(out) > 1:
        keep = [s for s in out if rng.random() > series_dropout]
        out = keep if keep else [out[rng.integers(0, len(out))]]

    if noise > 0:
        out = [(p, f + torch.randn_like(f) * noise) for p, f in out]
    return out


def build_batch(bank, descriptor, uids, rng=None, aug=None):
    vecs = []
    for uid in uids:
        series_feats = bank.by_study.get(uid, [])
        if aug is not None:
            series_feats = augment(series_feats, rng, **aug)
        vecs.append(descriptor(series_feats))
    return torch.stack(vecs)


def run_one(config, bank_tr, bank_ev, descriptor, studies, split, seed=0, epochs=60):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = Head(descriptor.out_dim, hidden=config["hidden"], dropout=config["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    train_uids = studies.iloc[split["train"]]["StudyInstanceUID"].to_numpy()
    y_train = torch.from_numpy(protocol.soft_targets(studies, split["train"])).to(DEVICE)
    aug = {
        "slice_keep": config["slice_keep"], "slice_dropout": config["slice_dropout"],
        "series_dropout": config["series_dropout"], "noise": config["noise"],
    }

    # Validation and gold descriptors are deterministic: no augmentation at evaluation.
    val_uids = studies.iloc[split["val"]]["StudyInstanceUID"].to_numpy()
    gold_uids = studies.iloc[split["gold"]]["StudyInstanceUID"].to_numpy()
    X_val = build_batch(bank_ev, descriptor, val_uids).to(DEVICE)
    X_gold = build_batch(bank_ev, descriptor, gold_uids).to(DEVICE)
    y_val = protocol.soft_targets(studies, split["val"])
    y_gold = protocol.gold_targets(studies, split["gold"])

    best = {"bce": np.inf}
    batch = config["batch"]
    for epoch in range(epochs):
        head.train()
        order = rng.permutation(len(train_uids))
        for start in range(0, len(order), batch):
            sel = order[start:start + batch]
            X = build_batch(bank_tr, descriptor, train_uids[sel], rng, aug).to(DEVICE)
            target = y_train[sel]
            if config["mixup"] > 0 and len(sel) > 1:
                lam = float(rng.beta(config["mixup"], config["mixup"]))
                perm = torch.randperm(len(sel), device=DEVICE)
                X = lam * X + (1 - lam) * X[perm]
                target = lam * target + (1 - lam) * target[perm]
            opt.zero_grad()
            loss_fn(head(X), target).backward()
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(head(X_val)).cpu().numpy()
            p_gold = torch.sigmoid(head(X_gold)).cpu().numpy()
        bce = float(protocol.soft_bce(y_val, p_val).mean())
        if bce < best["bce"]:
            best = {
                "bce": bce,
                "auc": float(np.nanmean(protocol.gold_auc(y_gold, p_gold))),
                "epoch": epoch,
                "p_val": p_val, "p_gold": p_gold,
            }
    return best


SWEEP = {
    "hidden": [0, 256],
    "dropout": [0.1, 0.3],
    "slice_keep": [1.0, 0.7],
    "series_dropout": [0.0, 0.15],
    "mixup": [0.0, 0.4],
}
FIXED = {"lr": 1e-3, "wd": 1e-2, "batch": 64, "slice_dropout": 0.1, "noise": 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="mri_core")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--head", default="cls", help="which read-out head of the backbone")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    for k, v in FIXED.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    for k, v in SWEEP.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v[0]), default=v[0])
    args = p.parse_args()

    studies, _ = dataset.all_studies()
    array, index, meta = featlib.load(args.backbone, args.size)
    head_names = meta["heads"]
    dim = meta["dim"] // len(head_names)
    which = head_names.index(args.head)
    array = array[:, which * dim:(which + 1) * dim]

    studies = studies[studies.StudyInstanceUID.isin(set(index["study"]))].reset_index(drop=True)
    split = protocol.make_split(studies)
    print(f"train {len(split['train'])}  val {len(split['val'])}  gold {len(split['gold'])}")
    print(f"baseline val soft-BCE: {protocol.baseline_bce(studies, split):.4f}")

    bank_tr = StudyBank(array, index, studies, split["train"])
    bank_ev = StudyBank(array, index, studies, np.concatenate([split["val"], split["gold"]]))
    descriptor = StudyDescriptor(dim)

    configs = []
    if args.sweep:
        keys = list(SWEEP)
        for combo in itertools.product(*[SWEEP[k] for k in keys]):
            configs.append({**FIXED, **dict(zip(keys, combo))})
    else:
        configs.append({**FIXED, **{k: getattr(args, k) for k in SWEEP}})

    rows = []
    for i, config in enumerate(configs, 1):
        started = time.perf_counter()
        runs = [run_one(config, bank_tr, bank_ev, descriptor, studies, split, s, args.epochs)
                for s in range(args.seeds)]
        row = {
            **{k: config[k] for k in SWEEP},
            "val_soft_bce": float(np.mean([r["bce"] for r in runs])),
            "gold_auc": float(np.mean([r["auc"] for r in runs])),
            "best_epoch": int(np.mean([r["epoch"] for r in runs])),
            "seconds": round(time.perf_counter() - started, 1),
        }
        rows.append(row)
        print(f"[{i}/{len(configs)}] BCE {row['val_soft_bce']:.4f}  goldAUC {row['gold_auc']:.4f}  "
              + "  ".join(f"{k}={config[k]}" for k in SWEEP), flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frame = dataset.pd.DataFrame(rows).sort_values("val_soft_bce")
    frame.to_csv(out / f"sweep_{args.backbone}_{args.head}.csv", index=False)
    print("\nbest by val soft BCE:")
    print(frame.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
