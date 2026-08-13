#!/usr/bin/env python3
"""Train and sweep architecture 2: learned pathology attention over series.

Identical protocol, split, features and augmentations to architecture 1 — the only thing
that changes is how series are combined into a study prediction. That is deliberate: the
question this directory answers is whether learned per-finding series selection beats
fixed pooling, and any other difference would confound it.

    python train.py --backbone mri_core --head cls --sweep
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
from augment import augment as feature_augment  # noqa: E402  (shared, in common/)
from model import PLANE_INDEX, SeriesAttentionModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SERIES = 16  # the cohort's maximum is 14; padding beyond that is never needed


def pool(feats: torch.Tensor) -> torch.Tensor:
    """The same fixed [mean, p90, max] slice pooling architecture 1 uses."""
    return torch.cat([
        feats.mean(0),
        torch.quantile(feats.float(), 0.9, dim=0).to(feats.dtype),
        feats.max(0).values,
    ])


class SeriesBank:
    """Per-study slice features plus the protocol metadata each series carries."""

    def __init__(self, array, index, studies, rows):
        wanted = set(studies.iloc[rows]["StudyInstanceUID"])
        idx = np.flatnonzero(index["study"].isin(wanted).to_numpy())
        feats = torch.from_numpy(np.asarray(array[idx], dtype=np.float32))
        sub = index.iloc[idx].reset_index(drop=True)
        self.by_study: dict[str, list] = {}
        for (study, series), group in sub.groupby(["study", "series"], sort=False):
            row = group.iloc[0]
            self.by_study.setdefault(study, []).append((
                row["plane"], feats[group.index.to_numpy()],
                int(row["fluid"]), int(row["fatsat"]),
            ))
        self.dim = feats.shape[1]


def build_batch(bank, uids, feat_dim, rng=None, aug=None):
    """Pad each study to MAX_SERIES tokens and return the mask that hides the padding."""
    n = len(uids)
    pooled = torch.zeros(n, MAX_SERIES, 3 * feat_dim)
    plane = torch.zeros(n, MAX_SERIES, dtype=torch.long)
    fluid = torch.zeros(n, MAX_SERIES, dtype=torch.long)
    fatsat = torch.zeros(n, MAX_SERIES, dtype=torch.long)
    mask = torch.zeros(n, MAX_SERIES, dtype=torch.bool)

    for i, uid in enumerate(uids):
        entries = bank.by_study.get(uid, [])
        if aug is not None:
            # Reuse architecture 1's augmentation on the (plane, feats) pairs, then put
            # the metadata back, so both architectures see identical perturbations.
            meta = {id(f): (fl, fs) for _, f, fl, fs in entries}
            perturbed = feature_augment([(p, f) for p, f, _, _ in entries], rng, **aug)
            entries = [(p, f, *meta.get(id(f), (0, 0))) for p, f in perturbed]
        for j, (p, f, fl, fs) in enumerate(entries[:MAX_SERIES]):
            if f.shape[0] == 0:
                continue
            pooled[i, j] = pool(f)
            plane[i, j] = PLANE_INDEX.get(p, len(PLANE_INDEX))
            fluid[i, j] = fl
            fatsat[i, j] = fs
            mask[i, j] = True
        if not mask[i].any():          # a study must contribute at least one token
            mask[i, 0] = True
    return pooled, plane, fluid, fatsat, mask


def run_one(config, bank_tr, bank_ev, studies, split, feat_dim, seed=0, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SeriesAttentionModel(
        feat_dim, d_model=config["d_model"], n_heads=config["n_heads"],
        dropout=config["dropout"], n_layers=config["n_layers"],
    ).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    train_uids = studies.iloc[split["train"]]["StudyInstanceUID"].to_numpy()
    y_train = torch.from_numpy(protocol.soft_targets(studies, split["train"])).to(DEVICE)
    aug = {k: config[k] for k in ("slice_keep", "slice_dropout", "series_dropout", "noise")}

    val_batch = [t.to(DEVICE) for t in build_batch(
        bank_ev, studies.iloc[split["val"]]["StudyInstanceUID"].to_numpy(), feat_dim)]
    gold_batch = [t.to(DEVICE) for t in build_batch(
        bank_ev, studies.iloc[split["gold"]]["StudyInstanceUID"].to_numpy(), feat_dim)]
    y_val = protocol.soft_targets(studies, split["val"])
    y_gold = protocol.gold_targets(studies, split["gold"])

    best = {"bce": np.inf, "auc": np.nan, "epoch": -1}
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(train_uids))
        for start in range(0, len(order), config["batch"]):
            sel = order[start:start + config["batch"]]
            batch = [t.to(DEVICE) for t in build_batch(bank_tr, train_uids[sel], feat_dim, rng, aug)]
            target = y_train[sel]
            opt.zero_grad()
            loss_fn(net(*batch), target).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(net(*val_batch)).cpu().numpy()
            p_gold = torch.sigmoid(net(*gold_batch)).cpu().numpy()
        bce = float(protocol.soft_bce(y_val, p_val).mean())
        if bce < best["bce"]:
            best = {"bce": bce, "auc": float(np.nanmean(protocol.gold_auc(y_gold, p_gold))),
                    "epoch": epoch}
    return best


SWEEP = {
    "d_model": [128, 256],
    "n_layers": [1, 2],
    "dropout": [0.1, 0.3],
    "series_dropout": [0.0, 0.15],
}
FIXED = {"lr": 1e-3, "wd": 1e-2, "batch": 64, "n_heads": 4,
         "slice_keep": 1.0, "slice_dropout": 0.1, "noise": 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="mri_core")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--head", default="cls")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    for k, v in {**FIXED, **{k: v[0] for k, v in SWEEP.items()}}.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    args = p.parse_args()

    studies, _ = dataset.all_studies()
    array, index, meta = featlib.load(args.backbone, args.size)
    dim = meta["dim"] // len(meta["heads"])
    which = meta["heads"].index(args.head)
    array = array[:, which * dim:(which + 1) * dim]

    studies = studies[studies.StudyInstanceUID.isin(set(index["study"]))].reset_index(drop=True)
    split = protocol.make_split(studies)
    print(f"train {len(split['train'])}  val {len(split['val'])}  gold {len(split['gold'])}")
    print(f"baseline val soft-BCE: {protocol.baseline_bce(studies, split):.4f}", flush=True)

    bank_tr = SeriesBank(array, index, studies, split["train"])
    bank_ev = SeriesBank(array, index, studies, np.concatenate([split["val"], split["gold"]]))

    configs = []
    if args.sweep:
        keys = list(SWEEP)
        for combo in itertools.product(*[SWEEP[k] for k in keys]):
            configs.append({**FIXED, **dict(zip(keys, combo))})
    else:
        configs.append({**FIXED, **{k: getattr(args, k) for k in SWEEP}})

    rows = []
    for i, config in enumerate(configs, 1):
        t0 = time.perf_counter()
        best = run_one(config, bank_tr, bank_ev, studies, split, dim, 0, args.epochs)
        rows.append({**{k: config[k] for k in SWEEP}, "val_soft_bce": best["bce"],
                     "gold_auc": best["auc"], "best_epoch": best["epoch"],
                     "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[{i}/{len(configs)}] BCE {best['bce']:.4f}  goldAUC {best['auc']:.4f}  "
              + "  ".join(f"{k}={config[k]}" for k in SWEEP), flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frame = dataset.pd.DataFrame(rows).sort_values("val_soft_bce")
    frame.to_csv(out / f"sweep_{args.backbone}_{args.head}.csv", index=False)
    print("\nbest by val soft BCE:")
    print(frame.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
