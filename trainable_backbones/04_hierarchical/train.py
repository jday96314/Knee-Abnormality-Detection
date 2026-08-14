#!/usr/bin/env python3
"""Train and sweep architecture 4: hierarchical slice + series Transformer.

Same features, same split, same augmentations as architectures 1 and 2, so the three are
directly comparable and the only thing varying is how slices and series are aggregated.

Adds the plan's consistency regularisation: two independently augmented views of each study
are pushed towards the same logits. That is the one training-side idea unique to this
architecture, and it is cheap to test as an on/off axis.

    python train.py --backbone mri_core --sweep
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
from augment import augment as feature_augment  # noqa: E402
from model import PLANE_INDEX, HierarchicalModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SERIES = 16   # cohort maximum is 14
MAX_SLICES = 48   # median series is 30 slices; the tail runs to a few hundred


class SeriesBank:
    """Per-study slice features, kept unpooled — this architecture needs the slice axis."""

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
                int(row["fluid"]), int(row["fatsat"])))
        self.dim = feats.shape[1]


def subsample(n: int, k: int, rng) -> np.ndarray:
    """Stratified `k` indices out of `n`, jittered when training."""
    if n <= k:
        return np.arange(n)
    edges = np.linspace(0, n, k + 1)
    if rng is None:
        return ((edges[:-1] + edges[1:]) / 2).astype(int).clip(0, n - 1)
    return np.array([rng.integers(int(a), max(int(a) + 1, int(b))) for a, b in
                     zip(edges[:-1], edges[1:])]).clip(0, n - 1)


def build_batch(bank, uids, dim, rng=None, aug=None):
    n = len(uids)
    feats = torch.zeros(n, MAX_SERIES, MAX_SLICES, dim)
    slice_mask = torch.zeros(n, MAX_SERIES, MAX_SLICES, dtype=torch.bool)
    plane = torch.zeros(n, MAX_SERIES, dtype=torch.long)
    fluid = torch.zeros(n, MAX_SERIES, dtype=torch.long)
    fatsat = torch.zeros(n, MAX_SERIES, dtype=torch.long)
    series_mask = torch.zeros(n, MAX_SERIES, dtype=torch.bool)

    for i, uid in enumerate(uids):
        entries = bank.by_study.get(uid, [])
        if aug is not None:
            meta = {id(f): (fl, fs) for _, f, fl, fs in entries}
            perturbed = feature_augment([(p, f) for p, f, _, _ in entries], rng, **aug)
            entries = [(p, f, *meta.get(id(f), (0, 0))) for p, f in perturbed]
        for j, (p, f, fl, fs) in enumerate(entries[:MAX_SERIES]):
            if f.shape[0] == 0:
                continue
            sel = subsample(f.shape[0], MAX_SLICES, rng)
            feats[i, j, :len(sel)] = f[sel]
            slice_mask[i, j, :len(sel)] = True
            plane[i, j] = PLANE_INDEX.get(p, len(PLANE_INDEX))
            fluid[i, j], fatsat[i, j] = fl, fs
            series_mask[i, j] = True
        if not series_mask[i].any():
            series_mask[i, 0] = True
            slice_mask[i, 0, 0] = True
    return feats, slice_mask, plane, fluid, fatsat, series_mask


def run_one(config, bank_tr, bank_ev, studies, split, dim, seed=0, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = HierarchicalModel(dim, d_model=config["d_model"], slice_layers=config["slice_layers"],
                            series_layers=config["series_layers"], dropout=config["dropout"],
                            mode=config["mode"]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    train_uids = studies.iloc[split["train"]]["StudyInstanceUID"].to_numpy()
    y_train = torch.from_numpy(protocol.soft_targets(studies, split["train"])).to(DEVICE)
    aug = {k: config[k] for k in ("slice_keep", "slice_dropout", "series_dropout", "noise")}

    def eval_batches(rows):
        uids = studies.iloc[rows]["StudyInstanceUID"].to_numpy()
        return [[t.to(DEVICE) for t in build_batch(bank_ev, uids[s:s + 64], dim)]
                for s in range(0, len(uids), 64)]

    val_batches, gold_batches = eval_batches(split["val"]), eval_batches(split["gold"])
    y_val = protocol.soft_targets(studies, split["val"])
    y_gold = protocol.gold_targets(studies, split["gold"])

    def predict(batches):
        net.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(net(*b)).cpu().numpy() for b in batches])

    best = {"bce": np.inf, "auc": np.nan, "epoch": -1, "val": None, "gold": None}
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(train_uids))
        for start in range(0, len(order), config["batch"]):
            sel = order[start:start + config["batch"]]
            target = y_train[sel]
            batch = [t.to(DEVICE) for t in build_batch(bank_tr, train_uids[sel], dim, rng, aug)]
            opt.zero_grad()
            logits = net(*batch)
            loss = loss_fn(logits, target)
            if config["consistency"] > 0:
                # A second, independently augmented view of the same studies. The penalty is
                # on probabilities rather than logits so it cannot be trivially satisfied by
                # shrinking the logit scale.
                other = [t.to(DEVICE) for t in build_batch(bank_tr, train_uids[sel], dim, rng, aug)]
                loss = loss + config["consistency"] * nn.functional.mse_loss(
                    torch.sigmoid(net(*other)), torch.sigmoid(logits).detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

        p_val = predict(val_batches)
        bce = float(protocol.soft_bce(y_val, p_val).mean())
        if bce < best["bce"]:
            p_gold = predict(gold_batches)
            best = {"bce": bce, "auc": float(np.nanmean(protocol.gold_auc(y_gold, p_gold))),
                    "epoch": epoch, "val": p_val, "gold": p_gold}
    return best


SWEEP = {
    "slice_layers": [1, 2],
    "series_layers": [1, 2],
    "mode": ["plane", "query"],
    "consistency": [0.0, 1.0],
}
FIXED = {"lr": 5e-4, "wd": 1e-2, "batch": 32, "d_model": 256, "dropout": 0.1,
         "slice_keep": 1.0, "slice_dropout": 0.1, "series_dropout": 0.15, "noise": 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="mri_core")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--head", default="cls")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--save-preds", metavar="TAG",
                   help="write results/preds_<TAG>.npz for common/ensemble.py (single config only)")
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

    if args.sweep:
        keys = list(SWEEP)
        configs = [{**FIXED, **dict(zip(keys, c))} for c in itertools.product(*SWEEP.values())]
    else:
        configs = [{**FIXED, **{k: getattr(args, k) for k in SWEEP}}]

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
    if args.save_preds and len(configs) == 1:
        np.savez(out / f"preds_{args.save_preds}.npz", val=best["val"], gold=best["gold"])
    frame = dataset.pd.DataFrame(rows).sort_values("val_soft_bce")
    # A single --save-preds run must not overwrite a full sweep's results file.
    suffix = "" if len(configs) > 1 else f"_single{'_' + args.save_preds if args.save_preds else ''}"
    frame.to_csv(out / f"sweep_{args.backbone}_{args.head}{suffix}.csv", index=False)
    print("\nbest by val soft BCE:")
    print(frame.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
