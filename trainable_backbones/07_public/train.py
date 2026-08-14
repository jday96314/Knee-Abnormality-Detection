#!/usr/bin/env python3
"""Train the public notebooks' architectures under this repository's evaluation protocol.

The notebooks select on macro ROC AUC over a grouped holdout of their own report labels
and report a Kaggle score. Neither number is comparable to anything else in this sweep,
so both are replaced here by the protocol every other architecture in
`trainable_backbones` answers to:

    train      80% of the pseudo-labelled studies
    validate   the remaining 20%, scored by soft binary cross-entropy
    report     ROC AUC over the 58 gold studies, never used to select anything

The split comes from `common/protocol.py` and is the same one architectures 1 and 2 saw,
so the numbers sit in the same table as theirs.

    python train.py --arch dinov2 --img 224                 # one configuration
    python train.py --sweep arch                            # a named sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))

import dataset  # noqa: E402
import protocol  # noqa: E402

import model as M  # noqa: E402
import pixels  # noqa: E402
import targets as T  # noqa: E402

DEVICE = os.environ.get("KNEE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
LABELS = dataset.LABELS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- corpus ----------------------------------------------------------------- #


def kfold_split(studies, fold, n_folds=5):
    """`protocol.make_split` generalised to K folds over the same permutation.

    The fold boundaries are the rounded k/K quantiles of the same shuffle the protocol
    uses, so the last fold *is* the protocol's validation fifth — a K-fold number and a
    single-split number from this file therefore share a fold rather than merely a seed.
    The 58 gold studies stay outside every fold, as they do there.
    """
    rng = np.random.default_rng(protocol.SPLIT_SEED)
    pseudo = np.flatnonzero(~studies["is_gold"].to_numpy())
    order = rng.permutation(len(pseudo))
    cuts = [int(round(k * len(pseudo) / n_folds)) for k in range(n_folds + 1)]
    chunks = [order[cuts[k]:cuts[k + 1]] for k in range(n_folds)]
    val = chunks[fold]
    train = np.concatenate([c for k, c in enumerate(chunks) if k != fold])
    return {"train": np.sort(pseudo[train]), "val": np.sort(pseudo[val]),
            "gold": np.flatnonzero(studies["is_gold"].to_numpy())}


class Corpus:
    """The slot cache joined to the protocol's split and to a choice of teacher."""

    def __init__(self, cfg, source="repo_blend", fold=None, n_folds=5):
        studies, _ = dataset.all_studies()
        uids, cache, mask = pixels.load(cfg)
        pos = {u: i for i, u in enumerate(uids)}
        studies = studies[studies.StudyInstanceUID.isin(pos)].reset_index(drop=True)

        self.cfg = cfg
        self.studies = studies
        self.fold = fold
        self.split = (protocol.make_split(studies) if fold is None
                      else kfold_split(studies, fold, n_folds))
        self.rows = np.array([pos[u] for u in studies.StudyInstanceUID])
        self.cache = cache
        self.mask = mask
        self.n_slot = cache.shape[1]
        self.n_slice = cache.shape[2]
        self.Y, self.W = T.build(source, studies)
        self.source = source

        # The evaluation always reads this repository's blend, whatever the model was
        # taught with; otherwise a teacher could be preferred for being easy to predict.
        self.y_val = protocol.soft_targets(studies, self.split["val"])
        self.y_gold = protocol.gold_targets(studies, self.split["gold"])

    def calibration_rows(self, seed=0, n=512):
        rng = np.random.default_rng(1000 + seed)
        tr = self.split["train"]
        return np.sort(rng.choice(tr, size=min(n, len(tr)), replace=False))

    def batch(self, sel, group, device, group_size=3):
        """One bag of slot images: [B, n_slot, group_size, img, img] uint8."""
        rows = self.rows[sel]
        order = np.argsort(rows)          # a memmap wants ascending, gather order back
        take = self.cache[rows[order], :, group * group_size:(group + 1) * group_size]
        back = np.empty_like(order)
        back[order] = np.arange(len(order))
        imgs = torch.from_numpy(np.ascontiguousarray(take[back])).to(device, non_blocking=True)
        m = torch.from_numpy(self.mask[rows]).to(device)
        return imgs, m


# --- training --------------------------------------------------------------- #


@torch.no_grad()
def predict(model, corpus, sel, img_size, group_size=3, batch=16):
    """Average the logits over the groups of each slot.

    Training sees one group at a time, which acts as augmentation along the stack;
    inference averages over all of them so a prediction does not depend on which group a
    single draw happened to pick.
    """
    model.eval()
    n_group = max(corpus.n_slice // group_size, 1)
    out = []
    for b in range(0, len(sel), batch):
        part = sel[b:b + batch]
        acc = None
        for g in range(n_group):
            imgs, m = corpus.batch(part, g, DEVICE, group_size)
            with torch.autocast("cuda", enabled=DEVICE == "cuda", dtype=torch.bfloat16):
                z = model(imgs, m, img_size).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / n_group).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)), np.float32)


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_calibration(p_fit, y_fit, steps=300):
    """Per-finding affine-in-logit recalibration, fitted on training-split predictions.

    A model taught by a different teacher predicts that teacher's marginal rate, not the
    blend's — the public report labels call a finding present roughly half again as often
    — so its soft BCE against the blend is dominated by an offset rather than by whether
    it ordered the studies correctly. Two parameters per finding, fitted on studies the
    model trained on and never on the validation fifth, separate the two.
    """
    z = torch.from_numpy(_logit(p_fit).astype(np.float32))
    y = torch.from_numpy(np.asarray(y_fit, np.float32))
    a = torch.ones(z.shape[1], requires_grad=True)
    b = torch.zeros(z.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z * a + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    return a.detach().numpy(), b.detach().numpy()


def apply_calibration(cal, p):
    a, b = cal
    return 1.0 / (1.0 + np.exp(-(_logit(p) * a + b)))


def run_slot_model(corpus, config, seed=0):
    """Fine-tune one encoder + slot head and return its best epoch by val soft BCE."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    model = M.build_slot_model(
        arch=config["arch"], img=config["img"], n_slot=corpus.n_slot,
        unfreeze_last=config["unfreeze"], pool=config["pool"], head=config["head"],
        prior=config["prior"]).to(DEVICE)
    trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": trainable, "lr": config["lr_backbone"]},
         {"params": model.head.parameters(), "lr": config["lr_head"]}],
        weight_decay=config["wd"])

    tr, va, go = corpus.split["train"], corpus.split["val"], corpus.split["gold"]
    batch = config["batch"]
    epochs = config["epochs"]
    steps = max(epochs * (len(tr) // batch), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[config["lr_backbone"], config["lr_head"]], total_steps=steps,
        pct_start=0.15)

    Y = torch.from_numpy(corpus.Y).to(DEVICE)
    W = torch.from_numpy(corpus.W).to(DEVICE)
    best, best_state = {"bce": np.inf}, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        tot, nstep = 0.0, 0
        for b in range(0, len(perm) - batch + 1, batch):
            sel = perm[b:b + batch]
            g = int(rng.integers(max(corpus.n_slice // config["group"], 1)))
            imgs, m = corpus.batch(sel, g, DEVICE, config["group"])
            if config["aug"]:
                imgs = M.augment(imgs, generator=gen)
            y, w = Y[sel], W[sel]
            with torch.autocast("cuda", enabled=DEVICE == "cuda", dtype=torch.bfloat16):
                z = model(imgs, m, config["img"])
                loss = (F.binary_cross_entropy_with_logits(
                    z.float(), y, reduction="none") * w).sum() / w.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nstep += 1

        p_val = predict(model, corpus, va, config["img"], config["group"])
        bce = float(protocol.soft_bce(corpus.y_val, p_val).mean())
        p_gold = predict(model, corpus, go, config["img"], config["group"])
        auc = float(np.nanmean(protocol.gold_auc(corpus.y_gold, p_gold)))
        log(f"    epoch {ep + 1}/{epochs} loss {tot / max(nstep, 1):.4f} "
            f"val soft BCE {bce:.4f}  gold AUC {auc:.4f}")
        if bce < best["bce"]:
            best = {"bce": bce, "auc": auc, "epoch": ep, "p_val": p_val, "p_gold": p_gold}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # The calibration set is drawn from the training split, so nothing the validation
    # fifth carries reaches it. It is a subset because a full pass costs as much as an
    # epoch and two parameters per finding do not need 3,479 studies to fit.
    model.load_state_dict(best_state)
    fit = corpus.calibration_rows(seed)
    p_fit = predict(model, corpus, fit, config["img"], config["group"])
    best["cal"] = fit_calibration(p_fit, protocol.soft_targets(corpus.studies, fit))
    del model, opt, best_state
    torch.cuda.empty_cache()
    return best


# --- family B: frozen RadImageNet features ---------------------------------- #


def rad_features(corpus, batch=128):
    """One frozen RadImageNet embedding per acquired slice, cached to disk.

    The encoder never trains in this arm, so the embeddings are a property of the pixel
    configuration alone and are computed once.
    """
    tag = pixels.cache_tag(corpus.cfg)
    path = pixels.CACHE / f"{tag}.rad.f16.npy"
    n, slots, slices = corpus.cache.shape[:3]
    token_mask = np.repeat(corpus.mask[:, :, None], slices, axis=2).reshape(n, -1)
    if path.exists():
        return np.load(path, mmap_mode="r"), token_mask.astype(np.float32)

    encoder = M.load_radimagenet(DEVICE)
    feats = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(n, slots * slices, M.RAD_TOKEN_DIM))
    flat_mask = token_mask.reshape(-1) > 0
    log(f"RadImageNet: encoding {int(flat_mask.sum()):,} acquired slices")
    t0 = time.time()
    for start in range(0, n, 8):
        stop = min(start + 8, n)
        block = np.ascontiguousarray(corpus.cache[start:stop]).reshape(-1, *corpus.cache.shape[3:])
        keep = np.flatnonzero(flat_mask[start * slots * slices:stop * slots * slices])
        out = np.zeros((len(block), M.RAD_TOKEN_DIM), np.float16)
        for b in range(0, len(keep), batch):
            ix = keep[b:b + batch]
            # The official contract: [-1, 1] over a grey image repeated into three channels.
            x = torch.from_numpy(block[ix]).to(DEVICE).float().div_(127.5).sub_(1.0)
            x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
            with torch.autocast("cuda", enabled=DEVICE == "cuda", dtype=torch.bfloat16):
                out[ix] = encoder(x).float().cpu().numpy().astype(np.float16)
        feats[start:stop] = out.reshape(stop - start, slots * slices, M.RAD_TOKEN_DIM)
        if start % 800 == 0:
            log(f"  {start}/{n} studies ({time.time() - t0:.0f}s)")
    feats.flush()
    del encoder
    torch.cuda.empty_cache()
    return np.load(path, mmap_mode="r"), token_mask.astype(np.float32)


def run_rad_model(corpus, config, seed=0):
    """Train the query head over frozen RadImageNet slice embeddings."""
    feats, token_mask = rad_features(corpus)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = torch.from_numpy(np.asarray(feats[corpus.rows], np.float16))
    Mk = torch.from_numpy(token_mask[corpus.rows])
    Y = torch.from_numpy(corpus.Y)
    W = torch.from_numpy(corpus.W)

    head = M.FoundationQueryHead(corpus.n_slot, corpus.n_slice).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=config["lr_head"],
                            weight_decay=config["wd"])
    tr, va, go = corpus.split["train"], corpus.split["val"], corpus.split["gold"]
    batch = config["batch"]

    def infer(sel):
        head.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(sel), 256):
                part = sel[b:b + 256]
                z = head(X[part].to(DEVICE), Mk[part].to(DEVICE))
                out.append(torch.sigmoid(z.float()).cpu().numpy())
        return np.concatenate(out)

    best = {"bce": np.inf}
    for ep in range(config["epochs"]):
        head.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm) - 1, batch):
            sel = perm[b:b + batch]
            z = head(X[sel].to(DEVICE), Mk[sel].to(DEVICE))
            y, w = Y[sel].to(DEVICE), W[sel].to(DEVICE)
            loss = (F.binary_cross_entropy_with_logits(z.float(), y, reduction="none")
                    * w).sum() / w.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        p_val, p_gold = infer(va), infer(go)
        bce = float(protocol.soft_bce(corpus.y_val, p_val).mean())
        auc = float(np.nanmean(protocol.gold_auc(corpus.y_gold, p_gold)))
        log(f"    epoch {ep + 1}/{config['epochs']} val soft BCE {bce:.4f}  gold AUC {auc:.4f}")
        if bce < best["bce"]:
            best = {"bce": bce, "auc": auc, "epoch": ep, "p_val": p_val, "p_gold": p_gold,
                    "p_fit": infer(corpus.calibration_rows(seed))}
    best["cal"] = fit_calibration(
        best["p_fit"], protocol.soft_targets(corpus.studies, corpus.calibration_rows(seed)))
    del head, opt
    torch.cuda.empty_cache()
    return best


# --- driver ----------------------------------------------------------------- #


DEFAULTS = {
    "arch": "dinov2", "img": 224, "scheme": "recovered", "slices": 9, "group": 3,
    "crop_mm": pixels.CROP_MM, "band": pixels.SLICE_BAND, "rules": "native",
    "unfreeze": 6, "pool": "cls_mean", "head": "slot", "prior": False, "aug": True,
    "lr_backbone": 8e-6, "lr_head": 1e-3, "wd": 0.02, "batch": 8, "epochs": 10,
    "source": "repo_blend", "cache_img": 336,
}

# One sweep, each entry varying a single axis away from the reference. A product over
# everything would be 500 fine-tunes; the axes here are the ones the notebooks argue for,
# and each is worth measuring against the same reference rather than against each other.
REFERENCE = {"name": "reference: dinov2-224"}
SWEEPS = {
    "main": [
        REFERENCE,
        # what the encoder is, and how much of the anatomy it sees
        {"name": "dinov2-336", "img": 336},
        {"name": "dinov3-224", "arch": "dinov3"},
        {"name": "dinov3-336", "arch": "dinov3", "img": 336},
        # how far the encoder is opened
        {"name": "frozen encoder", "unfreeze": 0},
        {"name": "unfreeze 2", "unfreeze": 2},
        {"name": "unfreeze 12", "unfreeze": 12},
        # the aggregation the notebooks argue for
        {"name": "mean over slots", "head": "mean"},
        {"name": "slot attention + anatomy prior", "prior": True},
        {"name": "focal patch pooling", "pool": "cls_mean_focal"},
        # the pixel dataset
        {"name": "public slot scheme", "scheme": "public"},
        {"name": "no augmentation", "aug": False},
        {"name": "30 epochs", "epochs": 30},
        # the label dataset
        {"name": "teacher: public mean", "source": "public_mean"},
        {"name": "teacher: public mean, unweighted", "source": "public_mean_flat"},
        {"name": "teacher: report_labels_v2", "source": "report_v2"},
        # the independent arm
        {"name": "RadImageNet + query head", "arch": "rad", "scheme": "rad", "img": 224,
         "cache_img": 224, "slices": 8, "group": 8, "crop_mm": 10_000.0,
         "band": (0.12, 0.88), "lr_head": 2e-4, "wd": 3e-3, "batch": 48, "epochs": 24},
    ],
}
# A long schedule, once the ten-epoch runs turned out not to have converged, and a sweep
# of the RadImageNet arm — which is frozen, so five folds of it cost under three minutes
# and there is no reason to measure it on one split.
SWEEPS["long"] = [
    {"name": "dinov2-336 x30", "img": 336, "epochs": 30},
    {"name": "dinov2-224 x30 unfreeze 12", "unfreeze": 12, "epochs": 30},
]
RAD = {"arch": "rad", "scheme": "rad", "img": 224, "cache_img": 224, "slices": 8,
       "group": 8, "crop_mm": 10_000.0, "band": (0.12, 0.88), "lr_head": 2e-4,
       "wd": 3e-3, "batch": 48, "epochs": 24}
SWEEPS["radsweep"] = [
    {"name": "rad: reference", **RAD},
    {"name": "rad: 60 epochs", **RAD, "epochs": 60},
    {"name": "rad: lr 5e-4", **RAD, "lr_head": 5e-4},
    {"name": "rad: lr 1e-4", **RAD, "lr_head": 1e-4},
    {"name": "rad: teacher public mean", **RAD, "source": "public_mean"},
    {"name": "rad: 130mm crop", **RAD, "crop_mm": 130.0},
    {"name": "rad: 6 recovered slots", **RAD, "scheme": "recovered", "slices": 9,
     "group": 9, "cache_img": 336, "crop_mm": 130.0, "band": (0.20, 0.80)},
]

# Follow-ups around the 130 mm crop, which turned out to be what the RadImageNet arm was
# missing. Physical normalisation matters more to it than anything else tried.
RAD_CROP = {**RAD, "crop_mm": 130.0}
SWEEPS["radsweep2"] = [
    {"name": "radc: reference 130mm", **RAD_CROP},
    {"name": "radc: lr 1e-4", **RAD_CROP, "lr_head": 1e-4},
    {"name": "radc: lr 5e-5", **RAD_CROP, "lr_head": 5e-5},
    {"name": "radc: band 0.20-0.80", **RAD_CROP, "band": (0.20, 0.80)},
    {"name": "radc: 6 recovered slots 130mm", **RAD, "scheme": "recovered", "slices": 9,
     "group": 9, "cache_img": 336, "crop_mm": 130.0, "band": (0.20, 0.80),
     "lr_head": 1e-4},
]
# The long schedule on five folds: the ten-epoch runs had not converged, so the
# cross-validated number for a fine-tuned encoder has to come from a schedule that had.
SWEEPS["long_folds"] = [
    {"name": "dinov2-224 x30", "epochs": 30},
]

# The configurations worth spending five folds on, once the single split has ranked them.
SWEEPS["folds"] = [REFERENCE,
                   {"name": "dinov2-336", "img": 336},
                   {"name": "frozen encoder", "unfreeze": 0},
                   {"name": "RadImageNet + query head", "arch": "rad", "scheme": "rad",
                    "img": 224, "cache_img": 224, "slices": 8, "group": 8,
                    "crop_mm": 10_000.0, "band": (0.12, 0.88), "lr_head": 2e-4,
                    "wd": 3e-3, "batch": 48, "epochs": 24}]


def pixel_config(config):
    rules = pixels.RULES_LEGACY if config["rules"] == "legacy" else pixels.RULES_NATIVE
    return pixels.config(config["scheme"], config["cache_img"], config["slices"],
                         config["crop_mm"], tuple(config["band"]), rules)


def run(config, seed=0, corpora=None, fold=None):
    key = (json.dumps(pixel_config(config), sort_keys=True, default=str),
           config["source"], fold)
    if corpora is not None and key in corpora:
        corpus = corpora[key]
    else:
        corpus = Corpus(pixel_config(config), config["source"], fold)
        if corpora is not None:
            corpora[key] = corpus
    runner = run_rad_model if config["arch"] == "rad" else run_slot_model
    return corpus, runner(corpus, config, seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default=None, choices=list(SWEEPS))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--folds", type=int, default=1,
                   help="1 = the protocol's single 80/20 split; K > 1 = K-fold, "
                        "reported out-of-fold over every pseudo-labelled study")
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--tag", default=None)
    p.add_argument("--only", default=None,
                   help="comma-separated substrings; run only the matching configurations")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", action="store_true", default=v)
        elif isinstance(v, tuple):
            continue
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = p.parse_args()

    base = {**DEFAULTS, **{k: getattr(args, k) for k in DEFAULTS if hasattr(args, k)}}
    configs = ([{**base, **c} for c in SWEEPS[args.sweep]] if args.sweep
               else [{**base, "name": args.tag or base["arch"]}])
    if args.only:
        want = [w.strip() for w in args.only.split(",")]
        configs = [c for c in configs if any(w in c["name"] for w in want)]
    folds = [None] if args.folds <= 1 else list(range(args.folds))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.sweep or args.tag or base["arch"]
    rows, corpora = [], {}
    for i, config in enumerate(configs, 1):
        log(f"[{i}/{len(configs)}] {config['name']}: " +
            " ".join(f"{k}={config[k]}" for k in
                     ("arch", "img", "scheme", "head", "pool", "unfreeze", "source")))
        started = time.perf_counter()
        runs = []
        for fold in folds:
            for seed in range(args.seeds):
                corpus, best = run(config, seed, corpora, fold)
                runs.append({**best, "fold": fold, "corpus": corpus})
                if len(folds) > 1:
                    log(f"  fold {fold} seed {seed}: val soft BCE {best['bce']:.4f}  "
                        f"gold AUC {best['auc']:.4f}")

        # Predictions from different folds score different studies, so they are stacked
        # into one out-of-fold vector rather than averaged. Gold is the other way round:
        # every fold model predicts all 58, so the honest number is the mean of the
        # per-fold AUCs, not the AUC of their averaged prediction, which is an ensemble.
        studies = runs[0]["corpus"].studies
        val_rows = np.concatenate([r["corpus"].split["val"] for r in runs])
        p_val = np.concatenate([r["p_val"] for r in runs])
        cal_val = np.concatenate([apply_calibration(r["cal"], r["p_val"]) for r in runs])
        y_val = protocol.soft_targets(studies, val_rows)
        y_gold = runs[0]["corpus"].y_gold
        p_gold = np.mean([r["p_gold"] for r in runs], axis=0)

        row = protocol.report(
            config["name"], y_val, p_val, y_gold, p_gold,
            extra={"val_soft_bce_cal": float(protocol.soft_bce(y_val, cal_val).mean()),
                   "gold_auc_per_fold": float(np.mean([r["auc"] for r in runs])),
                   "gold_auc_per_fold_std": float(np.std([r["auc"] for r in runs])),
                   "val_bce_per_fold_std": float(np.std([r["bce"] for r in runs])),
                   **{k: str(config[k]) for k in
                      ("arch", "img", "scheme", "head", "pool", "source")},
                   "unfreeze": config["unfreeze"], "prior": config["prior"],
                   "aug": config["aug"], "epochs": config["epochs"],
                   "folds": len(folds), "seeds": args.seeds,
                   "best_epoch": float(np.mean([r["epoch"] for r in runs])),
                   "seconds": round(time.perf_counter() - started, 1)})
        rows.append(row)
        # Kept so arms can be blended afterwards without refitting any of them: the
        # notebooks combine members on ranks, and that is a post-hoc operation over
        # exactly these vectors.
        np.savez(out / f"preds_{name}_{config['name'].replace('/', '-')}.npz",
                 val_rows=val_rows, p_val=p_val, p_gold=p_gold, y_val=y_val,
                 y_gold=y_gold, folds=np.array([r["fold"] for r in runs], dtype=object))
        log(f"  -> {config['name']}: val soft BCE {row['val_soft_bce']:.4f} "
            f"(calibrated {row['val_soft_bce_cal']:.4f})  gold AUC "
            f"{row['gold_auc_per_fold']:.4f} +- {row['gold_auc_per_fold_std']:.4f}"
            f"  ({row['seconds']:.0f}s)")
        pd.DataFrame(rows).to_csv(out / f"sweep_{name}.csv", index=False)

    frame = pd.DataFrame(rows)[["model", "val_soft_bce", "val_soft_bce_cal",
                                "gold_auc_per_fold", "gold_auc", "best_epoch", "seconds"]]
    log("\n" + frame.sort_values("val_soft_bce").to_string(index=False))


if __name__ == "__main__":
    main()
