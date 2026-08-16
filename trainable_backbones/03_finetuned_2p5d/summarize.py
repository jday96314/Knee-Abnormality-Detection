#!/usr/bin/env python3
"""Collate the Optuna studies into tables for the write-up.

    python summarize.py                # everything finished so far
    python summarize.py --importance   # add per-study hyperparameter importances

Three views, because they answer different questions:

  * the leaderboard  — which (backbone, head) pair won, and on how many trials
  * best parameters  — what the winners actually chose, side by side
  * importances      — which axes the search found mattered, per study

The trial count is printed beside every result on purpose. These studies are budget-capped
rather than convergence-capped, so a pair with 4 trials and a pair with 14 are not equally
strong evidence, and a leaderboard that hides that would be misleading.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import optuna
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DB_DIR = Path(os.environ.get("KNEE_OPTUNA_DIR", "/mnt/data01/knee_optuna"))
STORAGE = f"sqlite:///{DB_DIR / 'optuna.db'}"

AUG_KEYS = ["geom", "intensity", "noise", "bias", "blur", "erase", "phase", "series_dropout"]
OPT_KEYS = ["unfreeze", "lr", "enc_lr_scale", "wd", "batch", "epochs", "schedule",
            "dropout", "kernel"]


def load():
    rows, studies = [], {}
    for name in sorted(optuna.study.get_all_study_names(storage=STORAGE)):
        s = optuna.load_study(study_name=name, storage=STORAGE)
        studies[name] = s
        # Optuna stores the last reported intermediate value on PRUNED trials too, so
        # "value is not None" counts pruned trials as done. Only COMPLETE trials ran to
        # their full epoch budget and are eligible to be the study's best.
        complete = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            continue
        b = s.best_trial
        backbone, head = name.split("__")
        rows.append({
            "backbone": backbone, "head": head,
            "val_soft_bce": b.value,
            "gold_auc": b.user_attrs.get("gold_auc", float("nan")),
            "complete": len(complete),
            "trials_total": len(s.trials),
            "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in s.trials),
            **{k: b.params.get(k) for k in OPT_KEYS + AUG_KEYS},
        })
    return pd.DataFrame(rows), studies


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--importance", action="store_true")
    args = p.parse_args()

    frame, studies = load()
    if frame.empty:
        raise SystemExit("no completed trials yet")
    frame = frame.sort_values("val_soft_bce").reset_index(drop=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "tuning_summary.csv", index=False)

    lead = frame[["backbone", "head", "val_soft_bce", "gold_auc", "complete", "pruned", "trials_total"]]
    print("=== leaderboard ===")
    print(lead.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== chosen optimisation settings ===")
    print(frame[["backbone", "head"] + OPT_KEYS].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n=== chosen augmentation strengths (0 = off) ===")
    print(frame[["backbone", "head"] + AUG_KEYS].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))

    if args.importance:
        print("\n=== hyperparameter importance (fANOVA, completed trials only) ===")
        for name, s in studies.items():
            done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(done) < 6:
                print(f"{name}: {len(done)} complete trials — too few to attribute importance")
                continue
            try:
                imp = optuna.importance.get_param_importances(s)
            except Exception as exc:                      # noqa: BLE001
                print(f"{name}: importance failed ({type(exc).__name__})")
                continue
            top = ", ".join(f"{k} {v:.2f}" for k, v in list(imp.items())[:5])
            print(f"{name:24s} ({len(done):2d} trials)  {top}")

    print(f"\nwrote {RESULTS / 'tuning_summary.csv'}")


if __name__ == "__main__":
    main()
