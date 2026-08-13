#!/usr/bin/env python3
"""Blend the text-only and image-only predictions for all 4,407 training studies.

Uses the shared-weight strategy selected in `blend_experiments.py`: a logistic model on
the two logits, with one text slope and one image slope shared across all twelve findings,
plus a per-finding intercept. Fitted on the 58 gold-labelled studies.

    python make_blended_predictions.py

Two properties of the output matter:

* **The 58 labelled studies get out-of-fold predictions.** They trained the blender, so an
  in-sample prediction for them would be optimistic and would quietly contaminate any
  downstream evaluation that uses this file. Each is predicted only by fold-models that
  did not see it. The other 4,349 get the average over all fold-models.
* **`__sample_std` is disagreement between fold-models**, not the input models' own
  uncertainty. It is large where the blender's parameters are poorly determined, which on
  58 studies is worth knowing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "image_only"))
from dicom_io import LABELS  # noqa: E402

from blend_experiments import RNG_SEED, auc, bce, logit  # noqa: E402

N_SPLITS = 5
N_REPEATS = 10


def design(text_logit, image_logit, label_index, n_labels):
    """[shared text slope | shared image slope | per-finding intercepts]."""
    onehot = np.zeros((len(label_index), n_labels))
    onehot[np.arange(len(label_index)), label_index] = 1.0
    return np.hstack([text_logit.reshape(-1, 1), image_logit.reshape(-1, 1), onehot])


def fit_model(T, I, Y, rows, C=1.0):
    idx = np.array([r[0] for r in rows])
    lab = np.array([r[1] for r in rows])
    X = design(logit(T[idx, lab]), logit(I[idx, lab]), lab, Y.shape[1])
    model = LogisticRegression(C=C, solver="lbfgs", fit_intercept=False, max_iter=4000)
    model.fit(X, Y[idx, lab])
    return model


def predict_all(model, T_all, I_all, n_labels):
    """Predict every (study, finding) pair for a whole prediction matrix."""
    out = np.zeros_like(T_all)
    for j in range(n_labels):
        X = design(
            logit(T_all[:, j]), logit(I_all[:, j]), np.full(len(T_all), j), n_labels
        )
        out[:, j] = model.predict_proba(X)[:, 1]
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(here / "predictions"))
    parser.add_argument("--filename", default="blended_predictions.csv")
    args = parser.parse_args()

    text = pd.read_csv(root / "text_only" / "predictions" / "text_only_predictions.csv")
    image = pd.read_csv(root / "image_only" / "predictions" / "image_only_predictions.csv")
    labels = pd.read_csv(root / "image_only" / "artifacts" / "cohort_labels.csv")

    merged = text[["source_row", "StudyInstanceUID"]].copy()
    T_all = text.set_index("StudyInstanceUID").reindex(merged["StudyInstanceUID"])[LABELS].to_numpy(float)
    I_all = image.set_index("StudyInstanceUID").reindex(merged["StudyInstanceUID"])[LABELS].to_numpy(float)
    if np.isnan(T_all).any() or np.isnan(I_all).any():
        raise SystemExit("text and image predictions do not cover the same studies")

    gold_uids = labels["StudyInstanceUID"].tolist()
    position = {u: i for i, u in enumerate(merged["StudyInstanceUID"])}
    gold_rows = np.array([position[u] for u in gold_uids])
    T_gold, I_gold = T_all[gold_rows], I_all[gold_rows]
    Y = labels[LABELS].to_numpy(float)
    n, L = len(Y), len(LABELS)

    # Every fold-model predicts the whole dataset; gold studies keep only the folds that
    # excluded them, so their predictions stay honest.
    all_preds, oof = [], np.full((n, L, N_REPEATS), np.nan)
    for rep in range(N_REPEATS):
        splitters = [
            list(
                RepeatedStratifiedKFold(
                    n_splits=N_SPLITS, n_repeats=1, random_state=RNG_SEED + rep
                ).split(np.zeros(n), Y[:, j])
            )
            for j in range(L)
        ]
        for k in range(N_SPLITS):
            rows_tr = [(i, j) for j in range(L) for i in splitters[j][k][0]]
            model = fit_model(T_gold, I_gold, Y, rows_tr, args.C)
            all_preds.append(predict_all(model, T_all, I_all, L))
            for j in range(L):
                for i in splitters[j][k][1]:
                    oof[i, j, rep] = all_preds[-1][gold_rows[i], j]

    stack = np.stack(all_preds)
    mean_pred, std_pred = stack.mean(axis=0), stack.std(axis=0)

    # Substitute out-of-fold values for the studies that trained the blender.
    final = mean_pred.copy()
    final[gold_rows] = np.nanmean(oof, axis=2)
    final_std = std_pred.copy()
    final_std[gold_rows] = np.nanstd(oof, axis=2)

    out = merged.copy()
    for j, label in enumerate(LABELS):
        out[label] = final[:, j]
        out[f"{label}__sample_std"] = final_std[:, j]

    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / args.filename
    out.to_csv(path, index=False)

    # A model fitted on everything, purely to report the weights it chose.
    full = fit_model(T_gold, I_gold, Y, [(i, j) for j in range(L) for i in range(n)], args.C)
    gold_pred = final[gold_rows]
    per_label = {
        LABELS[j]: {
            "bce": round(bce(Y[:, j], gold_pred[:, j]), 4),
            "auc": round(auc(Y[:, j], gold_pred[:, j]), 4),
        }
        for j in range(L)
    }
    meta = {
        "method": "shared-weight logistic blend of text and image logits",
        "fitted_on": "the 58 gold-labelled studies",
        "C": args.C,
        "text_slope": round(float(full.coef_[0][0]), 4),
        "image_slope": round(float(full.coef_[0][1]), 4),
        "per_finding_intercept": {
            LABELS[j]: round(float(full.coef_[0][2 + j]), 4) for j in range(L)
        },
        "out_of_fold_macro_bce": round(float(np.mean([v["bce"] for v in per_label.values()])), 4),
        "out_of_fold_macro_auc": round(float(np.mean([v["auc"] for v in per_label.values()])), 4),
        "per_finding": per_label,
        "note": (
            "The 58 gold studies hold out-of-fold predictions; the remaining 4,349 hold the "
            "mean over all fold-models. Metrics above are out-of-fold and therefore honest, "
            "but the blender was selected on these same studies, so they still omit "
            "selection uncertainty."
        ),
    }
    (target / (Path(args.filename).stem + "_meta.json")).write_text(json.dumps(meta, indent=2))

    print(f"wrote {path}  ({len(out)} rows)")
    print(f"  text slope {meta['text_slope']}, image slope {meta['image_slope']}")
    print(f"  out-of-fold macro BCE {meta['out_of_fold_macro_bce']}, AUC {meta['out_of_fold_macro_auc']}")
    print("\nper-finding distribution of the blended targets:")
    print(out[LABELS].describe().T[["mean", "std", "min", "50%", "max"]].to_string(
        float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
