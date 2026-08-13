#!/usr/bin/env python3
"""Scale and blend the text-only and image-only predictions, minimising per-finding BCE.

BCE is a different objective from everything measured previously. AUC only cares about
ordering, so it is invariant to any monotone rescaling; BCE cares about the numbers
themselves. That changes the conclusions: the image predictions rank respectably
(0.70 macro AUC) yet score *worse than predicting the base rate* under BCE, because they
are uncalibrated. Calibration is therefore not a refinement here, it is the main event.

Every method with fitted parameters is evaluated out-of-fold under repeated stratified
k-fold, because there are only 58 labelled studies and 9-35 positives per finding. An
in-sample number on this data is close to meaningless — the gap between in-sample and
out-of-fold is reported so it can be seen rather than assumed.

    python blend_experiments.py
    python blend_experiments.py --repeats 20 --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "image_only"))
from dicom_io import LABELS  # noqa: E402

EPS = 1e-6
RNG_SEED = 20260813


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ---------------------------------------------------------------------------
# Methods. Each fits on the training rows and returns predictions for both sets,
# so the in-sample/out-of-fold gap can be reported.
# ---------------------------------------------------------------------------


@dataclass
class Fitted:
    train: np.ndarray
    test: np.ndarray


Method = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Fitted]


def _const(y_tr, n_tr, n_te) -> Fitted:
    rate = float(np.clip(y_tr.mean(), EPS, 1 - EPS))
    return Fitted(np.full(n_tr, rate), np.full(n_te, rate))


def m_prevalence(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    """Predict the training base rate for everyone. The bar any method must clear."""
    return _const(y_tr, len(t_tr), len(t_te))


def m_raw_text(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    return Fitted(t_tr, t_te)


def m_raw_image(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    return Fitted(i_tr, i_te)


def _platt(x_tr, y_tr, x_te, C=1.0):
    """Logistic recalibration on the logit: p' = sigmoid(a * logit(p) + b).

    Two parameters, which is about the most that 9-35 positives can support.
    """
    if len(np.unique(y_tr)) < 2:
        rate = float(np.clip(y_tr.mean(), EPS, 1 - EPS))
        return np.full(len(x_tr), rate), np.full(len(x_te), rate)
    model = LogisticRegression(C=C, solver="lbfgs")
    model.fit(logit(x_tr).reshape(-1, 1), y_tr)
    return (
        model.predict_proba(logit(x_tr).reshape(-1, 1))[:, 1],
        model.predict_proba(logit(x_te).reshape(-1, 1))[:, 1],
    )


def m_platt_text(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    return Fitted(*_platt(t_tr, y_tr, t_te))


def m_platt_image(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    return Fitted(*_platt(i_tr, y_tr, i_te))


def m_temperature_text(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    """Scale the logit by a single temperature, no intercept — one parameter."""
    grid = np.linspace(0.1, 3.0, 60)
    best = min(grid, key=lambda s: bce(y_tr, sigmoid(s * logit(t_tr))))
    return Fitted(sigmoid(best * logit(t_tr)), sigmoid(best * logit(t_te)))


def m_isotonic_text(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    """Nonparametric calibration. Included to show it overfits at this sample size."""
    if len(np.unique(y_tr)) < 2:
        return _const(y_tr, len(t_tr), len(t_te))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    iso.fit(t_tr, y_tr)
    return Fitted(iso.predict(t_tr), iso.predict(t_te))


def _blend_logreg(t_tr, i_tr, y_tr, t_te, i_te, C=1.0, use_image=True):
    """Logistic regression on the two logits: learns blend weights and calibration jointly."""
    if len(np.unique(y_tr)) < 2:
        rate = float(np.clip(y_tr.mean(), EPS, 1 - EPS))
        return np.full(len(t_tr), rate), np.full(len(t_te), rate)
    cols_tr = [logit(t_tr)] + ([logit(i_tr)] if use_image else [])
    cols_te = [logit(t_te)] + ([logit(i_te)] if use_image else [])
    X_tr, X_te = np.column_stack(cols_tr), np.column_stack(cols_te)
    model = LogisticRegression(C=C, solver="lbfgs")
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_tr)[:, 1], model.predict_proba(X_te)[:, 1]


def m_logreg_blend(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    return Fitted(*_blend_logreg(t_tr, i_tr, y_tr, t_te, i_te, C=1.0))


def m_logreg_blend_reg(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    """Same, heavily regularised — three parameters on ~46 training rows is a lot."""
    return Fitted(*_blend_logreg(t_tr, i_tr, y_tr, t_te, i_te, C=0.1))


def m_fixed_logit_blend(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    """Blend weight chosen on the training fold, then Platt-calibrated.

    Separating "how much image" from "what scale" keeps each step to one or two
    parameters, which is more robust here than fitting three at once.
    """
    grid = np.arange(0.0, 0.55, 0.05)
    def mix(w, t, i):
        return w * logit(i) + (1 - w) * logit(t)
    best, best_score = 0.0, np.inf
    for w in grid:
        p_tr, _ = _platt(sigmoid(mix(w, t_tr, i_tr)), y_tr, sigmoid(mix(w, t_te, i_te)))
        s = bce(y_tr, p_tr)
        if s < best_score:
            best, best_score = w, s
    tr, te = _platt(sigmoid(mix(best, t_tr, i_tr)), y_tr, sigmoid(mix(best, t_te, i_te)))
    return Fitted(tr, te)


def m_platt_text_then_image(t_tr, i_tr, y_tr, t_te, i_te, _y_te) -> Fitted:
    """Calibrate text alone, then add a small fixed image contribution in logit space."""
    tr_t, te_t = _platt(t_tr, y_tr, t_te)
    tr_i, te_i = _platt(i_tr, y_tr, i_te)
    grid = np.arange(0.0, 0.55, 0.05)
    best = min(grid, key=lambda w: bce(y_tr, sigmoid((1 - w) * logit(tr_t) + w * logit(tr_i))))
    return Fitted(
        sigmoid((1 - best) * logit(tr_t) + best * logit(tr_i)),
        sigmoid((1 - best) * logit(te_t) + best * logit(te_i)),
    )


METHODS: dict[str, Method] = {
    "prevalence (baseline)": m_prevalence,
    "text raw": m_raw_text,
    "image raw": m_raw_image,
    "text + temperature": m_temperature_text,
    "text + Platt": m_platt_text,
    "text + isotonic": m_isotonic_text,
    "image + Platt": m_platt_image,
    "blend: logreg(text,image) C=1": m_logreg_blend,
    "blend: logreg(text,image) C=0.1": m_logreg_blend_reg,
    "blend: weighted logit + Platt": m_fixed_logit_blend,
    "blend: Platt each, then weight": m_platt_text_then_image,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _fit_shared(X_tr, y_tr, lab_tr, X_te, lab_te, n_labels, C=1.0):
    """One slope shared by every finding, plus a per-finding intercept.

    Partial pooling. A separate Platt fit per finding spends two parameters on as few as
    9 positives; pooling the slope across all twelve estimates it from ~700 rows while
    still letting each finding keep its own base rate. That is the main defence against
    overfitting at this sample size.
    """
    onehot_tr = np.zeros((len(y_tr), n_labels))
    onehot_tr[np.arange(len(y_tr)), lab_tr] = 1.0
    onehot_te = np.zeros((X_te.shape[0], n_labels))
    onehot_te[np.arange(X_te.shape[0]), lab_te] = 1.0
    A_tr = np.hstack([X_tr, onehot_tr])
    A_te = np.hstack([X_te, onehot_te])
    model = LogisticRegression(C=C, solver="lbfgs", fit_intercept=False, max_iter=2000)
    model.fit(A_tr, y_tr)
    return model.predict_proba(A_te)[:, 1], model.predict_proba(A_tr)[:, 1]


def evaluate_shared(T, I, Y, repeats, splits, use_image, C=1.0, name=""):
    """Cross-validated shared-slope calibration, optionally blending in the image logit.

    Folds are taken per finding on that finding's labels, then the training rows of all
    findings are pooled to fit the shared parameters, so no study contributes to a fit
    that scores it.
    """
    n = Y.shape[0]; L = len(LABELS)
    oof = np.full((n, L, repeats), np.nan)
    insample = []
    for rep in range(repeats):
        splitters = [
            list(RepeatedStratifiedKFold(n_splits=splits, n_repeats=1,
                                         random_state=RNG_SEED + rep).split(np.zeros(n), Y[:, j]))
            for j in range(L)
        ]
        for k in range(splits):
            rows_tr, rows_te = [], []
            for j in range(L):
                tr, te = splitters[j][k]
                rows_tr += [(idx, j) for idx in tr]
                rows_te += [(idx, j) for idx in te]
            def design(rows):
                idx = np.array([r[0] for r in rows]); lab = np.array([r[1] for r in rows])
                cols = [logit(T[idx, lab])] + ([logit(I[idx, lab])] if use_image else [])
                return np.column_stack(cols), idx, lab
            X_tr, idx_tr, lab_tr = design(rows_tr)
            X_te, idx_te, lab_te = design(rows_te)
            y_tr = Y[idx_tr, lab_tr]
            pred_te, pred_tr = _fit_shared(X_tr, y_tr, lab_tr, X_te, lab_te, L, C)
            insample.append(bce(y_tr, pred_tr))
            for (i_, j_), p in zip(zip(idx_te, lab_te), pred_te):
                oof[i_, j_, rep] = p
    pooled = np.nanmean(oof, axis=2)
    per_label = [
        {"label": LABELS[j], "bce": bce(Y[:, j], pooled[:, j]),
         "auc": auc(Y[:, j], pooled[:, j]), "bce_insample": float(np.mean(insample))}
        for j in range(L)
    ]
    return {
        "method": name,
        "bce": float(np.mean([r["bce"] for r in per_label])),
        "auc": float(np.nanmean([r["auc"] for r in per_label])),
        "bce_insample": float(np.mean(insample)),
        "per_label": per_label,
        "_pooled": pooled,
    }


def evaluate(T: np.ndarray, I: np.ndarray, Y: np.ndarray, repeats: int, splits: int):
    """Out-of-fold BCE and AUC per finding, plus the in-sample gap."""
    rows = []
    for name, method in METHODS.items():
        oof = np.full((Y.shape[0], len(LABELS), repeats), np.nan)
        insample = np.zeros((len(LABELS), repeats * splits))
        for j in range(len(LABELS)):
            y = Y[:, j]
            cv = RepeatedStratifiedKFold(
                n_splits=splits, n_repeats=repeats, random_state=RNG_SEED
            )
            for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
                fit = method(T[tr, j], I[tr, j], y[tr], T[te, j], I[te, j], y[te])
                oof[te, j, k // splits] = fit.test
                insample[j, k] = bce(y[tr], fit.train)
        pooled = np.nanmean(oof, axis=2)
        per_label = [
            {
                "label": LABELS[j],
                "bce": bce(Y[:, j], pooled[:, j]),
                "auc": auc(Y[:, j], pooled[:, j]),
                "bce_insample": float(insample[j].mean()),
            }
            for j in range(len(LABELS))
        ]
        rows.append(
            {
                "method": name,
                "bce": float(np.mean([r["bce"] for r in per_label])),
                "auc": float(np.nanmean([r["auc"] for r in per_label])),
                "bce_insample": float(np.mean([r["bce_insample"] for r in per_label])),
                "per_label": per_label,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--out", default=str(here / "results"))
    args = parser.parse_args()

    root = here.parent
    labels = pd.read_csv(root / "image_only" / "artifacts" / "cohort_labels.csv")
    uids = labels["StudyInstanceUID"]
    text = pd.read_csv(root / "text_only" / "predictions" / "text_only_predictions.csv")
    image = pd.read_csv(root / "image_only" / "predictions" / "image_only_predictions.csv")
    T = text.set_index("StudyInstanceUID").reindex(uids)[LABELS].to_numpy(float)
    I = image.set_index("StudyInstanceUID").reindex(uids)[LABELS].to_numpy(float)
    Y = labels[LABELS].to_numpy(float)

    rows = evaluate(T, I, Y, args.repeats, args.splits)
    for use_image, C, nm in [
        (False, 1.0, "shared slope, text only"),
        (True, 1.0, "shared slope, text+image"),
        (True, 0.3, "shared slope, text+image (C=0.3)"),
    ]:
        r = evaluate_shared(T, I, Y, args.repeats, args.splits, use_image, C, nm)
        r.pop("_pooled", None)
        rows.append(r)
    rows.sort(key=lambda r: r["bce"])

    print(f"{len(uids)} labelled studies, {args.repeats}x{args.splits}-fold out-of-fold\n")
    print(f"{'method':34s} {'BCE':>7s} {'AUC':>7s} {'BCE in-sample':>14s} {'overfit gap':>12s}")
    print("-" * 80)
    for r in rows:
        gap = r["bce"] - r["bce_insample"]
        print(
            f"{r['method']:34s} {r['bce']:7.4f} {r['auc']:7.4f} "
            f"{r['bce_insample']:14.4f} {gap:+12.4f}"
        )

    best = rows[0]
    print(f"\nbest by BCE: {best['method']}")
    print(f"\n{'finding':18s} {'BCE':>7s} {'AUC':>7s}")
    for r in best["per_label"]:
        print(f"  {r['label']:16s} {r['bce']:7.4f} {r['auc']:7.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "per_label"} for r in rows]).to_csv(
        out / "methods.csv", index=False
    )
    pd.DataFrame(
        [{**p, "method": r["method"]} for r in rows for p in r["per_label"]]
    ).to_csv(out / "per_label.csv", index=False)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/methods.csv, per_label.csv, summary.json")


if __name__ == "__main__":
    main()


def evaluate_hierarchical(T, I, Y, repeats, splits, mode, C=1.0, name=""):
    """Blend weight per finding, calibration slope pooled across findings.

    The middle ground between the two extremes already measured. A fully per-finding fit
    spends every parameter on ~46 rows and overfits; a fully shared fit forces one
    image weight on all twelve findings even though the image channel is known to be
    strong on effusion and useless on ACL. Here the text slope and the intercepts stay
    per-finding-or-pooled as before, but the image gets its own coefficient per finding,
    estimated jointly so the shared parts still borrow strength.

    mode:
      'image_per_label'  pooled text slope, per-finding image weight
      'both_per_label'   per-finding text slope and image weight (pooled intercepts only)
    """
    n, L = Y.shape[0], len(LABELS)
    oof = np.full((n, L, repeats), np.nan)
    insample, weights = [], []
    for rep in range(repeats):
        splitters = [
            list(RepeatedStratifiedKFold(n_splits=splits, n_repeats=1,
                                         random_state=RNG_SEED + rep).split(np.zeros(n), Y[:, j]))
            for j in range(L)
        ]
        for k in range(splits):
            rows_tr, rows_te = [], []
            for j in range(L):
                tr, te = splitters[j][k]
                rows_tr += [(i, j) for i in tr]
                rows_te += [(i, j) for i in te]

            def design(rows):
                idx = np.array([r[0] for r in rows]); lab = np.array([r[1] for r in rows])
                onehot = np.zeros((len(idx), L)); onehot[np.arange(len(idx)), lab] = 1.0
                lt, li = logit(T[idx, lab]), logit(I[idx, lab])
                if mode == "image_per_label":
                    cols = [lt.reshape(-1, 1), onehot * li[:, None], onehot]
                else:
                    cols = [onehot * lt[:, None], onehot * li[:, None], onehot]
                return np.hstack(cols), idx, lab

            X_tr, idx_tr, lab_tr = design(rows_tr)
            X_te, idx_te, lab_te = design(rows_te)
            y_tr = Y[idx_tr, lab_tr]
            model = LogisticRegression(C=C, solver="lbfgs", fit_intercept=False, max_iter=4000)
            model.fit(X_tr, y_tr)
            insample.append(bce(y_tr, model.predict_proba(X_tr)[:, 1]))
            if mode == "image_per_label":
                weights.append(model.coef_[0][1:1 + L])
            for (i_, j_), p in zip(zip(idx_te, lab_te), model.predict_proba(X_te)[:, 1]):
                oof[i_, j_, rep] = p
    pooled = np.nanmean(oof, axis=2)
    per_label = [
        {"label": LABELS[j], "bce": bce(Y[:, j], pooled[:, j]),
         "auc": auc(Y[:, j], pooled[:, j]), "bce_insample": float(np.mean(insample))}
        for j in range(L)
    ]
    return {
        "method": name,
        "bce": float(np.mean([r["bce"] for r in per_label])),
        "auc": float(np.nanmean([r["auc"] for r in per_label])),
        "bce_insample": float(np.mean(insample)),
        "per_label": per_label,
        "_pooled": pooled,
        "_image_weights": np.mean(weights, axis=0) if weights else None,
    }
