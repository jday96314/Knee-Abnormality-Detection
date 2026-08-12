#!/usr/bin/env python3
"""Evaluate raw LLM predictions, ensembles, and leakage-safe OOF stackers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_llm_experiments import EXPERIMENTS, LABELS


SEED = 20260811
STATUS_SCORE = {
    "absent": 0.02,
    "not_mentioned": 0.15,
    "conflicting": 0.50,
    "suspected": 0.70,
    "present": 0.95,
}


def aucs(y: pd.DataFrame, predictions: pd.DataFrame, method: str) -> list[dict]:
    rows = []
    for label in LABELS:
        score = predictions[label].astype(float).to_numpy()
        rows.append(
            {
                "method": method,
                "label": label,
                "auc": roc_auc_score(y[label], score),
                "positives": int(y[label].sum()),
            }
        )
    rows.append(
        {
            "method": method,
            "label": "MACRO",
            "auc": float(np.mean([r["auc"] for r in rows])),
            "positives": int(y.to_numpy().sum()),
        }
    )
    return rows


def percentile_rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda col: rankdata(col, method="average") / len(col), axis=0)


def make_splits(y: np.ndarray, repeats: int = 20) -> list[tuple[np.ndarray, np.ndarray]]:
    # Five folds are possible for every target (minimum positives = 9).
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=repeats, random_state=SEED
    )
    return list(splitter.split(np.zeros(len(y)), y))


def repeated_oof_predict(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    estimator_factory: Callable[[], object],
    repeats: int = 20,
) -> np.ndarray:
    total = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    for train, test in make_splits(y, repeats):
        estimator = estimator_factory()
        estimator.fit(X.iloc[train] if hasattr(X, "iloc") else X[train], y[train])
        Xt = X.iloc[test] if hasattr(X, "iloc") else X[test]
        total[test] += estimator.predict_proba(Xt)[:, 1]
        counts[test] += 1
    if not np.all(counts == repeats):
        raise AssertionError((counts.min(), counts.max(), repeats))
    return total / counts


def condition_features(features: pd.DataFrame, label: str) -> pd.DataFrame:
    columns = []
    for experiment in EXPERIMENTS:
        columns.extend(
            [
                f"{experiment}__{label}__probability",
                f"{experiment}__{label}__confidence",
                f"{experiment}__{label}__probability_std",
                f"{experiment}__{label}__confidence_std",
                f"{experiment}__{label}__status_agreement",
                f"{experiment}__{label}__status",
            ]
        )
    return features[columns]


def all_condition_features(features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in features.columns
        if c.endswith("__probability")
        or c.endswith("__confidence")
        or c.endswith("__probability_std")
        or c.endswith("__confidence_std")
        or c.endswith("__status_agreement")
        or c.endswith("__status")
    ]
    return features[cols]


def stacker_factory(X: pd.DataFrame) -> Callable[[], Pipeline]:
    categorical = [c for c in X.columns if c.endswith("__status")]
    numeric = [c for c in X.columns if c not in categorical]

    def make() -> Pipeline:
        pre = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric),
                (
                    "status",
                    OneHotEncoder(handle_unknown="ignore", drop=None),
                    categorical,
                ),
            ]
        )
        return Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    LogisticRegression(
                        C=0.10,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

    return make


def text_factory() -> Pipeline:
    # Stateless hashing keeps every fold leakage-free without repeatedly learning and
    # sorting a large multilingual vocabulary from only ~46 training reports.
    vectorizer = FeatureUnion(
        [
            (
                "word",
                HashingVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    n_features=2**14,
                    alternate_sign=False,
                    norm="l2",
                ),
            ),
            (
                "char",
                HashingVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(3, 5),
                    n_features=2**14,
                    alternate_sign=False,
                    norm="l2",
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def bootstrap_macro_values(
    y: pd.DataFrame, pred: pd.DataFrame, counts: np.ndarray
) -> np.ndarray:
    """Vectorized case-bootstrap macro AUC, correctly retaining duplicate cases."""
    per_label = []
    for label in LABELS:
        yy = y[label].to_numpy().astype(bool)
        score = pred[label].to_numpy()
        # contribution[i,j] is the Mann-Whitney contribution of positive i and
        # negative j. Multiplicity weights implement ordinary case resampling.
        contribution = (
            yy[:, None]
            & ~yy[None, :]
        ) * (
            (score[:, None] > score[None, :])
            + 0.5 * (score[:, None] == score[None, :])
        )
        numerator = np.sum((counts @ contribution) * counts, axis=1)
        n_pos = counts[:, yy].sum(axis=1)
        n_neg = counts[:, ~yy].sum(axis=1)
        per_label.append(numerator / (n_pos * n_neg))
    values = np.column_stack(per_label)
    return values[np.isfinite(values).all(axis=1)].mean(axis=1)


def bootstrap_macro_ci(
    y: pd.DataFrame, pred: pd.DataFrame, counts: np.ndarray
) -> tuple[float, float]:
    values = bootstrap_macro_values(y, pred, counts)
    return tuple(np.percentile(values, [2.5, 97.5]).tolist())


def macro_auc(y: pd.DataFrame, pred: pd.DataFrame, idx: np.ndarray | None = None) -> float:
    if idx is None:
        idx = np.arange(len(y))
    scores = []
    for label in LABELS:
        yy = y[label].to_numpy()[idx]
        if len(np.unique(yy)) < 2:
            return float("nan")
        scores.append(roc_auc_score(yy, pred[label].to_numpy()[idx]))
    return float(np.mean(scores))


def paired_bootstrap_test(
    y: pd.DataFrame,
    a: pd.DataFrame,
    b: pd.DataFrame,
    a_name: str,
    b_name: str,
    counts: np.ndarray,
) -> dict:
    """Paired case-resampling test of macro AUC(a) - macro AUC(b)."""
    av = bootstrap_macro_values(y, a, counts)
    bv = bootstrap_macro_values(y, b, counts)
    diffs = av - bv
    # A two-sided bootstrap sign test is intentionally conservative at this sample size.
    p_value = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return {
        "method_a": a_name,
        "method_b": b_name,
        "observed_auc_difference": macro_auc(y, a) - macro_auc(y, b),
        "bootstrap_ci_low": float(np.percentile(diffs, 2.5)),
        "bootstrap_ci_high": float(np.percentile(diffs, 97.5)),
        "two_sided_p_value": float(p_value),
        "valid_bootstrap_samples": len(diffs),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    artifacts = here / "artifacts"
    features = pd.read_csv(artifacts / "llm_features.csv")
    source = pd.read_csv(root / "data/from_host/train.csv")
    reports = source.loc[features["source_row"].astype(int), "Report"].reset_index(drop=True)
    y = features[LABELS].astype(int)
    all_metrics: list[dict] = []
    prediction_sets: dict[str, pd.DataFrame] = {}

    repeat_counts: dict[str, int] = {}

    for experiment in EXPERIMENTS:
        repeats = 0
        while f"{experiment}__ACL__probability_r{repeats}" in features.columns:
            repeats += 1
        if repeats < 2:
            raise ValueError(f"Expected at least two stochastic repeats for {experiment}")
        repeat_counts[experiment] = repeats
        prob = pd.DataFrame(
            {
                label: features[f"{experiment}__{label}__probability"]
                for label in LABELS
            }
        )
        status = pd.DataFrame(
            {
                label: features[f"{experiment}__{label}__status"].map(STATUS_SCORE)
                for label in LABELS
            }
        )
        prediction_sets[experiment] = prob
        prediction_sets[experiment + "_status_only"] = status
        all_metrics += aucs(y, prob, experiment)
        all_metrics += aucs(y, status, experiment + "_status_only")
        for repeat in range(repeats):
            sampled = pd.DataFrame(
                {
                    label: features[f"{experiment}__{label}__probability_r{repeat}"]
                    for label in LABELS
                }
            )
            sampled_name = f"{experiment}_sample_{repeat}"
            prediction_sets[sampled_name] = sampled
            all_metrics += aucs(y, sampled, sampled_name)
        if repeats > 3:
            first_three = sum(
                pd.DataFrame(
                    {
                        label: features[
                            f"{experiment}__{label}__probability_r{repeat}"
                        ]
                        for label in LABELS
                    }
                )
                for repeat in range(3)
            ) / 3
            first_three_name = f"{experiment}_first3_average"
            prediction_sets[first_three_name] = first_three
            all_metrics += aucs(y, first_three, first_three_name)

    # Scale-robust ensembles of independently prompted views.
    for name, members in {
        "rank_ensemble_all": list(EXPERIMENTS),
        "rank_ensemble_no_reasoning": [
            "joint_extract",
            "joint_latent",
            "individual_latent",
        ],
        "rank_ensemble_joint": [
            "joint_extract",
            "joint_latent",
            "two_stage_reasoning",
        ],
    }.items():
        pred = sum(percentile_rank_frame(prediction_sets[m]) for m in members) / len(members)
        prediction_sets[name] = pred
        all_metrics += aucs(y, pred, name)

    # Repeated out-of-fold learned models. Every prediction for a row comes from models
    # that did not train on that row. Fixed regularization avoids meta-overfitting here.
    per_condition_stack = pd.DataFrame(index=features.index)
    cross_condition_stack = pd.DataFrame(index=features.index)
    text_baseline = pd.DataFrame(index=features.index)
    all_X = all_condition_features(features)
    for label in LABELS:
        yy = y[label].to_numpy()
        Xi = condition_features(features, label)
        per_condition_stack[label] = repeated_oof_predict(
            Xi, yy, stacker_factory(Xi), repeats=10
        )
        cross_condition_stack[label] = repeated_oof_predict(
            all_X, yy, stacker_factory(all_X), repeats=10
        )
        text_baseline[label] = repeated_oof_predict(
            reports.to_numpy(), yy, text_factory, repeats=10
        )
        print(f"OOF complete: {label}", flush=True)

    for name, pred in {
        "oof_per_condition_stack": per_condition_stack,
        "oof_cross_condition_stack": cross_condition_stack,
        "oof_hashed_text": text_baseline,
    }.items():
        prediction_sets[name] = pred
        all_metrics += aucs(y, pred, name)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(artifacts / "metrics_by_label.csv", index=False)
    macro = metrics.loc[metrics.label == "MACRO"].copy().sort_values("auc", ascending=False)
    rng = np.random.default_rng(SEED + 1)
    bootstrap_counts = rng.multinomial(
        len(y), np.repeat(1 / len(y), len(y)), size=5000
    )
    cis = []
    for method in macro.method:
        low, high = bootstrap_macro_ci(y, prediction_sets[method], bootstrap_counts)
        cis.append((method, low, high))
    ci_frame = pd.DataFrame(cis, columns=["method", "bootstrap_ci_low", "bootstrap_ci_high"])
    macro = macro.merge(ci_frame, on="method")
    macro.to_csv(artifacts / "metrics_macro.csv", index=False)

    comparisons = [
        ("joint_latent", "joint_extract"),
        ("joint_latent", "joint_latent_first3_average"),
        ("joint_latent_first3_average", "joint_extract"),
        ("individual_latent", "joint_latent"),
        ("individual_latent", "joint_latent_first3_average"),
        ("two_stage_reasoning", "joint_latent"),
        ("two_stage_reasoning", "joint_latent_first3_average"),
        ("rank_ensemble_all", "joint_latent"),
        ("oof_per_condition_stack", "rank_ensemble_all"),
        ("oof_cross_condition_stack", "oof_per_condition_stack"),
    ]
    tests = pd.DataFrame(
        [
            paired_bootstrap_test(
                y, prediction_sets[a], prediction_sets[b], a, b, bootstrap_counts
            )
            for a, b in comparisons
        ]
    )
    tests.to_csv(artifacts / "paired_bootstrap_tests.csv", index=False)

    sampling_rows = []
    macro_lookup = metrics.loc[metrics.label == "MACRO"].set_index("method")["auc"]
    for experiment in EXPERIMENTS:
        repeats = repeat_counts[experiment]
        values = [macro_lookup[f"{experiment}_sample_{r}"] for r in range(repeats)]
        sampling_rows.append(
            {
                "experiment": experiment,
                "samples": repeats,
                "mean_sample_macro_auc": float(np.mean(values)),
                "sample_macro_auc_std": float(np.std(values, ddof=1)),
                "min_sample_macro_auc": float(np.min(values)),
                "max_sample_macro_auc": float(np.max(values)),
                "averaged_prediction_macro_auc": float(macro_lookup[experiment]),
            }
        )
    pd.DataFrame(sampling_rows).to_csv(
        artifacts / "sampling_variability.csv", index=False
    )

    prediction_columns = {
        f"{method}__{label}": frame[label].to_numpy()
        for method, frame in prediction_sets.items()
        for label in LABELS
    }
    wide = pd.concat(
        [
            features[["source_row", "StudyInstanceUID", *LABELS]].reset_index(
                drop=True
            ),
            pd.DataFrame(prediction_columns),
        ],
        axis=1,
    )
    wide.to_csv(artifacts / "all_predictions.csv", index=False)

    result = {
        "n_labeled": len(features),
        "n_labels": len(LABELS),
        "llm_sampling_repeats": repeat_counts,
        "cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, seed=20260811)",
        "best_method": macro.iloc[0].to_dict(),
        "ranking": macro.to_dict(orient="records"),
        "paired_bootstrap_tests": tests.to_dict(orient="records"),
    }
    (artifacts / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(macro.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
