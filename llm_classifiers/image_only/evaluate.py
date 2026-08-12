#!/usr/bin/env python3
"""Score the cached image-only MedGemma runs.

Reads artifacts/raw/*.jsonl, averages repeated samples, and reports per-label and
macro discrimination against the labels in cohort_labels.csv.

Two things get as much weight as AUC here, because a small instruction-tuned VLM
under guided decoding fails in ways a headline AUC hides:

  degeneracy   how often a condition emits the same probability for every finding,
               or the same probability for a finding across every study. A constant
               score has no discrimination whatever its AUC rounds to.
  yield        how many requests parsed at all. A condition that answers 60% of the
               time is not comparable to one that answers always.

Confidence intervals and the contrasts against the reference condition use paired
case bootstrap over studies.

    python llm_classifiers/image_only/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from dicom_io import LABELS

REFERENCE_CONDITION = "ref_joint_definitions"
SCREEN_FILENAME = "screen.json"
N_BOOTSTRAP = 5000
RNG_SEED = 20260811


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records(raw_dir: Path, uids: set[str]) -> pd.DataFrame:
    """Load cached responses for the studies in the current cohort.

    Selection is by study, not by the cohort hash stamped on the record. A response
    describes one study under one image spec and stays valid however many other
    studies were extracted when it was produced, so a screening pilot's requests are
    scored alongside the rest instead of being thrown away.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["uid"] not in uids:
                    continue
                usage = record.get("usage") or {}
                stage_one = record.get("stage_one_usage") or {}
                rows.append(
                    {
                        "condition": record["condition"],
                        "cohort": record.get("cohort"),
                        "image_spec": record.get("image_spec"),
                        "uid": record["uid"],
                        "label": record["label"],
                        "repeat": record["repeat"],
                        "ok": bool(record.get("ok")),
                        "elapsed_seconds": record.get("elapsed_seconds"),
                        "finish_reason": record.get("finish_reason"),
                        "had_thinking": record.get("had_thinking"),
                        "n_images": record.get("n_images"),
                        "prompt_tokens": (usage.get("prompt_tokens") or 0)
                        + (stage_one.get("prompt_tokens") or 0),
                        "completion_tokens": (usage.get("completion_tokens") or 0)
                        + (stage_one.get("completion_tokens") or 0),
                        "result": record.get("result") or {},
                        "error": record.get("error"),
                    }
                )
    if not rows:
        raise SystemExit(f"no cached records found under {raw_dir}")
    frame = pd.DataFrame(rows)

    # Failed cells are retried on every run and appended again, so the file can hold
    # several attempts at one cell. Keep only the latest attempt per cell, otherwise
    # historical failures — including ones a later parser fix resolved — would drag
    # the reported yield down forever.
    return frame.drop_duplicates(
        subset=["condition", "uid", "label", "repeat"], keep="last"
    ).reset_index(drop=True)


def to_predictions(records: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition, uid, label): the mean probability across repeats."""
    accumulator: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in records[records["ok"]].itertuples(index=False):
        for label, probability in row.result.items():
            if label in LABELS and probability is not None:
                accumulator[(row.condition, row.uid, label)].append(float(probability))

    return pd.DataFrame(
        [
            {
                "condition": condition,
                "uid": uid,
                "label": label,
                "probability": float(np.mean(values)),
                "probability_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_samples": len(values),
            }
            for (condition, uid, label), values in accumulator.items()
        ]
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def label_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """AUC, or NaN when the label is single-class or the score is constant.

    A constant score is reported as undefined rather than as sklearn's 0.5, which
    would read as "chance performance" when the condition in fact produced no
    ranking at all — the distinction that matters most for degenerate guided output.

    Computed as the tie-corrected Mann-Whitney statistic; this is called millions of
    times inside the bootstrap, where `roc_auc_score`'s validation dominates runtime.
    """
    # A missing score means this study produced no prediction for this label, so it is
    # dropped from the label rather than ranked as if it had scored lowest.
    valid = ~np.isnan(score)
    truth, score = truth[valid], score[valid]
    if len(np.unique(truth)) < 2 or len(np.unique(score)) < 2:
        return float("nan")
    positive = truth > 0
    n_pos = int(positive.sum())
    n_neg = int(len(truth) - n_pos)
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    """Average precision over the studies this condition actually scored.

    A condition run only on a screening pilot has no prediction for the rest of the
    cohort. Those studies are dropped rather than imputed, matching `label_auc`;
    sklearn would otherwise reject the NaNs outright.
    """
    valid = ~np.isnan(score)
    truth, score = truth[valid], score[valid]
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        return float("nan")
    return float(average_precision_score(truth, score))


def score_matrix(
    predictions: pd.DataFrame, labels_df: pd.DataFrame, condition: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aligned (truth, score) matrices of shape (n_studies, n_labels)."""
    wide = (
        predictions[predictions["condition"] == condition]
        .pivot(index="uid", columns="label", values="probability")
        .reindex(index=labels_df["StudyInstanceUID"], columns=LABELS)
    )
    truth = labels_df[LABELS].to_numpy(dtype=float)
    return truth, wide.to_numpy(dtype=float), list(labels_df["StudyInstanceUID"])


def macro_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Mean AUC over the labels that are defined, all twelve computed at once.

    The bootstrap evaluates this millions of times, so the twelve labels are ranked
    in one `rankdata` call over the whole matrix rather than one call each.
    """
    valid = ~np.isnan(score)
    # Missing scores sort below every real probability, so after ranking they occupy
    # the lowest `n_missing` ranks and subtracting that count restores the ranking the
    # valid studies would have had on their own.
    ranks = rankdata(np.where(valid, score, -np.inf), axis=0) - (~valid).sum(axis=0)

    positive = (truth > 0) & valid
    negative = (truth <= 0) & valid
    n_pos = positive.sum(axis=0)
    n_neg = negative.sum(axis=0)

    # A column with no valid score at all makes nanmax/nanmin warn; such columns are
    # undefined anyway and are excluded by the n_pos/n_neg tests below.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        masked = np.where(valid, score, np.nan)
        spread = np.nanmax(masked, axis=0) > np.nanmin(masked, axis=0)
    defined = (n_pos > 0) & (n_neg > 0) & spread
    if not defined.any():
        return float("nan")

    rank_sum = np.where(positive, ranks, 0.0).sum(axis=0)
    aucs = (rank_sum - n_pos * (n_pos + 1) / 2.0) / np.maximum(n_pos * n_neg, 1)
    return float(aucs[defined].mean())


def bootstrap_macro(
    truth: np.ndarray, score: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = truth.shape[0]
    out = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        draw = rng.integers(0, n, n)
        out[index] = macro_auc(truth[draw], score[draw])
    return out


def paired_bootstrap(
    truth: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Difference in macro AUC (a - b) with a paired case-bootstrap CI and p-value."""
    n = truth.shape[0]
    diffs = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        draw = rng.integers(0, n, n)
        diffs[index] = macro_auc(truth[draw], score_a[draw]) - macro_auc(
            truth[draw], score_b[draw]
        )
    diffs = diffs[~np.isnan(diffs)]
    observed = macro_auc(truth, score_a) - macro_auc(truth, score_b)
    if diffs.size == 0:
        # Every resample was undefined, which happens on a cohort so small that
        # bootstrap draws routinely contain a single class for every label.
        return observed, float("nan"), float("nan"), float("nan")
    low, high = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p-value from the proportion of resamples crossing zero.
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    return observed, float(low), float(high), float(min(1.0, 2 * tail))


def screen_conditions(macro: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    """Decide which conditions are worth spending the full run's GPU time on.

    Two kinds of rule, with quite different reliability:

    Structural rules — parse yield and degeneracy — are near-certain even on a handful
    of studies. A condition that answers a constant vector, or fails to parse half the
    time, has no discriminative power, and no amount of extra data will change that.

    The accuracy rule is the one to treat with care. On ~15 studies the bootstrap
    interval on macro AUC is roughly +/-0.12, so a genuinely useful condition can land
    below the threshold by luck. `--screen-keep-if-ci-clears` softens this by keeping a
    condition whose upper confidence bound clears the threshold, at the cost of
    admitting more conditions to the full run.
    """
    passed: list[str] = []
    rejected: dict[str, str] = {}

    for row in macro.itertuples(index=False):
        auc = row.macro_auc
        reasons = []
        if row.request_yield < args.screen_min_yield:
            reasons.append(f"yield {row.request_yield:.2f} < {args.screen_min_yield}")
        flat = getattr(row, "flat_within_study_fraction", float("nan"))
        if not np.isnan(flat) and flat > args.screen_max_flat:
            reasons.append(f"flat {flat:.2f} > {args.screen_max_flat}")
        constant = getattr(row, "constant_labels", 0)
        if constant > args.screen_max_constant_labels:
            reasons.append(f"{constant} constant labels > {args.screen_max_constant_labels}")
        if np.isnan(auc):
            reasons.append("macro AUC undefined (no ranking produced)")
        else:
            bound = row.ci_high if args.screen_keep_if_ci_clears else auc
            label = "CI upper" if args.screen_keep_if_ci_clears else "AUC"
            if np.isnan(bound) or bound < args.screen_min_auc:
                reasons.append(f"{label} {bound:.3f} < {args.screen_min_auc}")

        if reasons:
            rejected[row.condition] = "; ".join(reasons)
        else:
            passed.append(row.condition)

    return {
        "passed": passed,
        "rejected": rejected,
        "criteria": {
            "min_macro_auc": args.screen_min_auc,
            "min_request_yield": args.screen_min_yield,
            "max_flat_within_study_fraction": args.screen_max_flat,
            "max_constant_labels": args.screen_max_constant_labels,
            "keep_if_ci_clears": bool(args.screen_keep_if_ci_clears),
        },
    }


def degeneracy_report(
    predictions: pd.DataFrame, condition: str, n_studies: int
) -> dict[str, Any]:
    """How much of the output is actually varying."""
    subset = predictions[predictions["condition"] == condition]
    if subset.empty:
        return {}
    wide = subset.pivot(index="uid", columns="label", values="probability").reindex(
        columns=LABELS
    )

    across_studies = wide.nunique(dropna=True)
    within_study = wide.nunique(axis=1, dropna=True)
    return {
        "flat_within_study_fraction": float((within_study <= 1).mean()),
        "constant_labels": int((across_studies <= 1).sum()),
        "mean_unique_values_per_label": float(across_studies.mean()),
        "mean_probability": float(np.nanmean(wide.to_numpy(dtype=float))),
        "study_coverage": float(len(wide) / n_studies),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--out-dir", default=str(here / "artifacts"))
    parser.add_argument(
        "--cohort",
        default=None,
        help="cohort hash to score; defaults to the one in run_manifest.json",
    )
    parser.add_argument("--reference", default=REFERENCE_CONDITION)
    parser.add_argument(
        "--write-screen",
        action="store_true",
        help=f"write {SCREEN_FILENAME}, the condition list used by --sweep sane",
    )
    parser.add_argument("--screen-min-auc", type=float, default=0.6)
    parser.add_argument("--screen-min-yield", type=float, default=0.9)
    parser.add_argument("--screen-max-flat", type=float, default=0.5)
    parser.add_argument("--screen-max-constant-labels", type=int, default=6)
    parser.add_argument(
        "--screen-keep-if-ci-clears",
        action="store_true",
        help="judge on the bootstrap upper bound rather than the point estimate, so a "
        "small pilot discards fewer conditions by chance",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "run_manifest.json"
    cohort = args.cohort
    if cohort is None and manifest_path.exists():
        cohort = json.loads(manifest_path.read_text())["cohort"]

    labels_df = pd.read_csv(out_dir / "cohort_labels.csv")
    records = load_records(out_dir / "raw", set(labels_df["StudyInstanceUID"]))

    predictions = to_predictions(records)
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    conditions = sorted(predictions["condition"].unique())

    # A condition whose every request failed to parse produces no prediction rows and
    # would otherwise vanish from the report, reading as if it had never been run.
    silent = sorted(set(records["condition"].unique()) - set(conditions))
    if silent:
        print(f"conditions with no parseable output at all: {', '.join(silent)}")
    # The pre-specified reference can itself be degenerate, in which case every
    # contrast against it is undefined. Fall back to the strongest condition so the
    # table stays informative, and label the fallback clearly: a baseline chosen after
    # seeing the scores is not a pre-specified contrast and must not be read as one.
    reference = args.reference
    if reference in conditions:
        truth_ref, score_ref, _ = score_matrix(predictions, labels_df, reference)
        if np.isnan(macro_auc(truth_ref, score_ref)):
            defined = [
                condition
                for condition in conditions
                if not np.isnan(
                    macro_auc(*score_matrix(predictions, labels_df, condition)[:2])
                )
            ]
            if defined:
                reference = max(
                    defined,
                    key=lambda name: macro_auc(
                        *score_matrix(predictions, labels_df, name)[:2]
                    ),
                )
                truth_ref, score_ref, _ = score_matrix(predictions, labels_df, reference)
                print(
                    f"note: {args.reference} produced no ranking, so contrasts use the "
                    f"best-scoring condition ({reference}) as a post-hoc baseline"
                )
    else:
        truth_ref = score_ref = None

    by_label_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    for condition in conditions:
        truth, score, _ = score_matrix(predictions, labels_df, condition)
        attempts = records[records["condition"] == condition]
        yield_rate = float(attempts["ok"].mean()) if len(attempts) else float("nan")

        for index, label in enumerate(LABELS):
            column = score[:, index]
            by_label_rows.append(
                {
                    "condition": condition,
                    "label": label,
                    "positives": int(truth[:, index].sum()),
                    "auc": label_auc(truth[:, index], column),
                    "average_precision": average_precision(truth[:, index], column),
                    "unique_scores": int(len(np.unique(column[~np.isnan(column)]))),
                    "mean_score": float(np.nanmean(column)),
                }
            )

        observed = macro_auc(truth, score)
        samples = bootstrap_macro(truth, score, np.random.default_rng(RNG_SEED))
        samples = samples[~np.isnan(samples)]
        row = {
            "condition": condition,
            "macro_auc": observed,
            "ci_low": float(np.percentile(samples, 2.5)) if len(samples) else float("nan"),
            "ci_high": float(np.percentile(samples, 97.5)) if len(samples) else float("nan"),
            # Conditions run only on a screening pilot cover fewer studies than ones
            # run on the whole cohort; their scores are not directly comparable.
            "studies_scored": int(attempts["uid"].nunique()),
            "request_yield": yield_rate,
            "mean_latency_seconds": float(attempts["elapsed_seconds"].mean(skipna=True)),
            "mean_prompt_tokens": float(attempts["prompt_tokens"].mean()),
            "mean_completion_tokens": float(attempts["completion_tokens"].mean()),
            "requests": int(len(attempts)),
            "thinking_fraction": float(
                attempts["had_thinking"].fillna(False).astype(bool).mean()
            ),
            "truncated_fraction": float(
                (attempts["finish_reason"] == "length").mean()
            ),
        }
        row.update(degeneracy_report(predictions, condition, len(labels_df)))
        macro_rows.append(row)

        if truth_ref is not None and condition != reference:
            difference, low, high, p_value = paired_bootstrap(
                truth_ref, score, score_ref, np.random.default_rng(RNG_SEED + 1)
            )
            contrast_rows.append(
                {
                    "condition": condition,
                    "reference": reference,
                    "macro_auc_difference": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                }
            )

    by_label = pd.DataFrame(by_label_rows)
    macro = pd.DataFrame(macro_rows).sort_values("macro_auc", ascending=False)
    contrasts = pd.DataFrame(contrast_rows).sort_values(
        "macro_auc_difference", ascending=False
    )

    by_label.to_csv(out_dir / "metrics_by_label.csv", index=False)
    macro.to_csv(out_dir / "metrics_macro.csv", index=False)
    if len(contrasts):
        contrasts.to_csv(out_dir / "contrasts_vs_reference.csv", index=False)

    n_pos = labels_df[LABELS].sum().astype(int)
    print(f"\ncohort {cohort}: {len(labels_df)} studies")
    print(f"positives per label: {n_pos.to_dict()}\n")

    if macro["studies_scored"].nunique() > 1:
        print(
            "warning: conditions cover different numbers of studies "
            f"({macro['studies_scored'].min()}-{macro['studies_scored'].max()}); "
            "their scores are not directly comparable. Conditions left behind by a "
            "screening pilot need the full run before they can be ranked against the rest.\n"
        )

    display = macro[
        [
            "condition",
            "macro_auc",
            "ci_low",
            "ci_high",
            "studies_scored",
            "request_yield",
            "flat_within_study_fraction",
            "constant_labels",
            "mean_latency_seconds",
        ]
    ].rename(
        columns={
            "macro_auc": "AUC",
            "studies_scored": "n",
            "request_yield": "yield",
            "flat_within_study_fraction": "flat",
            "constant_labels": "const",
            "mean_latency_seconds": "sec",
        }
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(display.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if len(contrasts):
        print(f"\npaired contrasts vs {reference}:")
        with pd.option_context("display.width", 200):
            print(
                contrasts.to_string(
                    index=False, float_format=lambda v: f"{v:.4f}"
                )
            )

    scored = macro[macro["macro_auc"].notna()]
    if scored.empty:
        raise SystemExit(
            "\nno condition produced a defined macro AUC on this cohort: every label is "
            "single-class or every score is constant. Wait for more studies to extract."
        )

    # Only conditions measured on the whole cohort are eligible to be called best. A
    # condition left behind by a screening pilot covers fewer studies, so its score is
    # a different quantity — and on a small subset it is also the noisier one, which
    # makes it exactly the sort of thing that wins a ranking it does not deserve.
    full = int(macro["studies_scored"].max())
    eligible = scored[scored["studies_scored"] == full]
    best = eligible.iloc[0]
    print(
        f"\nbest condition on all {full} studies: {best['condition']} "
        f"macro AUC {best['macro_auc']:.4f} "
        f"({best['ci_low']:.4f}-{best['ci_high']:.4f})"
    )
    partial = scored[scored["studies_scored"] < full]
    if len(partial) and partial.iloc[0]["macro_auc"] > best["macro_auc"]:
        row = partial.iloc[0]
        print(
            f"  ({row['condition']} scores higher at {row['macro_auc']:.4f}, but only on "
            f"{int(row['studies_scored'])} studies, so it is not comparable. Run it on the "
            "full cohort with --condition before believing it.)"
        )
    print(
        "\nSmall cohort: these intervals reflect case sampling only, not the fact that "
        "the conditions were chosen by looking at these same studies."
    )

    summary = {
        "cohort": cohort,
        "n_studies": int(len(labels_df)),
        "positives": n_pos.to_dict(),
        "best_condition": best["condition"],
        "best_macro_auc": float(best["macro_auc"]),
        "conditions_scored": conditions,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.write_screen:
        screen = screen_conditions(macro, args)
        screen.update(
            cohort=cohort,
            n_studies=int(len(labels_df)),
            positives=n_pos.to_dict(),
        )
        (out_dir / SCREEN_FILENAME).write_text(json.dumps(screen, indent=2))

        kept, dropped = len(screen["passed"]), len(screen["rejected"])
        print(f"\nscreen: {kept} of {kept + dropped} conditions kept -> {SCREEN_FILENAME}")
        for name, reason in sorted(screen["rejected"].items()):
            print(f"  drop {name:32s} {reason}")

        widths = (macro["ci_high"] - macro["ci_low"]).dropna()
        if len(widths):
            print(
                f"\nThe AUC rule is the unreliable half of this screen: on {len(labels_df)} "
                f"studies the 95% intervals are {widths.median():.2f} wide, so a condition "
                "can fall under the threshold by chance. Re-run with "
                "--screen-keep-if-ci-clears to judge on the upper bound instead, or raise "
                "--pilot-studies. Structural rejections (constant output, low yield) are "
                "safe at any cohort size."
            )
        print("\nRun the survivors on the full cohort with:  --sweep sane")


if __name__ == "__main__":
    main()
