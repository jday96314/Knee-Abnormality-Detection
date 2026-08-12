#!/usr/bin/env python3
"""Evaluate prompt, sampling-temperature, and rerun-count refinements."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from evaluate import bootstrap_macro_values
from refine_raw_ensemble import REFINEMENTS, load_cache
from run_llm_experiments import LABELS, load_cache as load_main_cache, load_labeled


SEED = 20260812


def prediction_frame(
    data: pd.DataFrame, records: dict, repeats: list[int], main_format: bool = False
) -> pd.DataFrame:
    values = {label: [] for label in LABELS}
    for row in data.itertuples(index=False):
        for label in LABELS:
            sampled = []
            for repeat in repeats:
                if main_format:
                    item = records[(row.StudyInstanceUID, "__joint__", repeat)]
                    sampled.append(
                        float(item["result"]["predictions"][label]["probability"])
                    )
                else:
                    sampled.append(
                        float(records[(row.StudyInstanceUID, repeat)]["probabilities"][label])
                    )
            values[label].append(float(np.mean(sampled)))
    return pd.DataFrame(values)


def sample_arrays(
    data: pd.DataFrame, records: dict, repeats: list[int], main_format: bool = False
) -> dict[str, np.ndarray]:
    arrays = {label: [] for label in LABELS}
    for row in data.itertuples(index=False):
        for label in LABELS:
            sampled = []
            for repeat in repeats:
                if main_format:
                    item = records[(row.StudyInstanceUID, "__joint__", repeat)]
                    sampled.append(
                        float(item["result"]["predictions"][label]["probability"])
                    )
                else:
                    sampled.append(
                        float(records[(row.StudyInstanceUID, repeat)]["probabilities"][label])
                    )
            arrays[label].append(sampled)
    return {label: np.asarray(value) for label, value in arrays.items()}


def aggregate_frame(arrays: dict[str, np.ndarray], method: str) -> pd.DataFrame:
    values = {}
    for label, matrix in arrays.items():
        if method == "mean":
            score = matrix.mean(axis=1)
        elif method == "median":
            score = np.median(matrix, axis=1)
        elif method == "trimmed_mean":
            ordered = np.sort(matrix, axis=1)
            score = ordered[:, 1:-1].mean(axis=1)
        elif method == "logit_mean":
            score = expit(logit(np.clip(matrix, 1e-4, 1 - 1e-4)).mean(axis=1))
        elif method == "rank_mean":
            ranked = np.column_stack(
                [rankdata(matrix[:, col], method="average") for col in range(matrix.shape[1])]
            )
            score = ranked.mean(axis=1)
        else:
            raise ValueError(method)
        values[label] = score
    return pd.DataFrame(values)


def macro_auc(y: pd.DataFrame, prediction: pd.DataFrame) -> float:
    return float(
        np.mean(
            [roc_auc_score(y[label], prediction[label]) for label in LABELS]
        )
    )


def complete_repeats(data, records, main_format: bool) -> int:
    repeat = 0
    while True:
        if main_format:
            keys = [
                (uid, "__joint__", repeat) for uid in data.StudyInstanceUID
            ]
        else:
            keys = [(uid, repeat) for uid in data.StudyInstanceUID]
        if not all(key in records for key in keys):
            return repeat
        repeat += 1


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    artifacts = here / "artifacts/refinement"
    artifacts.mkdir(parents=True, exist_ok=True)
    data = load_labeled(root / "data/from_host/train.csv")
    y = data[LABELS].astype(int).reset_index(drop=True)

    sources = {}
    main_records = load_main_cache(here / "artifacts/raw/joint_latent.jsonl")
    baseline_repeats = complete_repeats(data, main_records, main_format=True)
    sources["baseline_t06"] = (main_records, baseline_repeats, True)
    for name in REFINEMENTS:
        records = load_cache(artifacts / "raw" / f"{name}.jsonl")
        repeats = complete_repeats(data, records, main_format=False)
        if repeats:
            sources[name] = (records, repeats, False)

    if set(REFINEMENTS) - set(sources):
        raise ValueError(f"Missing refinements: {sorted(set(REFINEMENTS) - set(sources))}")
    if baseline_repeats < 10:
        raise ValueError(f"Expected 10 baseline repeats, found {baseline_repeats}")

    metric_rows = []
    count_rows = []
    primary_predictions = {}
    per_label_rows = []
    for name, (records, repeats, main_format) in sources.items():
        primary_n = 5
        primary = prediction_frame(
            data, records, list(range(primary_n)), main_format=main_format
        )
        primary_predictions[name] = primary
        auc = macro_auc(y, primary)
        metric_rows.append(
            {
                "config": name,
                "samples": primary_n,
                "macro_auc": auc,
                "available_repeats": repeats,
            }
        )
        for label in LABELS:
            per_label_rows.append(
                {
                    "config": name,
                    "label": label,
                    "auc": roc_auc_score(y[label], primary[label]),
                }
            )

        max_count = repeats if name == "baseline_t06" else min(repeats, 5)
        for count in range(1, max_count + 1):
            subset_aucs = []
            for subset in itertools.combinations(range(repeats), count):
                pred = prediction_frame(
                    data, records, list(subset), main_format=main_format
                )
                subset_aucs.append(macro_auc(y, pred))
            prefix = prediction_frame(
                data, records, list(range(count)), main_format=main_format
            )
            count_rows.append(
                {
                    "config": name,
                    "samples": count,
                    "subsets": len(subset_aucs),
                    "mean_subset_macro_auc": float(np.mean(subset_aucs)),
                    "std_subset_macro_auc": float(np.std(subset_aucs)),
                    "min_subset_macro_auc": float(np.min(subset_aucs)),
                    "max_subset_macro_auc": float(np.max(subset_aucs)),
                    "prefix_macro_auc": macro_auc(y, prefix),
                }
            )

    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(
        len(y), np.repeat(1 / len(y), len(y)), size=5000
    )
    baseline = primary_predictions["baseline_t06"]
    baseline_boot = bootstrap_macro_values(y, baseline, counts)
    comparison_rows = []
    for name, prediction in primary_predictions.items():
        values = bootstrap_macro_values(y, prediction, counts)
        low, high = np.percentile(values, [2.5, 97.5])
        metric = next(row for row in metric_rows if row["config"] == name)
        metric["bootstrap_ci_low"] = float(low)
        metric["bootstrap_ci_high"] = float(high)
        if name == "baseline_t06":
            continue
        diff = values - baseline_boot
        comparison_rows.append(
            {
                "config": name,
                "baseline": "baseline_t06",
                "auc_difference": macro_auc(y, prediction) - macro_auc(y, baseline),
                "bootstrap_ci_low": float(np.percentile(diff, 2.5)),
                "bootstrap_ci_high": float(np.percentile(diff, 97.5)),
                "two_sided_p_value": float(
                    min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
                ),
            }
        )

    # Aggregation rules for the 10 baseline samples.
    baseline_arrays = sample_arrays(
        data, main_records, list(range(baseline_repeats)), main_format=True
    )
    aggregation_rows = []
    aggregation_predictions = {}
    for method in ["mean", "median", "trimmed_mean", "logit_mean", "rank_mean"]:
        prediction = aggregate_frame(baseline_arrays, method)
        aggregation_predictions[method] = prediction
        values = bootstrap_macro_values(y, prediction, counts)
        aggregation_rows.append(
            {
                "aggregation": method,
                "samples": baseline_repeats,
                "macro_auc": macro_auc(y, prediction),
                "bootstrap_ci_low": float(np.percentile(values, 2.5)),
                "bootstrap_ci_high": float(np.percentile(values, 97.5)),
            }
        )

    # Fixed equal-weight mixtures test whether weaker prompts add useful diversity.
    mixture_rows = []
    config_names = list(primary_predictions)
    for left, right in itertools.combinations(config_names, 2):
        probability_mix = (primary_predictions[left] + primary_predictions[right]) / 2
        rank_mix = sum(
            frame.apply(
                lambda column: rankdata(column, method="average") / len(column), axis=0
            )
            for frame in [primary_predictions[left], primary_predictions[right]]
        ) / 2
        for kind, prediction in [
            ("probability_mean", probability_mix),
            ("rank_mean", rank_mix),
        ]:
            mixture_rows.append(
                {
                    "left": left,
                    "right": right,
                    "aggregation": kind,
                    "macro_auc": macro_auc(y, prediction),
                }
            )
    all_probability_mix = sum(primary_predictions.values()) / len(primary_predictions)
    all_rank_mix = sum(
        frame.apply(
            lambda column: rankdata(column, method="average") / len(column), axis=0
        )
        for frame in primary_predictions.values()
    ) / len(primary_predictions)
    mixture_rows.extend(
        [
            {
                "left": "all_configs",
                "right": "all_configs",
                "aggregation": "probability_mean",
                "macro_auc": macro_auc(y, all_probability_mix),
            },
            {
                "left": "all_configs",
                "right": "all_configs",
                "aggregation": "rank_mean",
                "macro_auc": macro_auc(y, all_rank_mix),
            },
        ]
    )

    # Extended diversity study after collecting 10 runs for both the baseline and
    # image-review framing. Compare prompt diversity with matched baseline-only budgets.
    image_records, image_repeats, _ = sources["image_reviewer_t06"]
    extended_summary = {}
    if image_repeats >= 10:
        image_arrays = sample_arrays(
            data, image_records, list(range(10)), main_format=False
        )
        diversity_rng = np.random.default_rng(SEED + 1)
        diversity_rows = []
        for each_count in range(1, 6):
            prompt_subsets = list(itertools.combinations(range(10), each_count))
            pairs = list(itertools.product(prompt_subsets, repeat=2))
            if len(pairs) > 2000:
                selected = diversity_rng.choice(len(pairs), size=2000, replace=False)
                pairs = [pairs[index] for index in selected]
            mixture_aucs = []
            for baseline_subset, image_subset in pairs:
                prediction = pd.DataFrame(
                    {
                        label: np.concatenate(
                            [
                                baseline_arrays[label][:, baseline_subset],
                                image_arrays[label][:, image_subset],
                            ],
                            axis=1,
                        ).mean(axis=1)
                        for label in LABELS
                    }
                )
                mixture_aucs.append(macro_auc(y, prediction))

            baseline_count = each_count * 2
            baseline_subset_aucs = []
            for subset in itertools.combinations(range(10), baseline_count):
                prediction = pd.DataFrame(
                    {
                        label: baseline_arrays[label][:, subset].mean(axis=1)
                        for label in LABELS
                    }
                )
                baseline_subset_aucs.append(macro_auc(y, prediction))
            diversity_rows.append(
                {
                    "total_requests": baseline_count,
                    "runs_per_prompt": each_count,
                    "mixture_subsets_evaluated": len(mixture_aucs),
                    "mixture_mean_macro_auc": float(np.mean(mixture_aucs)),
                    "mixture_std_macro_auc": float(np.std(mixture_aucs)),
                    "baseline_subsets_evaluated": len(baseline_subset_aucs),
                    "baseline_mean_macro_auc": float(
                        np.mean(baseline_subset_aucs)
                    ),
                    "baseline_std_macro_auc": float(np.std(baseline_subset_aucs)),
                    "mean_mixture_advantage": float(
                        np.mean(mixture_aucs) - np.mean(baseline_subset_aucs)
                    ),
                }
            )
        pd.DataFrame(diversity_rows).to_csv(
            artifacts / "prompt_diversity_by_budget.csv", index=False
        )

        fixed_candidates = {
            "baseline_mean_10": aggregate_frame(baseline_arrays, "mean"),
            "baseline_rank_10": aggregate_frame(baseline_arrays, "rank_mean"),
            "image_mean_10": aggregate_frame(image_arrays, "mean"),
            "image_rank_10": aggregate_frame(image_arrays, "rank_mean"),
        }
        combined_arrays = {
            label: np.concatenate(
                [baseline_arrays[label], image_arrays[label]], axis=1
            )
            for label in LABELS
        }
        fixed_candidates["mixed_mean_20"] = aggregate_frame(
            combined_arrays, "mean"
        )
        fixed_candidates["mixed_rank_20"] = aggregate_frame(
            combined_arrays, "rank_mean"
        )
        fixed_candidate_rows = []
        reference_values = bootstrap_macro_values(
            y, fixed_candidates["baseline_mean_10"], counts
        )
        for name, prediction in fixed_candidates.items():
            values = bootstrap_macro_values(y, prediction, counts)
            diff = values - reference_values
            fixed_candidate_rows.append(
                {
                    "candidate": name,
                    "macro_auc": macro_auc(y, prediction),
                    "difference_vs_baseline_mean_10": macro_auc(y, prediction)
                    - macro_auc(y, fixed_candidates["baseline_mean_10"]),
                    "difference_ci_low": float(np.percentile(diff, 2.5)),
                    "difference_ci_high": float(np.percentile(diff, 97.5)),
                    "two_sided_p_value": float(
                        min(
                            1.0,
                            2
                            * min(np.mean(diff <= 0), np.mean(diff >= 0)),
                        )
                    ),
                }
            )
        pd.DataFrame(fixed_candidate_rows).sort_values(
            "macro_auc", ascending=False
        ).to_csv(artifacts / "extended_prompt_ensembles.csv", index=False)

        candidate_predictions = {
            "baseline_mean_5": primary_predictions["baseline_t06"],
            "baseline_mean_10": fixed_candidates["baseline_mean_10"],
            "baseline_rank_10": fixed_candidates["baseline_rank_10"],
            "probability_only_mean_5": primary_predictions[
                "probability_only_t06"
            ],
            "mixed_mean_20": fixed_candidates["mixed_mean_20"],
        }
        candidate_output = data[
            ["source_row", "StudyInstanceUID", *LABELS]
        ].copy()
        candidate_columns = {
            f"{name}__{label}": frame[label].to_numpy()
            for name, frame in candidate_predictions.items()
            for label in LABELS
        }
        candidate_output = pd.concat(
            [candidate_output.reset_index(drop=True), pd.DataFrame(candidate_columns)],
            axis=1,
        )
        candidate_output.to_csv(
            artifacts / "candidate_predictions.csv", index=False
        )
        extended_frame = pd.DataFrame(fixed_candidate_rows).sort_values(
            "macro_auc", ascending=False
        )
        extended_summary = {
            "best_extended_candidate": extended_frame.iloc[0].to_dict(),
            "recommended_cost_performance": "baseline_mean_5",
            "recommended_max_stability": "baseline_mean_10",
            "batch_ranking_option": "baseline_rank_10",
        }

    metrics = pd.DataFrame(metric_rows).sort_values("macro_auc", ascending=False)
    metrics.to_csv(artifacts / "config_metrics.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(
        artifacts / "config_metrics_by_label.csv", index=False
    )
    pd.DataFrame(count_rows).to_csv(
        artifacts / "rerun_count_summary.csv", index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        artifacts / "config_paired_bootstrap.csv", index=False
    )
    pd.DataFrame(aggregation_rows).sort_values(
        "macro_auc", ascending=False
    ).to_csv(artifacts / "aggregation_methods.csv", index=False)
    pd.DataFrame(mixture_rows).sort_values("macro_auc", ascending=False).to_csv(
        artifacts / "prompt_mixtures.csv", index=False
    )

    output = data[["source_row", "StudyInstanceUID", *LABELS]].copy()
    for name, frame in primary_predictions.items():
        for label in LABELS:
            output[f"{name}__{label}"] = frame[label].to_numpy()
    output.to_csv(artifacts / "refinement_predictions.csv", index=False)

    runtime_rows = []
    for name, (records, repeats, _) in sources.items():
        values = list(records.values())
        runtime_rows.append(
            {
                "config": name,
                "requests": len(values),
                "available_repeats": repeats,
                "mean_request_seconds": float(
                    np.mean([item["elapsed_seconds"] for item in values])
                ),
                "mean_completion_tokens": float(
                    np.mean([item["usage"]["completion_tokens"] for item in values])
                ),
                "total_completion_tokens": int(
                    sum(item["usage"]["completion_tokens"] for item in values)
                ),
            }
        )
    pd.DataFrame(runtime_rows).to_csv(
        artifacts / "runtime_summary.csv", index=False
    )

    summary = {
        "primary_comparison_samples": 5,
        "best_config": metrics.iloc[0].to_dict(),
        "baseline_available_repeats": baseline_repeats,
        "configs": metrics.to_dict(orient="records"),
        "extended": extended_summary,
        "caveat": "Exploratory selection on the same 58 labeled cases.",
    }
    (artifacts / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
