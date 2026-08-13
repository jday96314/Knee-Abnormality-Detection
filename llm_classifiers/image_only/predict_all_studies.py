#!/usr/bin/env python3
"""Pseudo-label every training study from images alone, for distillation.

Runs the best-performing MedGemma configuration over all 4,407 training studies —
including the 4,349 with no ground-truth labels — and writes soft targets in the same
layout as `../text_only/predictions/text_only_predictions.csv`.

The intended use is pretraining a lightweight image-only model: the 58 gold-labelled
studies are far too few, but a teacher that scores 0.71 macro AUC from pixels can
label seventy-five times as many.

    python llm_classifiers/image_only/predict_all_studies.py --dry-run   # cost first
    python llm_classifiers/image_only/predict_all_studies.py --warm-cache

Configurations (`--config`), measured on the 58 labelled studies:

    best   0.709  rank/probability average of two TTA conditions   75 requests/study
    fast   0.659  single digit-logprob condition                   12 requests/study

`best` is about six times the cost of `fast` for +0.05 macro AUC. For distillation the
cheaper teacher may well be the better trade; `--dry-run` prints both estimates.

Two properties of these outputs matter for how they are consumed:

* They are **soft targets, not labels.** A teacher at 0.71 macro AUC is wrong often,
  and wrong in a structured way — near-chance on ACL and MCL, strong on effusion and
  fracture. Weighting the distillation loss per finding by the teacher's per-finding
  AUC (recorded in the sidecar JSON) is more defensible than treating all twelve alike.
* The probability scale is **compressed and uncalibrated**. The digit-expectation
  member concentrates in roughly 0.79-0.90, so absolute values carry little
  information and only the ordering is meaningful. `--calibrate rank` maps each
  finding's scores to their within-cohort percentile, which is usually the more useful
  target for a student trained with a ranking or soft-BCE objective.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from dicom_io import LABELS, set_order_cache
from run_medgemma_experiments import (
    CONDITIONS,
    Endpoint,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    Runner,
    required_specs,
    warm_cache,
)

CONFIGS = {
    # Both TTA members of the best measured ensemble (0.709 macro AUC).
    # Measured 0.7093 on the 58 labelled studies — statistically identical to the
    # 75-request-per-study pairing below (0.7092, p=0.97) for a tenth of the calls. The
    # expensive per-finding digit member adds nothing these two do not already supply,
    # which is only visible by testing ensembles against each other rather than
    # assuming the best singles make the best pair.
    "best": ["v3_two_stage_tta_slices3_t07", "v2_two_stage_t07_x5"],
    # The original pairing, kept for reference: same accuracy, ~9x the requests.
    "best_expensive": ["v3_digit_tta_slices_flip6", "v3_two_stage_tta_slices3_t07"],
    # The single cheapest condition worth using (0.659), ~6x less compute.
    "fast": ["v2_digit"],
}

# Per-finding AUC of the `best` ensemble on the 58 labelled studies. Written into the
# sidecar so a downstream trainer can weight or drop findings the teacher cannot do.
TEACHER_AUC = {
    "ACL": 0.578, "MCL": 0.544, "Medial Meniscus": 0.689, "Lateral Meniscus": 0.734,
    "Medial OA": 0.674, "Lateral OA": 0.682, "PF OA": 0.724, "Effusion": 0.883,
    "Synovitis": 0.762, "Baker's": 0.666, "Contusion": 0.757, "Fracture": 0.815,
}


def build_endpoints(args: argparse.Namespace) -> list["Endpoint"]:
    """Parse `--endpoint URL[,concurrency[,max_hours]]`, or fall back to `--base-url`.

    Concurrency is per endpoint because servers saturate at different points: measured
    here, one 4090 ceilings near 2.6 req/s while two 5090s reach 17 req/s and stop
    improving past concurrency 32. Pushing past a server's ceiling adds queueing delay,
    not throughput.

    `max_hours` lets a borrowed machine be handed back on schedule — it stops taking new
    work at the deadline and the remaining endpoints drain the queue.
    """
    specs = args.endpoint or [f"{args.base_url},{args.concurrency}"]
    endpoints: list[Endpoint] = []
    for spec in specs:
        parts = [p.strip() for p in spec.split(",")]
        concurrency = int(parts[1]) if len(parts) > 1 and parts[1] else args.concurrency
        hours = float(parts[2]) if len(parts) > 2 and parts[2] else None
        endpoints.append(
            Endpoint(
                client=AsyncOpenAI(base_url=parts[0], api_key=args.api_key, timeout=args.timeout),
                concurrency=concurrency,
                name=parts[0].split("//")[-1].split(":")[0],
                max_seconds=hours * 3600 if hours else None,
            )
        )
    return endpoints


def all_training_studies(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every study in train.csv that is fully extracted, labelled or not.

    `source_row` is the row index in train.csv, matching the text-only predictions file
    so the two can be joined directly.
    """
    train = pd.read_csv(data_dir / "train.csv")
    series = pd.read_csv(data_dir / "train_series.csv")
    root = data_dir / "train_series"

    expected = series.groupby("StudyInstanceUID")["SeriesInstanceUID"].apply(set)
    on_disk = {p.name for p in root.iterdir() if p.is_dir()}

    keep = []
    for row, uid in enumerate(train["StudyInstanceUID"]):
        if uid not in on_disk:
            continue
        folder = root / uid
        if {p.name for p in folder.iterdir() if p.is_dir()} != expected.get(uid, set()):
            continue
        keep.append(row)

    studies = train.iloc[keep].copy()
    studies.insert(0, "source_row", keep)
    missing = len(train) - len(studies)
    if missing:
        print(f"warning: {missing} of {len(train)} studies are not fully extracted", flush=True)
    return studies.reset_index(drop=True), series


def collect_predictions(raw_dir: Path, names: list[str]) -> pd.DataFrame:
    """Mean probability and across-view spread per (study, finding, condition).

    The spread is the standard deviation over a condition's repeats — for a TTA
    condition that is disagreement between different views of the same knee, which is
    a more useful uncertainty signal than sampling noise at fixed input.
    """
    rows: dict[tuple[str, str, str], list[float]] = {}
    for name in names:
        path = raw_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        latest: dict[tuple[str, str, int], dict] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                latest[(record["uid"], record["label"], record["repeat"])] = record
        for record in latest.values():
            if not record.get("ok"):
                continue
            for label, value in (record.get("result") or {}).items():
                if label in LABELS and value is not None:
                    rows.setdefault((name, record["uid"], label), []).append(float(value))

    return pd.DataFrame(
        [
            {
                "condition": name,
                "StudyInstanceUID": uid,
                "label": label,
                "probability": float(np.mean(values)),
                "spread": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
            for (name, uid, label), values in rows.items()
        ]
    )


def build_table(
    studies: pd.DataFrame, predictions: pd.DataFrame, names: list[str], calibrate: str
) -> pd.DataFrame:
    """Average the configured conditions into one row per study, wide by finding."""
    out = studies[["source_row", "StudyInstanceUID"]].copy()

    for label in LABELS:
        subset = predictions[predictions["label"] == label]
        wide = subset.pivot_table(
            index="StudyInstanceUID", columns="condition", values="probability"
        ).reindex(out["StudyInstanceUID"])
        spread = subset.pivot_table(
            index="StudyInstanceUID", columns="condition", values="spread"
        ).reindex(out["StudyInstanceUID"])

        present = [n for n in names if n in wide.columns]
        # Equal weight per condition; probability-averaging measured indistinguishable
        # from rank-averaging (+0.0002, p=0.95) and keeps the 0-1 scale.
        value = wide[present].mean(axis=1).to_numpy()
        # Within-view spread, plus disagreement between conditions when there are two.
        within = spread[present].mean(axis=1).to_numpy()
        between = wide[present].std(axis=1, ddof=1).to_numpy() if len(present) > 1 else 0.0
        combined = np.sqrt(np.nan_to_num(within) ** 2 + np.nan_to_num(between) ** 2)

        if calibrate == "rank":
            valid = ~np.isnan(value)
            ranked = np.full_like(value, np.nan, dtype=float)
            ranked[valid] = pd.Series(value[valid]).rank(pct=True).to_numpy()
            value = ranked

        out[label] = value
        out[f"{label}__sample_std"] = combined

    return out


async def run_pipelined(
    runner, names: list[str], uids: list[str], data_dir: Path, image_cache: Path,
    args: argparse.Namespace,
) -> None:
    """Render chunk N+1 while inferring on chunk N.

    Rendering is CPU/IO-bound locally and inference is GPU-bound remotely, so doing
    them in sequence wastes whichever resource is idle. Measured here they are of
    comparable size (~3.4 h of rendering against ~4.8 h of inference), so overlapping
    turns a sum into a maximum. The pipeline is one chunk deep, which is enough: the
    GPU never waits on the renderer as long as the renderer stays a chunk ahead.
    """
    from concurrent.futures import ProcessPoolExecutor

    from run_medgemma_experiments import _warm_init, _warm_one

    workers = args.warm_workers or max(1, (os.cpu_count() or 4) - 2)
    specs = list(required_specs(names).values())
    chunks = [uids[i : i + args.chunk] for i in range(0, len(uids), args.chunk)]
    loop = asyncio.get_running_loop()
    print(
        f"pipelining {len(chunks)} chunks of {args.chunk} studies "
        f"({len(specs)} specs, {workers} render workers)",
        flush=True,
    )

    pool = ProcessPoolExecutor(
        workers,
        initializer=_warm_init,
        initargs=(str(data_dir), str(data_dir / "train_series.csv"), str(image_cache)),
    )
    try:
        def submit(chunk: list[str]):
            return [
                loop.run_in_executor(pool, _warm_one, (uid, spec))
                for spec in specs
                for uid in chunk
            ]

        pending = submit(chunks[0])
        for index, chunk in enumerate(chunks):
            warm_started = time.perf_counter()
            await asyncio.gather(*pending)
            waited = time.perf_counter() - warm_started
            # Start the next chunk rendering before this one's requests go out.
            pending = submit(chunks[index + 1]) if index + 1 < len(chunks) else []
            print(
                f"[chunk {index + 1}/{len(chunks)}] warm ready (waited {waited:.0f}s)",
                flush=True,
            )
            for name in names:
                await runner.run_condition(CONDITIONS[name], chunk)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


async def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    names = CONFIGS[args.config] if not args.condition else list(args.condition)
    studies, series = all_training_studies(data_dir)
    if args.limit:
        studies = studies.head(args.limit)
    uids = studies["StudyInstanceUID"].tolist()

    # Rendered PNGs depend only on the DICOMs and the image spec, never on the model or
    # the server, so all runs share one cache.
    image_cache = (
        Path(args.image_cache_dir)
        if args.image_cache_dir
        else Path(__file__).resolve().parent / "artifacts" / "image_cache"
    )
    set_order_cache(image_cache / "_slice_order")

    per_study = sum(
        (12 if CONDITIONS[n].per_label else 1) * CONDITIONS[n].samples for n in names
    )
    print(f"config {args.config!r}: {names}")
    print(f"{len(uids)} studies x {per_study} requests/study = {len(uids) * per_study:,} requests")

    if args.dry_run:
        specs = required_specs(names)
        print(f"\ndistinct image specs to render: {len(specs)} x {len(uids)} studies")
        print(f"  rendering (12 workers, ~5s/study-render): {len(specs) * len(uids) * 5 / 12 / 3600:.1f} h")
        for name in names:
            condition = CONDITIONS[name]
            reqs = (12 if condition.per_label else 1) * condition.samples * len(uids)
            # Latency measured during the 58-study sweeps.
            latency = {"v3_digit_tta_slices_flip6": 5.8, "v3_two_stage_tta_slices3_t07": 31.6,
                       "v2_digit": 2.2}.get(name, 6.0)
            print(f"  {name:30s} {reqs:8,} req x {latency:5.1f}s = {reqs * latency / 3600:6.1f} GPU-h")
        total = sum(
            (12 if CONDITIONS[n].per_label else 1) * CONDITIONS[n].samples * len(uids)
            * {"v3_digit_tta_slices_flip6": 5.8, "v3_two_stage_tta_slices3_t07": 31.6,
               "v2_digit": 2.2}.get(n, 6.0)
            for n in names
        )
        for concurrency in (24, 48, 64):
            print(f"  wall-clock at concurrency {concurrency:3d}: {total / 3600 / concurrency:6.1f} h")
        return

    if args.warm_cache and not args.pipeline:
        workers = args.warm_workers or max(1, (os.cpu_count() or 4) - 2)
        warm_cache(data_dir, uids, names, workers, image_cache)

    endpoints = build_endpoints(args)
    print("endpoints:")
    for e in endpoints:
        budget = f", stops after {e.max_seconds/3600:.1f}h" if e.max_seconds else ""
        print(f"  {e.name}  concurrency {e.concurrency}{budget}")
    runner = Runner(
        client=endpoints[0].client, model=args.model, data_dir=data_dir, series=series,
        out_dir=out_dir, cohort=f"all{len(uids)}", concurrency=args.concurrency,
        max_retries=args.max_retries, labels=None, image_cache=image_cache,
        endpoints=endpoints,
    )
    started = time.perf_counter()
    if args.pipeline:
        await run_pipelined(runner, names, uids, data_dir, image_cache, args)
    else:
        for name in names:
            await runner.run_condition(CONDITIONS[name], uids)
    print(f"inference done in {(time.perf_counter() - started) / 3600:.2f} h", flush=True)

    write_outputs(out_dir, studies, names, args)


def write_outputs(out_dir: Path, studies: pd.DataFrame, names: list[str], args: argparse.Namespace) -> None:
    predictions = collect_predictions(out_dir / "raw", names)
    target = Path(args.predictions_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Both scalings are written from one run because they suit different students.
    # The raw teacher probability is heavily compressed (typically 0.80-0.88), so as an
    # absolute target it carries almost no signal; the percentile version spreads the
    # same ordering over [0,1] and is usually what a distillation loss wants.
    table = build_table(studies, predictions, names, "none")
    csv_path = target / args.filename
    table.to_csv(csv_path, index=False)
    ranked = build_table(studies, predictions, names, "rank")
    rank_path = target / (Path(args.filename).stem + "_rank.csv")
    ranked.to_csv(rank_path, index=False)
    if args.calibrate == "rank":
        table = ranked

    covered = int(table[LABELS].notna().all(axis=1).sum())
    print(f"\nwrote {csv_path}")
    print(f"wrote {rank_path}  (same ordering, percentile-scaled)")
    print(f"  {len(table)} rows, {covered} with a full set of predictions")
    if covered < len(table):
        print(f"  {len(table) - covered} studies incomplete; re-run to fill them in")

    sidecar = {
        "config": args.config,
        "conditions": names,
        "model": args.model,
        "calibrate": args.calibrate,
        "n_studies": int(len(table)),
        "n_complete": covered,
        "teacher_macro_auc_on_58_labelled": {"best": 0.709, "best_expensive": 0.709}.get(args.config, 0.659),
        "teacher_per_finding_auc": TEACHER_AUC,
        "note": (
            "Soft targets from an image-only teacher, not ground truth. Per-finding AUC "
            "varies from ~0.54 (MCL) to ~0.88 (Effusion); weight or filter accordingly. "
            "Raw probabilities are compressed and uncalibrated - prefer --calibrate rank "
            "if the student is trained on relative ordering."
        ),
    }
    (target / (Path(args.filename).stem + "_meta.json")).write_text(json.dumps(sidecar, indent=2))

    described = table[LABELS].describe().T[["mean", "std", "min", "50%", "max"]]
    print("\nper-finding distribution of the written targets:")
    print(described.to_string(float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(here.parents[1] / "data" / "from_host"))
    parser.add_argument("--out-dir", default=str(here / "artifacts_pseudolabel"))
    parser.add_argument("--predictions-dir", default=str(here / "predictions"))
    parser.add_argument("--filename", default="image_only_predictions.csv")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="best")
    parser.add_argument("--condition", action="append", default=[], help="override --config")
    parser.add_argument("--calibrate", choices=["none", "rank"], default="none",
                        help="'rank' replaces each finding's scores with within-cohort percentiles")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--endpoint", action="append", default=[],
        help="URL[,concurrency[,max_hours]] - repeatable. Work is pulled from a shared "
        "queue so faster servers take proportionally more of it. max_hours releases a "
        "borrowed machine on schedule; the rest finish the queue.",
    )
    parser.add_argument("--api-key", default="not-checked")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--max-retries", type=int, default=8,
                        help="with capped-exponential backoff this rides out a ~5 min server restart")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument(
        "--pipeline", action="store_true",
        help="overlap rendering and inference chunk-by-chunk instead of warming "
        "everything first; total time becomes the max of the two rather than the sum",
    )
    parser.add_argument("--chunk", type=int, default=400, help="studies per pipeline chunk")
    parser.add_argument("--warm-workers", type=int, default=0)
    parser.add_argument("--image-cache-dir", default="")
    parser.add_argument("--limit", type=int, default=0, help="first N studies, for testing")
    parser.add_argument("--dry-run", action="store_true", help="print cost and exit")
    parser.add_argument("--write-only", action="store_true",
                        help="rebuild the CSV from cached responses without new requests")
    args = parser.parse_args()

    if args.write_only:
        studies, _ = all_training_studies(Path(args.data_dir))
        if args.limit:
            studies = studies.head(args.limit)
        names = CONFIGS[args.config] if not args.condition else list(args.condition)
        write_outputs(Path(args.out_dir), studies, names, args)
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
