#!/usr/bin/env python3
"""Run cached, parallel text-only LLM inference experiments.

The script intentionally does not read labels into the prompt. Labels are copied to the
compiled feature table only after inference, for downstream evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI


LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

STATUS_VALUES = ["present", "suspected", "absent", "not_mentioned", "conflicting"]

DEFINITIONS = {
    "ACL": "Any abnormal injury of the anterior cruciate ligament: sprain, partial tear, complete tear, avulsion, or chronic insufficiency.",
    "MCL": "Any abnormal injury of the medial collateral ligament: sprain, partial tear, complete tear, or avulsion. Do not count unrelated medial soft-tissue edema alone.",
    "Medial Meniscus": "A tear or destructive abnormality of the medial meniscus, including root/radial/complex tears or maceration. Do not count isolated intrasubstance degeneration that does not reach an articular surface.",
    "Lateral Meniscus": "A tear or destructive abnormality of the lateral meniscus, including root/radial/complex tears or maceration. Do not count isolated intrasubstance degeneration that does not reach an articular surface.",
    "Medial OA": "Osteoarthritis or meaningful chondral/cartilage loss in the medial tibiofemoral compartment, including focal high-grade chondrosis/defect, denudation, or osteophyte-associated degeneration.",
    "Lateral OA": "Osteoarthritis or meaningful chondral/cartilage loss in the lateral tibiofemoral compartment, including focal high-grade chondrosis/defect, denudation, or osteophyte-associated degeneration.",
    "PF OA": "Patellofemoral osteoarthritis or meaningful patellar/trochlear chondral loss, chondromalacia, fissuring, ulceration, or osteochondral defect.",
    "Effusion": "Abnormal knee joint fluid/effusion or hemarthrosis, including a small or trace effusion when actually present.",
    "Synovitis": "Synovitis, reactive synovial inflammation, or abnormal synovial thickening/proliferation. Effusion alone is not sufficient.",
    "Baker's": "A Baker/popliteal synovial cyst, including a small, ruptured, or decompressed cyst.",
    "Contusion": "Bone marrow or soft-tissue/muscle contusion (bone bruise), but not nonspecific edema solely attributable to osteoarthritis unless described as contusion/bruise or trauma pattern.",
    "Fracture": "Any acute/subacute fracture, insufficiency/stress/subchondral/osteochondral fracture, or bony avulsion. Do not count an osteochondral cartilage defect unless a fracture is stated or strongly implied.",
}


SYSTEM = """You are a careful musculoskeletal radiologist doing probabilistic phenotyping from a knee MRI report. Reports may be in any language. Interpret the medical meaning directly; do not translate verbatim. The report can be inaccurate, incomplete, internally inconsistent, or copied from a template. The target is the actual abnormality in the examined knee, not merely whether a word occurs. Return only the JSON required by the schema."""

REASONING_SYSTEM = """You are a careful musculoskeletal radiologist producing bounded rationale notes for a downstream knee MRI classifier. Reports may be multilingual, inaccurate, incomplete, internally inconsistent, or templated. Return exactly 12 lines in the requested target order and no other text. Each line must use: TARGET | evidence status | probability | rationale of at most 40 words. Do not restate the task, definitions, or report. Do not add a preamble or conclusion. Never repeat a phrase. Stop immediately after the Fracture line."""


@dataclass(frozen=True)
class Experiment:
    name: str
    mode: str
    thinking: bool
    prompt_kind: str


EXPERIMENTS = {
    "joint_extract": Experiment("joint_extract", "joint", False, "extract"),
    "joint_latent": Experiment("joint_latent", "joint", False, "latent"),
    "individual_latent": Experiment("individual_latent", "individual", False, "latent"),
    "two_stage_reasoning": Experiment(
        "two_stage_reasoning", "two_stage", True, "latent"
    ),
}


def prediction_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": STATUS_VALUES},
            "probability": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["status", "probability", "confidence"],
        "additionalProperties": False,
    }


def joint_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "object",
                "properties": {label: prediction_schema() for label in LABELS},
                "required": LABELS,
                "additionalProperties": False,
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }


def response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def format_definitions(labels: list[str]) -> str:
    return "\n".join(f"- {label}: {DEFINITIONS[label]}" for label in labels)


def joint_prompt(report: str, kind: str) -> str:
    if kind == "extract":
        task = """First classify the report evidence for every target as present, suspected, absent, not_mentioned, or conflicting. Then estimate P(actual abnormality is present | this report). Stay fairly close to the explicit report evidence; use indirect findings only when medically compelling. A diagnosis in clinical history or the question being investigated is not a positive finding. A negated finding is absent, not not_mentioned."""
    else:
        task = """Infer every actual target abnormality, accounting for the fact that the report can miss, misstate, or incompletely discuss a condition. First classify explicit report evidence as present, suspected, absent, not_mentioned, or conflicting. Then estimate P(actual abnormality is present | all report evidence), using secondary signs, injury mechanisms, co-occurring findings, and the reliability/specificity of statements. Do not make probabilities extreme merely because a templated sentence says normal. Conversely, do not invent an abnormality just because it was not discussed. Confidence measures confidence in your probability assessment, not severity."""
    return f"""{task}

Target definitions:
{format_definitions(LABELS)}

MRI REPORT:
---
{report}
---"""


def individual_prompt(report: str, label: str) -> str:
    return f"""Focus on exactly one target: {label}.
Definition: {DEFINITIONS[label]}

Classify explicit evidence as present, suspected, absent, not_mentioned, or conflicting. Then estimate P(the actual abnormality is present | the entire report). The report may miss, misstate, template-negate, or incompletely discuss the target. Use secondary signs, injury mechanisms, co-occurring findings, and statement reliability. A diagnosis only in clinical history/question is not a positive finding. Confidence measures confidence in the probability assessment, not severity.

MRI REPORT:
---
{report}
---"""


def load_labeled(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(["StudyInstanceUID", "Report", *LABELS]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    labeled = df.loc[df[LABELS].notna().any(axis=1)].copy()
    if not labeled[LABELS].notna().all(axis=None):
        raise ValueError("Partially labeled rows are not supported")
    labeled.insert(0, "source_row", labeled.index)
    return labeled.reset_index(drop=True)


def load_cache(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("ok"):
                out[
                    (
                        item["uid"],
                        item.get("label", "__joint__"),
                        int(item.get("repeat", 0)),
                    )
                ] = item
    return out


async def run_experiment(
    client: AsyncOpenAI,
    experiment: Experiment,
    data: pd.DataFrame,
    out_dir: Path,
    model: str,
    concurrency: int,
    max_retries: int,
    repeats: int,
    temperature: float,
) -> None:
    cache_path = out_dir / "raw" / f"{experiment.name}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    jobs: list[tuple[str, str, int, str]] = []
    for row in data.itertuples(index=False):
        labels = LABELS if experiment.mode == "individual" else ["__joint__"]
        for label in labels:
            for repeat in range(repeats):
                if (row.StudyInstanceUID, label, repeat) not in cache:
                    prompt = (
                        individual_prompt(row.Report, label)
                        if experiment.mode == "individual"
                        else joint_prompt(row.Report, experiment.prompt_kind)
                    )
                    jobs.append((row.StudyInstanceUID, label, repeat, prompt))

    print(f"{experiment.name}: {len(jobs)} pending, {len(cache)} cached", flush=True)
    if not jobs:
        return

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    started = time.perf_counter()

    async def one(uid: str, label: str, repeat: int, prompt: str) -> None:
        nonlocal completed
        schema = prediction_schema() if experiment.mode == "individual" else joint_schema()
        last_error = ""
        async with sem:
            for attempt in range(max_retries + 1):
                request_start = time.perf_counter()
                try:
                    # Distinct, reproducible seeds give independent stochastic samples while
                    # preserving exact resumability and auditability.
                    seed_bytes = f"{experiment.name}|{uid}|{label}|{repeat}".encode()
                    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")
                    analysis_text = ""
                    analysis_usage = None
                    if experiment.mode == "two_stage":
                        analysis_completion = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": REASONING_SYSTEM},
                                {
                                    "role": "user",
                                    "content": "Analyze explicit evidence, negation, ambiguity, secondary signs, injury patterns, and possible omissions. Produce exactly the 12 requested rationale lines for a downstream scorer.\n\n"
                                    + prompt,
                                },
                            ],
                            temperature=min(1.0, temperature + 0.1),
                            top_p=0.95,
                            seed=seed,
                            max_tokens=10000,
                            extra_body={
                                "chat_template_kwargs": {"enable_thinking": True}
                            },
                        )
                        analysis_message = analysis_completion.choices[0].message
                        native_reasoning = analysis_message.reasoning or ""
                        # Feed only the bounded final rationale into the scorer. Hidden/native
                        # reasoning is retained for audit but can be verbose or meandering.
                        analysis_text = analysis_message.content or ""
                        if not analysis_text.strip():
                            raise ValueError("Reasoning stage returned no text")
                        analysis_usage = (
                            analysis_completion.usage.model_dump()
                            if analysis_completion.usage
                            else None
                        )

                    scoring_prompt = prompt
                    if analysis_text:
                        scoring_prompt += """

FALLIBLE REASONING NOTES FROM A SEPARATE PASS:
---
""" + analysis_text + """
---
Use these notes as additional evidence, but correct them when they conflict with the report or target definitions."""

                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": scoring_prompt},
                        ],
                        temperature=temperature,
                        top_p=0.95,
                        seed=seed,
                        max_tokens=3000 if experiment.thinking else 900,
                        response_format=response_format(
                            f"{experiment.name}_prediction", schema
                        ),
                        extra_body={
                            "chat_template_kwargs": {
                                "enable_thinking": experiment.thinking
                                and experiment.mode != "two_stage"
                            }
                        },
                    )
                    message = completion.choices[0].message
                    parsed = json.loads(message.content or "")
                    elapsed = time.perf_counter() - request_start
                    final_usage = completion.usage.model_dump() if completion.usage else None
                    if analysis_usage and final_usage:
                        combined_usage = {
                            key: int(analysis_usage.get(key) or 0)
                            + int(final_usage.get(key) or 0)
                            for key in [
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                            ]
                        }
                    else:
                        combined_usage = final_usage
                    item = {
                        "experiment": experiment.name,
                        "uid": uid,
                        "label": label,
                        "repeat": repeat,
                        "seed": seed,
                        "temperature": temperature,
                        "reasoning_temperature": (
                            min(1.0, temperature + 0.1)
                            if experiment.mode == "two_stage"
                            else None
                        ),
                        "reasoning_frequency_penalty": (
                            0.0 if experiment.mode == "two_stage" else None
                        ),
                        "ok": True,
                        "elapsed_seconds": elapsed,
                        "usage": combined_usage,
                        "analysis_usage": analysis_usage,
                        "reasoning_chars": len(analysis_text)
                        + len(native_reasoning if experiment.mode == "two_stage" else "")
                        + len(message.reasoning or ""),
                        "analysis": analysis_text or None,
                        "analysis_native_reasoning": (
                            native_reasoning if experiment.mode == "two_stage" else None
                        ),
                        "result": parsed,
                    }
                    async with write_lock:
                        with cache_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    break
                except Exception as exc:  # network/server/parse failures are retried
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < max_retries:
                        await asyncio.sleep(min(2**attempt, 8))
            else:
                raise RuntimeError(
                    f"{experiment.name} failed for {uid}/{label}/repeat={repeat}: {last_error}"
                )

        completed += 1
        if completed % max(1, min(25, len(jobs) // 10)) == 0 or completed == len(jobs):
            rate = completed / max(time.perf_counter() - started, 1e-6)
            print(
                f"{experiment.name}: {completed}/{len(jobs)} ({rate:.2f} requests/s)",
                flush=True,
            )

    await asyncio.gather(*(one(*job) for job in jobs))


def complete_repeat_count(
    records: dict[tuple[str, str, int], dict[str, Any]],
    data: pd.DataFrame,
    experiment: Experiment,
) -> int:
    repeat = 0
    while True:
        keys = []
        for row in data.itertuples(index=False):
            labels = LABELS if experiment.mode == "individual" else ["__joint__"]
            keys.extend((row.StudyInstanceUID, label, repeat) for label in labels)
        if not all(key in records for key in keys):
            return repeat
        repeat += 1


def compile_features(data: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    base = data[["source_row", "StudyInstanceUID", *LABELS]].copy()
    feature_rows: list[dict[str, Any]] = [{} for _ in range(len(base))]
    for name, experiment in EXPERIMENTS.items():
        records = load_cache(out_dir / "raw" / f"{name}.jsonl")
        repeats = complete_repeat_count(records, data, experiment)
        if repeats < 1:
            raise ValueError(f"No complete inference repeat for {name}")
        for i, row in base.iterrows():
            target = feature_rows[i]
            uid = row["StudyInstanceUID"]
            for label in LABELS:
                prefix = f"{name}__{label}"
                sampled = []
                for repeat in range(repeats):
                    key = (
                        uid,
                        label if experiment.mode == "individual" else "__joint__",
                        repeat,
                    )
                    if key not in records:
                        raise ValueError(
                            f"Missing cached result: {name}/{uid}/{label}/repeat={repeat}"
                        )
                    result = records[key]["result"]
                    pred = (
                        result
                        if experiment.mode == "individual"
                        else result["predictions"][label]
                    )
                    sampled.append(pred)
                    target[f"{prefix}__probability_r{repeat}"] = float(
                        pred["probability"]
                    )
                    target[f"{prefix}__confidence_r{repeat}"] = float(
                        pred["confidence"]
                    )
                    target[f"{prefix}__status_r{repeat}"] = pred["status"]
                target[f"{prefix}__probability"] = np.mean(
                    [float(x["probability"]) for x in sampled]
                )
                target[f"{prefix}__confidence"] = np.mean(
                    [float(x["confidence"]) for x in sampled]
                )
                target[f"{prefix}__probability_std"] = np.std(
                    [float(x["probability"]) for x in sampled]
                )
                target[f"{prefix}__confidence_std"] = np.std(
                    [float(x["confidence"]) for x in sampled]
                )
                # The modal evidence status is a compact aggregate; all sampled statuses
                # remain available in the rN columns.
                statuses = pd.Series([x["status"] for x in sampled])
                target[f"{prefix}__status"] = statuses.mode().iloc[0]
                target[f"{prefix}__status_agreement"] = (
                    statuses.value_counts().iloc[0] / repeats
                )
    return pd.concat(
        [base.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1
    )


def write_runtime_summary(out_dir: Path) -> None:
    rows = []
    for name in EXPERIMENTS:
        records = load_cache(out_dir / "raw" / f"{name}.jsonl")
        values = list(records.values())
        elapsed = [float(x["elapsed_seconds"]) for x in values]
        prompt_tokens = [
            int(x["usage"]["prompt_tokens"])
            for x in values
            if x.get("usage") and x["usage"].get("prompt_tokens") is not None
        ]
        completion_tokens = [
            int(x["usage"]["completion_tokens"])
            for x in values
            if x.get("usage") and x["usage"].get("completion_tokens") is not None
        ]
        rows.append(
            {
                "experiment": name,
                "requests": len(values),
                "sum_request_seconds": sum(elapsed),
                "mean_request_seconds": sum(elapsed) / len(elapsed),
                "p95_request_seconds": sorted(elapsed)[
                    min(len(elapsed) - 1, math.floor(0.95 * len(elapsed)))
                ],
                "prompt_tokens": sum(prompt_tokens),
                "completion_tokens": sum(completion_tokens),
                "reasoning_chars": sum(int(x.get("reasoning_chars", 0)) for x in values),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "runtime_summary.csv", index=False)


async def async_main(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    data_path = Path(args.data) if args.data else root / "data/from_host/train.csv"
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_labeled(data_path)
    selected = list(EXPERIMENTS) if args.experiments == ["all"] else args.experiments
    unknown = set(selected) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/") + "/v1",
        timeout=args.timeout,
        max_retries=0,
    )
    for name in selected:
        await run_experiment(
            client,
            EXPERIMENTS[name],
            data,
            out_dir,
            args.model,
            args.concurrency,
            args.max_retries,
            args.repeats,
            args.temperature,
        )
    await client.close()

    # Compile only when every configured experiment is complete.
    if set(selected) == set(EXPERIMENTS) or all(
        (out_dir / "raw" / f"{name}.jsonl").exists() for name in EXPERIMENTS
    ):
        features = compile_features(data, out_dir)
        features.to_csv(out_dir / "llm_features.csv", index=False)
        write_runtime_summary(out_dir)
        print(f"Wrote {out_dir / 'llm_features.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--base-url", default="http://mlserver3:8000")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local-vllm"))
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--experiments", nargs="+", default=["all"], choices=["all", *EXPERIMENTS]
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
