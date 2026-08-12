#!/usr/bin/env python3
"""Predict all 12 knee labels from reports with the selected text-only config.

Configuration: joint actual-abnormality prompt, thinking disabled, temperature 0.6,
five stochastic samples, arithmetic probability mean.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from run_llm_experiments import LABELS, SYSTEM, joint_prompt, joint_schema, response_format


def read_cache(path: Path) -> dict[tuple[str, int], dict]:
    cache = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("ok"):
                cache[(item["uid"], int(item["sample"]))] = item
    return cache


async def infer(args: argparse.Namespace, data: pd.DataFrame, cache_path: Path) -> None:
    cache = read_cache(cache_path)
    jobs = []
    for row in data.itertuples(index=False):
        missing = [
            sample
            for sample in range(args.samples)
            if (row.StudyInstanceUID, sample) not in cache
        ]
        if missing:
            jobs.append((row.StudyInstanceUID, row.Report, missing))
    pending_samples = sum(len(missing) for _, _, missing in jobs)
    print(
        f"{len(jobs)} pending reports / {pending_samples} samples; "
        f"{len(cache)} samples cached",
        flush=True,
    )
    if not jobs:
        return

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/") + "/v1",
        timeout=args.timeout,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    completed_reports = 0
    completed_samples = 0
    started = time.perf_counter()

    async def one(uid: str, report: str, missing: list[int]) -> None:
        nonlocal completed_reports, completed_samples
        seed = int.from_bytes(
            hashlib.sha256(
                f"production|joint_latent|{uid}|{','.join(map(str, missing))}".encode()
            ).digest()[:4],
            "big",
        )
        error = ""
        async with semaphore:
            for attempt in range(args.max_retries + 1):
                request_started = time.perf_counter()
                try:
                    completion = await client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": joint_prompt(report, "latent"),
                            },
                        ],
                        temperature=0.6,
                        top_p=0.95,
                        seed=seed,
                        n=len(missing),
                        max_tokens=900,
                        response_format=response_format(
                            "knee_predictions", joint_schema()
                        ),
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    )
                    if len(completion.choices) != len(missing):
                        raise ValueError(
                            f"Expected {len(missing)} choices, got "
                            f"{len(completion.choices)}"
                        )
                    elapsed = time.perf_counter() - request_started
                    items = []
                    for sample, choice in zip(missing, completion.choices):
                        result = json.loads(choice.message.content or "")
                        probabilities = {
                            label: float(result["predictions"][label]["probability"])
                            for label in LABELS
                        }
                        items.append(
                            {
                                "uid": uid,
                                "sample": sample,
                                "seed": seed,
                                "choice_index": choice.index,
                                "ok": True,
                                "elapsed_seconds": elapsed,
                                "usage": completion.usage.model_dump()
                                if completion.usage
                                else None,
                                "probabilities": probabilities,
                            }
                        )
                    async with write_lock:
                        with cache_path.open("a", encoding="utf-8") as handle:
                            for item in items:
                                handle.write(json.dumps(item) + "\n")
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.max_retries:
                        await asyncio.sleep(min(2**attempt, 8))
            else:
                raise RuntimeError(f"Failed {uid}/samples={missing}: {error}")

        completed_reports += 1
        completed_samples += len(missing)
        if completed_reports % 10 == 0 or completed_reports == len(jobs):
            rate = completed_samples / max(time.perf_counter() - started, 1e-6)
            remaining_minutes = (
                pending_samples - completed_samples
            ) / max(rate, 1e-6) / 60
            print(
                f"{completed_reports}/{len(jobs)} reports, "
                f"{completed_samples}/{pending_samples} samples "
                f"({rate:.2f} samples/s; "
                f"~{remaining_minutes:.1f} min remaining)",
                flush=True,
            )

    await asyncio.gather(*(one(*job) for job in jobs))
    await client.close()


def compile_predictions(
    data: pd.DataFrame, cache_path: Path, output_path: Path, samples: int
) -> None:
    cache = read_cache(cache_path)
    rows = []
    for source_row, row in data.iterrows():
        sampled = []
        for sample in range(samples):
            key = (row["StudyInstanceUID"], sample)
            if key not in cache:
                raise ValueError(f"Missing cached prediction: {key}")
            sampled.append(cache[key]["probabilities"])
        output = {
            "source_row": source_row,
            "StudyInstanceUID": row["StudyInstanceUID"],
        }
        for label in LABELS:
            values = [float(item[label]) for item in sampled]
            output[label] = float(np.mean(values))
            output[f"{label}__sample_std"] = float(np.std(values, ddof=0))
        rows.append(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} predictions to {output_path}", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    data = pd.read_csv(input_path)
    required = {"StudyInstanceUID", "Report"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.StudyInstanceUID.duplicated().any():
        raise ValueError("StudyInstanceUID must be unique")
    if data.Report.isna().any():
        raise ValueError("Report contains null values")

    output_path = Path(args.output)
    cache_path = Path(args.cache) if args.cache else output_path.with_suffix(".jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    await infer(args, data, cache_path)
    compile_predictions(data, cache_path, output_path, args.samples)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(root / "data/from_host/train.csv"))
    parser.add_argument(
        "--output", default=str(here / "predictions/text_only_predictions.csv")
    )
    parser.add_argument("--cache", default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--base-url", default="http://mlserver3:8000")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local-vllm"))
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
