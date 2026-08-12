#!/usr/bin/env python3
"""Run cached prompt/sampling refinements for raw probability ensembles."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from run_llm_experiments import (
    DEFINITIONS,
    LABELS,
    STATUS_VALUES,
    SYSTEM,
    format_definitions,
    joint_prompt,
    joint_schema,
    load_labeled,
    response_format,
)


@dataclass(frozen=True)
class Refinement:
    name: str
    temperature: float
    prompt_kind: str
    probability_only: bool = False


REFINEMENTS = {
    "baseline_t02": Refinement("baseline_t02", 0.2, "baseline"),
    "baseline_t10": Refinement("baseline_t10", 1.0, "baseline"),
    "image_reviewer_t06": Refinement("image_reviewer_t06", 0.6, "image_reviewer"),
    "skeptical_t06": Refinement("skeptical_t06", 0.6, "skeptical"),
    "decomposed_t06": Refinement("decomposed_t06", 0.6, "decomposed"),
    "probability_only_t06": Refinement(
        "probability_only_t06", 0.6, "baseline", probability_only=True
    ),
}


def probability_only_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "object",
                "properties": {
                    label: {"type": "number", "minimum": 0, "maximum": 1}
                    for label in LABELS
                },
                "required": LABELS,
                "additionalProperties": False,
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }


def refinement_prompt(report: str, config: Refinement) -> str:
    if config.prompt_kind == "baseline":
        task = """Infer every actual target abnormality, accounting for the fact that the report can miss, misstate, or incompletely discuss a condition. Estimate P(actual abnormality is present | all report evidence), using secondary signs, injury mechanisms, co-occurring findings, and the reliability/specificity of statements. Do not make probabilities extreme merely because a templated sentence says normal. Conversely, do not invent an abnormality just because it was not discussed."""
    elif config.prompt_kind == "image_reviewer":
        task = """The target labels come from an independent expert review of the source knee MRI, while this report is a noisy proxy that can omit findings, contain template errors, or disagree with the images. Predict the probability that the independent image reviewer would mark each target present. Use explicit findings, highly specific secondary imaging signs, injury patterns, and internal report reliability. A copied normal sentence is weaker when contradicted by specific findings. Do not predict a condition solely because it is commonly associated with another finding."""
    elif config.prompt_kind == "skeptical":
        task = """Estimate the probability that each actual abnormality is present despite possible report error or omission, but be conservative about indirect inference. Give greatest weight to specific positive or negative imaging statements. Use secondary signs only when they are anatomically and diagnostically specific; generic trauma, edema, effusion, or common co-occurrence is insufficient by itself. Treat non-discussion as uncertainty rather than as either a positive or a definite negative."""
    elif config.prompt_kind == "decomposed":
        task = """For each target, internally perform four steps before scoring: (1) identify explicit positive, suspected, negative, conflicting, or missing evidence; (2) assess whether relevant sentences look specific or templated; (3) identify only medically valid secondary signs and injury-pattern evidence; and (4) reconcile these into P(actual abnormality is present | report). Do not expose this internal analysis. Avoid both literal keyword extraction and speculative co-occurrence."""
    else:
        raise ValueError(config.prompt_kind)

    output = (
        "Return only the required probability JSON."
        if config.probability_only
        else "Also classify explicit evidence status and report confidence in the probability estimate as required by the schema. Confidence is not severity."
    )
    return f"""{task}

{output}

Target definitions:
{format_definitions(LABELS)}

MRI REPORT:
---
{report}
---"""


def load_cache(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("ok"):
                records[(item["uid"], int(item["repeat"]))] = item
    return records


async def run_config(
    client: AsyncOpenAI,
    config: Refinement,
    data,
    raw_dir: Path,
    model: str,
    repeats: int,
    concurrency: int,
    max_retries: int,
) -> None:
    path = raw_dir / f"{config.name}.jsonl"
    cache = load_cache(path)
    jobs = []
    for row in data.itertuples(index=False):
        for repeat in range(repeats):
            if (row.StudyInstanceUID, repeat) not in cache:
                jobs.append(
                    (
                        row.StudyInstanceUID,
                        repeat,
                        refinement_prompt(row.Report, config),
                    )
                )
    print(f"{config.name}: {len(jobs)} pending, {len(cache)} cached", flush=True)
    if not jobs:
        return

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    completed = 0
    started = time.perf_counter()
    schema = probability_only_schema() if config.probability_only else joint_schema()

    async def one(uid: str, repeat: int, prompt: str) -> None:
        nonlocal completed
        seed_bytes = f"refinement|{config.name}|{uid}|{repeat}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")
        error = ""
        async with sem:
            for attempt in range(max_retries + 1):
                request_started = time.perf_counter()
                try:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=config.temperature,
                        top_p=0.95,
                        seed=seed,
                        max_tokens=500 if config.probability_only else 900,
                        response_format=response_format(
                            f"{config.name}_prediction", schema
                        ),
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    )
                    message = completion.choices[0].message
                    result = json.loads(message.content or "")
                    if config.probability_only:
                        probabilities = {
                            label: float(result["predictions"][label])
                            for label in LABELS
                        }
                    else:
                        probabilities = {
                            label: float(result["predictions"][label]["probability"])
                            for label in LABELS
                        }
                    item = {
                        "config": config.name,
                        "uid": uid,
                        "repeat": repeat,
                        "seed": seed,
                        "temperature": config.temperature,
                        "top_p": 0.95,
                        "ok": True,
                        "elapsed_seconds": time.perf_counter() - request_started,
                        "usage": completion.usage.model_dump()
                        if completion.usage
                        else None,
                        "probabilities": probabilities,
                        "result": result,
                    }
                    async with lock:
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < max_retries:
                        await asyncio.sleep(min(2**attempt, 8))
            else:
                raise RuntimeError(
                    f"{config.name} failed for {uid}/repeat={repeat}: {error}"
                )
        completed += 1
        interval = max(1, min(25, len(jobs) // 10))
        if completed % interval == 0 or completed == len(jobs):
            rate = completed / max(time.perf_counter() - started, 1e-6)
            print(
                f"{config.name}: {completed}/{len(jobs)} ({rate:.2f} requests/s)",
                flush=True,
            )

    await asyncio.gather(*(one(*job) for job in jobs))


async def main_async(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    data = load_labeled(root / "data/from_host/train.csv")
    raw_dir = here / "artifacts/refinement/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = list(REFINEMENTS) if args.configs == ["all"] else args.configs
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/") + "/v1",
        timeout=args.timeout,
        max_retries=0,
    )
    for name in selected:
        await run_config(
            client,
            REFINEMENTS[name],
            data,
            raw_dir,
            args.model,
            args.repeats,
            args.concurrency,
            args.max_retries,
        )
    await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://mlserver3:8000")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local-vllm"))
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--configs", nargs="+", default=["all"], choices=["all", *REFINEMENTS]
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
