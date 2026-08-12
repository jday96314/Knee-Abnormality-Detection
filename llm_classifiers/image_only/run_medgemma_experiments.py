#!/usr/bin/env python3
"""Zero-shot image-only knee abnormality experiments against a vLLM MedGemma server.

Every condition is one field's change from a single reference, so each axis can be
read as a contrast:

  prompt       joint / per-finding / two-stage readings          (see prompts.py)
  guided       json_schema-constrained vs free-form + tolerant parsing
  format       the scale the answer is asked on: 0-1, 0-100, or an ordinal enum
  sampling     temperature, top_p, and number of averaged seeded samples
  image        which series, how many slices, resolution, windowing, montage
  downsample   how a long series is reduced: fixed, proportional, or slab MIP/mean
  context      the study-level image budget, from 2k to 65k image tokens

The last three are not decoration. Series reach 320 slices and studies 589, so at
~256 tokens per image a naive "send everything" would cost ~150k tokens for the
largest study and overrun the window for about 2% of them.

No labels and no report text ever enter a prompt. Labels are joined afterwards,
in evaluate.py.

Raw responses are appended to artifacts/raw/<condition>.jsonl as they succeed, so
interrupted runs resume. Reuse is keyed by study and image spec rather than by cohort,
so a screening pilot's requests are inherited by the full run.

    python llm_classifiers/image_only/run_medgemma_experiments.py --sweep all --dry-run
    python llm_classifiers/image_only/run_medgemma_experiments.py --sweep all --pilot-studies 15
    python llm_classifiers/image_only/run_medgemma_experiments.py --sweep sane
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from dicom_io import (
    MAX_IMAGES_HARD_CAP,
    LABELS,
    TOKENS_PER_IMAGE,
    ImageSpec,
    build_study_images,
    cohort_id,
    complete_labeled_studies,
    image_count,
    plan_study_images,
    slices_represented,
)
from prompts import (
    DESCRIBE_SYSTEM,
    JOINT_STRATEGIES,
    PER_LABEL_STRATEGIES,
    SYSTEM_BASE,
    SYSTEM_TERSE,
    binary_json,
    binary_yesno,
    describe_prompt,
    score_reading_prompt,
)

DEFAULT_BASE_URL = "http://mlserver2:8000/v1"
DEFAULT_MODEL = "google/medgemma-1.5-4b-it"

# MedGemma 1.5 emits an optional thinking block delimited by these sentinels. The
# server is not started with a reasoning parser, so the block arrives inside
# `content` and has to be stripped before parsing free-form output.
THOUGHT_RE = re.compile(r"<unused94>.*?<unused95>", re.DOTALL)
THOUGHT_OPEN_RE = re.compile(r"<unused94>.*", re.DOTALL)


class ParseFailure(Exception):
    """Raised when a response arrives but cannot be scored.

    Carries the offending text so a failed cell can be diagnosed from the cache
    instead of only from a truncated log line.
    """

    def __init__(self, reason: str, content: str, finish_reason: str | None) -> None:
        super().__init__(reason)
        self.content = content
        self.finish_reason = finish_reason


# A bare 0-1 number invites a uniform hedge: the model answers 0.5 for all twelve
# findings. These alternative response formats ask for the same judgement on scales
# the model may commit to more readily, and are swept as their own axis.
LIKERT = ["absent", "very unlikely", "unlikely", "possible", "likely", "very likely", "definite"]
LIKERT_VALUES = dict(zip(LIKERT, [0.02, 0.08, 0.20, 0.50, 0.75, 0.90, 0.97]))


@dataclass(frozen=True)
class Condition:
    """One fully specified experimental cell."""

    name: str
    strategy: str
    guided: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    samples: int = 1
    image: ImageSpec = field(default_factory=ImageSpec)
    axis: str = "reference"
    max_tokens: int = 700
    answer_format: str = "probability"  # or "percent_int", "likert"

    @property
    def per_label(self) -> bool:
        return self.strategy in PER_LABEL_STRATEGIES

    @property
    def uses_logprobs(self) -> bool:
        return self.strategy == "binary_logprob"


REFERENCE = Condition(
    name="ref_joint_definitions",
    strategy="joint_definitions",
    guided=True,
    temperature=0.0,
    samples=1,
    image=ImageSpec(),
    axis="reference",
)


def build_conditions() -> dict[str, Condition]:
    conditions: list[Condition] = [REFERENCE]

    # --- Axis 1: prompting strategy (guided on, greedy, reference images) ---
    for strategy in ["joint_plain", "joint_checklist", "joint_prevalence_free"]:
        conditions.append(replace(REFERENCE, name=f"prompt_{strategy}",
                                  strategy=strategy, axis="prompt"))
    conditions += [
        replace(REFERENCE, name="prompt_binary_json", strategy="binary_json",
                axis="prompt", max_tokens=200),
        replace(REFERENCE, name="prompt_binary_logprob", strategy="binary_logprob",
                axis="prompt", max_tokens=1),
        replace(REFERENCE, name="prompt_two_stage", strategy="two_stage",
                axis="prompt", max_tokens=1400),
    ]

    # --- Axis 2: guided decoding off, same prompts ---
    for strategy in ["joint_definitions", "joint_checklist", "binary_json"]:
        conditions.append(
            replace(
                REFERENCE,
                name=f"guided_off_{strategy}",
                strategy=strategy,
                guided=False,
                axis="guided",
                # Unconstrained output may open with a MedGemma thinking block before
                # the JSON, so the budget has to cover both or the object is truncated.
                max_tokens=2500 if strategy.startswith("joint") else 800,
            )
        )
    conditions.append(
        replace(REFERENCE, name="guided_off_two_stage", strategy="two_stage",
                guided=False, axis="guided", max_tokens=1400)
    )

    # --- Axis 2b: answer scale under guided decoding ---
    # A constrained bare probability is where the uniform-0.5 hedge appears; these
    # ask for the same judgement on a coarser scale the model may commit to.
    for answer_format in ["percent_int", "likert"]:
        conditions.append(
            replace(REFERENCE, name=f"format_{answer_format}",
                    answer_format=answer_format, axis="format")
        )
        conditions.append(
            replace(REFERENCE, name=f"format_{answer_format}_checklist",
                    strategy="joint_checklist", answer_format=answer_format, axis="format")
        )
    conditions.append(
        replace(REFERENCE, name="format_likert_two_stage", strategy="two_stage",
                answer_format="likert", axis="format", max_tokens=1400)
    )

    # --- Axis 3: sampling configuration (reference prompt, guided on) ---
    conditions += [
        replace(REFERENCE, name="sample_t06_x3", temperature=0.6, top_p=0.95,
                samples=3, axis="sampling"),
        replace(REFERENCE, name="sample_t06_x5", temperature=0.6, top_p=0.95,
                samples=5, axis="sampling"),
        replace(REFERENCE, name="sample_t10_x3", temperature=1.0, top_p=0.95,
                samples=3, axis="sampling"),
        replace(REFERENCE, name="sample_t06_p80_x3", temperature=0.6, top_p=0.80,
                samples=3, axis="sampling"),
        replace(REFERENCE, name="sample_t03_x3", temperature=0.3, top_p=0.95,
                samples=3, axis="sampling"),
    ]

    # --- Axis 4: how the study is rendered ---
    image_variants = {
        "image_sag_only": ImageSpec(plan="sag_fluid", slices_per_series=4),
        "image_tri_fluid": ImageSpec(plan="tri_fluid", slices_per_series=4),
        "image_sparse2": ImageSpec(slices_per_series=2),
        "image_dense8": ImageSpec(slices_per_series=8),
        "image_montage9": ImageSpec(slices_per_series=9, layout="montage"),
        "image_size448": ImageSpec(size=448),
        "image_dicom_window": ImageSpec(window="dicom"),
    }
    for name, spec in image_variants.items():
        conditions.append(replace(REFERENCE, name=name, image=spec, axis="image"))

    # --- Axis 5: downsampling strategy for long series ---
    # Series reach 320 slices, so a fixed count samples a 20-slice and a 320-slice
    # acquisition 16x differently. These hold the image count roughly constant and
    # vary only how those images are drawn from the stack.
    downsample_variants = {
        "down_proportional": ImageSpec(sampling="proportional", stride=8, slices_per_series=12),
        "down_proportional_dense": ImageSpec(
            sampling="proportional", stride=4, slices_per_series=24
        ),
        "down_slab_mip4": ImageSpec(sampling="slab_mip", slices_per_series=4),
        "down_slab_mip8": ImageSpec(sampling="slab_mip", slices_per_series=8),
        "down_slab_mean4": ImageSpec(sampling="slab_mean", slices_per_series=4),
        "down_montage16": ImageSpec(slices_per_series=16, layout="montage"),
        "down_montage25": ImageSpec(slices_per_series=25, layout="montage"),
        "down_montage_mip16": ImageSpec(
            slices_per_series=16, layout="montage", sampling="slab_mip"
        ),
        "down_full_coverage": ImageSpec(center_fraction=1.0),
    }
    for name, spec in downsample_variants.items():
        conditions.append(replace(REFERENCE, name=name, image=spec, axis="downsample"))

    # --- Axis 6: context length ---
    # Every image costs ~256 prompt tokens. These take every slice of every series
    # (median 5 series, up to 14) so the study-level budget is the only thing
    # limiting the request, then vary that budget over a ladder from 2k to 65k image
    # tokens. The point is to find where MedGemma stops benefiting from more pixels,
    # which for a 4B model is usually far short of the 128k the server advertises.
    for cap in [8, 16, 32, 64, 128, 256]:
        conditions.append(
            replace(
                REFERENCE,
                name=f"ctx_{cap}img_{cap * 256 // 1024}k",
                image=ImageSpec(
                    plan="all_series",
                    sampling="proportional",
                    stride=1,
                    slices_per_series=MAX_IMAGES_HARD_CAP,
                    max_images=cap,
                ),
                axis="context",
                # A long image prefix makes truncation more likely, not less.
                max_tokens=900,
            )
        )

    return {condition.name: condition for condition in conditions}


CONDITIONS = build_conditions()

SWEEPS = {
    "smoke": ["ref_joint_definitions", "prompt_binary_logprob", "guided_off_joint_definitions"],
    "prompt": [n for n, c in CONDITIONS.items() if c.axis in ("reference", "prompt")],
    "guided": [n for n, c in CONDITIONS.items() if c.axis in ("reference", "guided")],
    "format": [n for n, c in CONDITIONS.items() if c.axis in ("reference", "format")],
    "sampling": [n for n, c in CONDITIONS.items() if c.axis in ("reference", "sampling")],
    "image": [n for n, c in CONDITIONS.items() if c.axis in ("reference", "image")],
    "downsample": [n for n, c in CONDITIONS.items() if c.axis in ("reference", "downsample")],
    "context": [n for n, c in CONDITIONS.items() if c.axis in ("reference", "context")],
    "all": list(CONDITIONS),
}


# ---------------------------------------------------------------------------
# Schemas and output parsing
# ---------------------------------------------------------------------------


def answer_property(answer_format: str) -> dict[str, Any]:
    if answer_format == "percent_int":
        return {"type": "integer", "minimum": 0, "maximum": 100}
    if answer_format == "likert":
        return {"type": "string", "enum": LIKERT}
    return {"type": "number", "minimum": 0, "maximum": 1}


def joint_schema(answer_format: str = "probability") -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {label: answer_property(answer_format) for label in LABELS},
        "required": LABELS,
        "additionalProperties": False,
    }


def single_schema(answer_format: str = "probability") -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"probability": answer_property(answer_format)},
        "required": ["probability"],
        "additionalProperties": False,
    }


def answer_scale_instruction(answer_format: str) -> str:
    """Text appended so the prompt matches the schema the response is constrained to."""
    if answer_format == "percent_int":
        return (
            "\n\nAnswer each value as a whole-number percentage from 0 to 100, not as a "
            "decimal fraction."
        )
    if answer_format == "likert":
        return (
            "\n\nAnswer each value with exactly one of these words: "
            + ", ".join(f'"{level}"' for level in LIKERT)
            + "."
        )
    return ""


def response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def strip_thinking(text: str) -> str:
    text = THOUGHT_RE.sub("", text)
    # A block opened but never closed means the answer was truncated inside thinking.
    return THOUGHT_OPEN_RE.sub("", text).strip()


def extract_payload(text: str) -> Any:
    """Recover the answer from free-form output.

    Handles thinking blocks, ```json fences, and prose wrapped around the object.
    Unguided per-finding requests often answer with a bare fenced scalar rather than
    an object, so a scalar is returned as-is and interpreted by the caller.
    """
    text = strip_thinking(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    depth = 0
    start = -1
    for index, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                if isinstance(parsed, dict):
                    return parsed

    # Last resort for a one-number answer wrapped in prose, e.g. "Probability: 0.85".
    # Only unambiguous cases qualify: with several numbers present there is no safe
    # way to tell which one is the answer, so the cell is failed instead of guessed.
    numbers = re.findall(r"\d*\.?\d+\s*%?", text)
    if len(numbers) == 1:
        return numbers[0].strip()
    raise ValueError(f"no JSON answer in output: {text[:200]!r}")


def _normalise(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


_LABEL_ALIASES = {_normalise(label): label for label in LABELS}
_LABEL_ALIASES.update(
    {
        _normalise("anterior cruciate ligament"): "ACL",
        _normalise("medial collateral ligament"): "MCL",
        _normalise("medial meniscus tear"): "Medial Meniscus",
        _normalise("lateral meniscus tear"): "Lateral Meniscus",
        _normalise("bakers cyst"): "Baker's",
        _normalise("baker"): "Baker's",
        _normalise("patellofemoral OA"): "PF OA",
        _normalise("joint effusion"): "Effusion",
        _normalise("bone contusion"): "Contusion",
    }
)


def _to_probability(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        ordinal = LIKERT_VALUES.get(value.strip().lower())
        if ordinal is not None:
            return ordinal
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return None
        number = float(match.group())
        if "%" in value:
            number /= 100.0
    elif isinstance(value, dict):
        for key in ("probability", "prob", "p", "value", "score", "likelihood"):
            if key in value:
                return _to_probability(value[key])
        # Unguided per-finding requests invent their own key names
        # ({"finding_present": 1.0}). With exactly one key there is no ambiguity
        # about which value is the answer, so accept it rather than fail the cell.
        if len(value) == 1:
            return _to_probability(next(iter(value.values())))
        return None
    else:
        return None
    if number > 1.0:  # the model sometimes answers on a 0-100 scale
        number = number / 100.0 if number <= 100.0 else 1.0
    return min(max(number, 0.0), 1.0)


def parse_joint(payload: dict[str, Any]) -> dict[str, float]:
    """Map a returned object onto the twelve labels, tolerating renamed keys."""
    if len(payload) == 1:
        inner = next(iter(payload.values()))
        if isinstance(inner, dict) and len(inner) >= len(LABELS) // 2:
            payload = inner

    resolved: dict[str, float] = {}
    for key, value in payload.items():
        label = _LABEL_ALIASES.get(_normalise(key))
        if label is None:
            normalised = _normalise(key)
            matches = [
                candidate
                for alias, candidate in _LABEL_ALIASES.items()
                if alias and (alias in normalised or normalised in alias)
            ]
            label = matches[0] if len(set(matches)) == 1 else None
        if label is None:
            continue
        probability = _to_probability(value)
        if probability is not None:
            resolved.setdefault(label, probability)

    missing = [label for label in LABELS if label not in resolved]
    if missing:
        raise ValueError(f"missing labels in output: {missing}")
    return resolved


def yes_probability(top_logprobs: list[Any]) -> float:
    """P(yes) / (P(yes) + P(no)) over the first generated token's alternatives."""
    yes = no = 0.0
    for entry in top_logprobs:
        token = entry.token.strip().lower().strip('"*')
        probability = math.exp(entry.logprob)
        if token.startswith("yes"):
            yes += probability
        elif token.startswith("no"):
            no += probability
    total = yes + no
    if total <= 0.0:
        raise ValueError("neither yes nor no appeared in the top logprobs")
    return yes / total


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------


def seed_for(parts: str) -> int:
    return int.from_bytes(hashlib.sha256(parts.encode()).digest()[:4], "big")


def image_content(images: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [{"type": "image_url", "image_url": {"url": uri}} for _, uri in images]


@dataclass
class Job:
    uid: str
    label: str  # "__joint__" for whole-study requests
    repeat: int


class Runner:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        data_dir: Path,
        series: pd.DataFrame,
        out_dir: Path,
        cohort: str,
        concurrency: int,
        max_retries: int,
    ) -> None:
        self.client = client
        self.model = model
        self.data_dir = data_dir
        self.series = series
        self.out_dir = out_dir
        self.cohort = cohort
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.write_lock = asyncio.Lock()
        self._image_cache: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._image_lock = asyncio.Lock()
        self._reading_cache: dict[tuple[str, str], str] = {}

    async def images_for(self, uid: str, spec: ImageSpec) -> list[tuple[str, str]]:
        key = (uid, spec.key())
        async with self._image_lock:
            if key in self._image_cache:
                return self._image_cache[key]
        images = await asyncio.to_thread(
            build_study_images,
            self.data_dir,
            self.series,
            uid,
            spec,
            self.out_dir / "image_cache",
        )
        if not images:
            raise ValueError(f"no series matched plan {spec.plan!r} for study {uid}")
        async with self._image_lock:
            self._image_cache[key] = images
        return images

    async def describe(self, uid: str, condition: Condition, seed: int) -> tuple[str, Any]:
        """Stage one of the two-stage strategy: a free-text reading of the images."""
        images = await self.images_for(uid, condition.image)
        captions = [caption for caption, _ in images]
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DESCRIBE_SYSTEM},
                {
                    "role": "user",
                    "content": image_content(images)
                    + [{"type": "text", "text": describe_prompt(captions)}],
                },
            ],
            temperature=condition.temperature,
            top_p=condition.top_p,
            seed=seed,
            max_tokens=1200,
        )
        reading = strip_thinking(completion.choices[0].message.content or "")
        if not reading:
            raise ValueError("describe stage returned no text")
        return reading, completion.usage

    async def run_one(self, condition: Condition, job: Job) -> dict[str, Any]:
        seed = seed_for(f"{condition.name}|{job.uid}|{job.label}|{job.repeat}")
        images = await self.images_for(job.uid, condition.image)
        captions = [caption for caption, _ in images]
        started = time.perf_counter()
        reading = None
        stage_one_usage = None

        scale = answer_scale_instruction(condition.answer_format)

        if condition.strategy == "two_stage":
            reading, stage_one_usage = await self.describe(job.uid, condition, seed)
            messages = [
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": score_reading_prompt(reading) + scale},
            ]
            schema = joint_schema(condition.answer_format)
        elif condition.per_label:
            builder = binary_yesno if condition.uses_logprobs else binary_json
            system = SYSTEM_TERSE if condition.uses_logprobs else SYSTEM_BASE
            text = builder(captions, job.label)
            if not condition.uses_logprobs:
                text += scale
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": image_content(images) + [{"type": "text", "text": text}],
                },
            ]
            schema = single_schema(condition.answer_format)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_BASE},
                {
                    "role": "user",
                    "content": image_content(images)
                    + [
                        {
                            "type": "text",
                            "text": JOINT_STRATEGIES[condition.strategy](captions) + scale,
                        }
                    ],
                },
            ]
            schema = joint_schema(condition.answer_format)

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": condition.temperature,
            "top_p": condition.top_p,
            "seed": seed,
            "max_tokens": condition.max_tokens,
        }
        if condition.uses_logprobs:
            request.update(logprobs=True, top_logprobs=20)
        elif condition.guided:
            request["response_format"] = response_format(condition.name, schema)

        completion = await self.client.chat.completions.create(**request)
        choice = completion.choices[0]
        content = choice.message.content or ""

        try:
            if condition.uses_logprobs:
                top = choice.logprobs.content[0].top_logprobs
                result = {job.label: yes_probability(top)}
                raw_alternatives = [
                    {"token": entry.token, "logprob": entry.logprob} for entry in top
                ]
            else:
                payload = extract_payload(content)
                raw_alternatives = None
                if condition.per_label:
                    value = (
                        payload.get("probability", payload)
                        if isinstance(payload, dict)
                        else payload
                    )
                    probability = _to_probability(value)
                    if probability is None:
                        raise ValueError(f"no probability in {payload!r}")
                    result = {job.label: probability}
                else:
                    if not isinstance(payload, dict):
                        raise ValueError(f"expected an object of 12 findings, got {payload!r}")
                    result = parse_joint(payload)
        except (ValueError, KeyError, IndexError, AttributeError) as exc:
            raise ParseFailure(str(exc), content, choice.finish_reason) from exc

        return {
            "condition": condition.name,
            "cohort": self.cohort,
            "image_spec": condition.image.key(),
            "uid": job.uid,
            "label": job.label,
            "repeat": job.repeat,
            "seed": seed,
            "n_images": len(images),
            "ok": True,
            "elapsed_seconds": time.perf_counter() - started,
            "finish_reason": choice.finish_reason,
            "had_thinking": "<unused94>" in content,
            "usage": completion.usage.model_dump() if completion.usage else None,
            "stage_one_usage": (
                stage_one_usage.model_dump() if stage_one_usage else None
            ),
            "reading": reading,
            "raw_content": None if condition.uses_logprobs else content,
            "top_logprobs": raw_alternatives,
            "result": result,
        }

    async def run_condition(self, condition: Condition, uids: list[str]) -> None:
        cache_path = self.out_dir / "raw" / f"{condition.name}.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        done = load_cache_keys(cache_path, condition.image.key())

        labels = LABELS if condition.per_label else ["__joint__"]
        jobs = [
            Job(uid, label, repeat)
            for uid in uids
            for label in labels
            for repeat in range(condition.samples)
            if (uid, label, repeat) not in done
        ]
        print(
            f"{condition.name}: {len(jobs)} pending, {len(done)} cached",
            flush=True,
        )
        if not jobs:
            return

        completed = 0
        failures: list[str] = []
        started = time.perf_counter()

        async def worker(job: Job) -> None:
            nonlocal completed
            last_error = ""
            last_failure: ParseFailure | None = None
            async with self.semaphore:
                for attempt in range(self.max_retries + 1):
                    try:
                        record = await self.run_one(condition, job)
                    except ParseFailure as exc:
                        last_error = f"ParseFailure: {exc}"
                        last_failure = exc
                        # Greedy decoding is deterministic, so a re-request with the
                        # same seed would reproduce the same unparseable text.
                        if condition.temperature == 0.0:
                            break
                        if attempt < self.max_retries:
                            await asyncio.sleep(min(2**attempt, 8))
                        continue
                    except Exception as exc:  # noqa: BLE001 - transport errors are retried
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt < self.max_retries:
                            await asyncio.sleep(min(2**attempt, 8))
                        continue
                    async with self.write_lock:
                        with cache_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed += 1
                    report_progress()
                    return

            # A cell that never parses is itself a result: record it, with the text
            # that defeated the parser, and let the sweep continue.
            failures.append(f"{job.uid[:12]}/{job.label}: {last_error}")
            async with self.write_lock:
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "condition": condition.name,
                                "cohort": self.cohort,
                                "image_spec": condition.image.key(),
                                "uid": job.uid,
                                "label": job.label,
                                "repeat": job.repeat,
                                "ok": False,
                                "error": last_error,
                                "finish_reason": (
                                    last_failure.finish_reason if last_failure else None
                                ),
                                "raw_content": (
                                    last_failure.content if last_failure else None
                                ),
                                "result": {},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            completed += 1
            report_progress()

        def report_progress() -> None:
            step = max(1, min(25, len(jobs) // 8))
            if completed % step == 0 or completed == len(jobs):
                rate = completed / max(time.perf_counter() - started, 1e-6)
                print(
                    f"  {condition.name}: {completed}/{len(jobs)} ({rate:.2f} req/s)",
                    flush=True,
                )

        await asyncio.gather(*(worker(job) for job in jobs))
        if failures:
            print(
                f"  {condition.name}: {len(failures)} unrecoverable "
                f"(e.g. {failures[0]})",
                flush=True,
            )


def load_cache_keys(path: Path, image_spec: str) -> set[tuple[str, str, int]]:
    """Keys already present for this image spec.

    Records from a different image spec describe different pictures and must not be
    reused, even for the same study.

    Cohort deliberately does not gate reuse. A record describes one study under one
    image spec and stays valid however many other studies happen to be extracted, so
    a screening pilot's requests are inherited by the full run for free. Cohort still
    gates *scoring*, in evaluate.py, where mixing studies would change the metric.
    """
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("image_spec") != image_spec:
                continue
            if not record.get("ok"):
                continue
            keys.add((record["uid"], record["label"], record["repeat"]))
    return keys


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled, series = complete_labeled_studies(data_dir)
    available = len(labeled)
    if args.limit_studies:
        labeled = labeled.head(args.limit_studies)
    if args.pilot_studies:
        labeled = pilot_subset(labeled, args.pilot_studies)
    series = series[series["StudyInstanceUID"].isin(labeled["StudyInstanceUID"])]
    uids = labeled["StudyInstanceUID"].tolist()
    cohort = cohort_id(uids)

    scope = "screening pilot of" if args.pilot_studies else ""
    print(
        f"cohort {cohort}: {scope} {len(uids)} labeled studies "
        f"({available} fully extracted)".replace("  ", " "),
        flush=True,
    )
    positives = labeled[LABELS].sum().astype(int)
    print("positives per label:", positives.to_dict(), flush=True)
    # The pilot is a deliberate subset, so the full-cohort floor does not apply to it.
    if not args.pilot_studies and len(uids) < args.min_studies:
        raise SystemExit(
            f"only {len(uids)} labeled studies are fully extracted; "
            f"pass --min-studies {len(uids)} to score anyway"
        )
    if positives.min() == 0 or (len(uids) - positives).min() == 0:
        print(
            "warning: at least one label has no positive or no negative case in this "
            "cohort; its AUC will be undefined until extraction adds more studies",
            flush=True,
        )

    requested: list[str] = []
    for sweep in args.sweep or []:
        requested += load_screen(out_dir) if sweep == "sane" else SWEEPS[sweep]
    requested += list(args.condition)
    names = list(dict.fromkeys(requested))
    unknown = [name for name in names if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")

    if args.dry_run:
        # Deliberately before anything is written: a dry run must not disturb the
        # cohort that evaluate.py will score.
        dry_run(data_dir, series, uids, names, args.context_length)
        return

    labeled.to_csv(out_dir / "cohort_labels.csv", index=False)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    runner = Runner(
        client=client,
        model=args.model,
        data_dir=data_dir,
        series=series,
        out_dir=out_dir,
        cohort=cohort,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )

    manifest = {
        "cohort": cohort,
        "model": args.model,
        "base_url": args.base_url,
        "n_studies": len(uids),
        "studies": uids,
        "conditions": {
            name: {**asdict(CONDITIONS[name]), "image": asdict(CONDITIONS[name].image)}
            for name in names
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    overall = time.perf_counter()
    for name in names:
        await runner.run_condition(CONDITIONS[name], uids)
    print(f"done in {time.perf_counter() - overall:.1f}s", flush=True)


SCREEN_FILENAME = "screen.json"


def pilot_subset(labeled: pd.DataFrame, n_studies: int, target: int = 3) -> pd.DataFrame:
    """Pick a small screening cohort that covers as many labels as possible.

    A uniform random 15 of these studies routinely leaves several findings with no
    positive case, and a label with no positives contributes nothing to macro AUC —
    so a random pilot screens on fewer labels than it appears to. This greedily adds
    the study that fills the most still-thin (label, class) cells, up to `target` each.

    The result is a deliberately label-enriched subset. That is the right bias for
    ranking conditions against each other, and the wrong one for quoting an absolute
    accuracy, which is why the screen only ever selects conditions and never reports
    a headline number.
    """
    if n_studies >= len(labeled):
        return labeled

    # Deterministic ordering so the same cohort yields the same pilot every time.
    order = sorted(
        labeled.index,
        key=lambda i: hashlib.sha256(labeled.at[i, "StudyInstanceUID"].encode()).hexdigest(),
    )
    positives = {label: 0 for label in LABELS}
    negatives = {label: 0 for label in LABELS}
    chosen: list[int] = []
    remaining = list(order)

    while len(chosen) < n_studies and remaining:
        best_index, best_gain = remaining[0], -1
        for index in remaining:
            gain = 0
            for label in LABELS:
                if labeled.at[index, label] == 1:
                    gain += positives[label] < target
                else:
                    gain += negatives[label] < target
            if gain > best_gain:
                best_index, best_gain = index, gain
        for label in LABELS:
            if labeled.at[best_index, label] == 1:
                positives[label] += 1
            else:
                negatives[label] += 1
        chosen.append(best_index)
        remaining.remove(best_index)

    return labeled.loc[chosen].reset_index(drop=True)


def load_screen(out_dir: Path) -> list[str]:
    """Condition names that survived screening, written by `evaluate.py --write-screen`."""
    path = out_dir / SCREEN_FILENAME
    if not path.exists():
        raise SystemExit(
            f"--sweep sane needs {path}, which does not exist yet. Produce it with:\n"
            "  run_medgemma_experiments.py --sweep all --pilot-studies 15\n"
            "  evaluate.py --write-screen"
        )
    screen = json.loads(path.read_text())
    passed = [name for name in screen.get("passed", []) if name in CONDITIONS]
    if not passed:
        raise SystemExit(
            f"{path} lists no surviving conditions. Loosen the screen thresholds "
            "(evaluate.py --screen-min-auc ...) or inspect the rejection reasons in it."
        )
    print(
        f"sane sweep: {len(passed)} of {len(screen.get('passed', [])) + len(screen.get('rejected', {}))} "
        f"conditions survived screening on {screen.get('n_studies')} studies "
        f"({path.name})",
        flush=True,
    )
    return passed


def dry_run(
    data_dir: Path,
    series: pd.DataFrame,
    uids: list[str],
    names: list[str],
    context_length: int,
) -> None:
    """Report the image and prompt-token cost of each condition without calling the model.

    Image specs, not prompts, are what can overrun the context window, so this is the
    check to run before committing GPU time to a new rendering strategy.
    """
    seen: dict[str, ImageSpec] = {}
    for name in names:
        spec = CONDITIONS[name].image
        seen.setdefault(spec.key(), spec)

    print(
        f"\ndry run over {len(uids)} studies, {TOKENS_PER_IMAGE} tokens/image, "
        f"server context {context_length:,}"
    )
    print(
        "token columns count image tokens only; instructions and per-image captions add "
        "roughly another 1-2k\n"
    )
    header = (
        f"{'condition':30s} {'series':>6s} {'img med':>8s} {'max':>5s} "
        f"{'imgtok med':>10s} {'imgtok max':>10s} {'%ctx':>6s} {'slices med':>11s} {'/img':>5s}"
    )
    print(header)
    print("-" * len(header))

    cache: dict[str, tuple[list[int], list[int], list[int]]] = {}
    for name in names:
        spec = CONDITIONS[name].image
        key = spec.key()
        if key not in cache:
            counts, series_counts, covered = [], [], []
            for uid in uids:
                stacks, allocated = plan_study_images(data_dir, series, uid, spec)
                counts.append(image_count(spec, allocated))
                series_counts.append(len(stacks))
                covered.append(slices_represented(spec, stacks, allocated))
            cache[key] = (counts, series_counts, covered)
        counts, series_counts, covered = cache[key]
        if not counts:
            continue
        median = int(np.median(counts))
        largest = max(counts)
        median_slices = int(np.median(covered))
        print(
            f"{name:30s} {int(np.median(series_counts)):6d} {median:8d} {largest:5d} "
            f"{median * TOKENS_PER_IMAGE:10,d} {largest * TOKENS_PER_IMAGE:10,d} "
            f"{100 * largest * TOKENS_PER_IMAGE / context_length:5.1f}% "
            f"{median_slices:11d} {median_slices / max(median, 1):5.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", default=str(here.parents[1] / "data" / "from_host"))
    parser.add_argument("--out-dir", default=str(here / "artifacts"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="not-checked")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--sweep",
        action="append",
        choices=sorted([*SWEEPS, "sane"]),
        help=(
            "named group of conditions; repeatable (default: all). 'sane' runs only "
            f"the conditions that survived screening, read from {SCREEN_FILENAME}"
        ),
    )
    parser.add_argument(
        "--condition", action="append", default=[], help="run specific conditions by name"
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--limit-studies", type=int, default=0)
    parser.add_argument(
        "--pilot-studies",
        type=int,
        default=0,
        help=(
            "run on a label-balanced screening subset of this many studies. Results are "
            "cached per study, so the full run reuses them rather than re-requesting"
        ),
    )
    parser.add_argument(
        "--min-studies",
        type=int,
        default=30,
        help="refuse to run on a cohort smaller than this (extraction is ongoing)",
    )
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report image and token cost per condition without calling the model",
    )
    parser.add_argument("--context-length", type=int, default=131072)
    args = parser.parse_args()
    if not args.sweep and not args.condition:
        args.sweep = ["all"]
    return args


def main() -> None:
    args = parse_args()
    if args.list_conditions:
        for name, condition in CONDITIONS.items():
            print(
                f"{name:32s} axis={condition.axis:9s} strategy={condition.strategy:22s} "
                f"guided={str(condition.guided):5s} fmt={condition.answer_format:12s} "
                f"T={condition.temperature} x{condition.samples} "
                f"images={condition.image.key()}"
            )
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
