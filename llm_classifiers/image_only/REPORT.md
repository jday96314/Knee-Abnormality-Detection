# Image-only knee-abnormality experiment report

How accurately can `google/medgemma-1.5-4b-it` identify the twelve findings in
`data/from_host/documentation.md` **from MRI pixels alone**, zero-shot? No report text, no
labels, and no worked examples ever entered a prompt.

## Headline

**Macro ROC AUC 0.634 (95% case-bootstrap CI 0.600–0.673)** on all 58 labeled studies, using
`format_likert_two_stage`: describe the images in free text first, then score that description
onto a seven-level ordinal scale under schema-constrained decoding.

That single number is misleading on its own. Nearly all of the signal is joint effusion. The model
is **at chance on ACL tears** and near-chance on menisci and fracture — the findings that need
careful multi-slice reading.

For scale: the text-only pipeline in `../text_only/` reaches 0.909 macro AUC on these same 58
studies. The two are not competing methods — reports *state* the findings, images only *show*
them — but the gap indicates how little of the diagnosis this model recovers from pixels.

## Search space

43 conditions, each exactly one field's change from a reference condition
(`ref_joint_definitions`: all twelve findings in one guided request, greedy, 16 images).

| Axis | n | What varied |
|---|---:|---|
| prompt | 6 | joint (plain / definitions / checklist / anti-degeneracy), one request per finding (JSON or single-token Yes-No scored from logprobs), two-stage read-then-score |
| guided | 4 | `response_format: json_schema` on vs off, with tolerant parsing |
| format | 5 | answer scale: 0–1 float, 0–100 integer, seven-level ordinal enum |
| sampling | 5 | temperature 0.3/0.6/1.0, `top_p` 0.80/0.95, 1/3/5 averaged seeded samples |
| image | 7 | series plan, 2/4/8/9 slices per series, 448 vs 896 px, percentile vs DICOM windowing, montage |
| downsample | 9 | fixed vs proportional sampling, slab MIP, slab mean, montage at 16/25, full-depth coverage |
| context | 6 | study-level image budget from 8 to 256 images (2k–65k image tokens) |

The last two axes exist because this archive's tail is severe: series reach 320 slices and studies
589, so at ~256 tokens per image, sending everything would cost ~40k tokens for the median study,
151k for the largest, and would overrun the 128k window for about 2% of them.

## Method

1. **Screening pilot.** All 43 conditions on 15 studies chosen greedily to cover as many labels as
   possible — every finding got ≥3 positives and ≥3 negatives, so all twelve contribute to the
   macro average. 1,320 requests, 34 minutes.
2. **Screen.** Conditions were kept only if they parsed reliably, produced a non-degenerate
   ranking, and had a macro-AUC bootstrap **upper** bound clearing 0.60
   (`--screen-keep-if-ci-clears`). **10 of 43 survived.**
3. **Full run.** Survivors on all 58 studies. Because responses are cached per study and image
   spec rather than per cohort, the pilot's 15 studies were inherited: 1,806 new requests,
   13 minutes.

Total 3,126 logged requests, **99.5% parse yield**. Screening avoided roughly 3,500 requests, about
two-thirds of the unscreened grid.

## Results on all 58 studies

| Condition | Macro AUC | 95% CI | flat | s/req |
|---|---:|---:|---:|---:|
| **`format_likert_two_stage`** — describe, then score on an ordinal scale | **0.634** | 0.600–0.673 | 0.29 | 11.8 |
| `prompt_two_stage` — describe, then score as probabilities | 0.610 | 0.575–0.657 | 0.29 | 13.6 |
| `format_likert_checklist` — search-pattern prompt, ordinal scale | 0.609 | 0.572–0.690 | 0.57 | 7.5 |
| `sample_t10_x3` — reference prompt, temperature 1.0, 3 samples | 0.609 | 0.555–0.664 | 0.00 | 4.7 |
| `prompt_joint_prevalence_free` — forbid one value for all findings | 0.597 | 0.541–0.653 | 0.45 | 7.9 |
| `sample_t06_x3` — temperature 0.6, 3 samples | 0.575 | 0.513–0.638 | 0.31 | 4.7 |
| `sample_t06_x5` — temperature 0.6, 5 samples | 0.572 | 0.498–0.644 | 0.09 | 3.5 |
| `prompt_binary_logprob` — Yes/No per finding, scored from logprobs | 0.540 | 0.482–0.595 | 0.00 | 1.0 |
| `sample_t06_p80_x3` — temperature 0.6, `top_p` 0.80 | 0.509 | 0.445–0.569 | 0.43 | 4.8 |
| `prompt_binary_json` — one JSON probability per finding | 0.491 | 0.454–0.530 | 0.00 | 1.1 |

`flat` is the fraction of studies given one identical value for all twelve findings. It is printed
beside AUC deliberately: a condition can look plausible and have ranked almost nothing.

### Paired contrasts against the best condition

5,000 paired case-bootstrap resamples over studies, all on the same 58 cases.

| vs `format_likert_two_stage` | Δ macro AUC | 95% CI | p |
|---|---:|---:|---:|
| `prompt_two_stage` | −0.024 | −0.050 to +0.007 | 0.131 |
| `format_likert_checklist` | −0.025 | −0.076 to +0.059 | 0.670 |
| `sample_t10_x3` | −0.025 | −0.086 to +0.032 | 0.370 |
| `prompt_joint_prevalence_free` | −0.037 | −0.096 to +0.017 | 0.167 |
| `sample_t06_x3` | −0.059 | −0.132 to +0.008 | 0.088 |
| `sample_t06_x5` | −0.062 | −0.141 to +0.010 | 0.087 |
| `prompt_binary_logprob` | −0.094 | −0.165 to −0.028 | 0.007 |
| `sample_t06_p80_x3` | −0.125 | −0.206 to −0.051 | 0.001 |
| `prompt_binary_json` | −0.143 | −0.200 to −0.092 | <0.001 |

The top five are statistically indistinguishable from each other: on 58 studies this design cannot
separate a 0.634 condition from a 0.597 one. What it *can* separate is the top group from the
per-finding conditions and from `top_p = 0.80`, all of which are clearly worse. Read the table as
"two-stage and temperature-sampled joint prompts form a leading group", not as a ranking within it.

Averaging five samples did not beat three (−0.006 between them), and tightening `top_p` to 0.80 hurt
substantially (−0.125 against the best, −0.066 against the otherwise identical `sample_t06_x3`),
consistent with the collapse being a matter of the model concentrating on one hedged answer.

### Per-finding AUC

| Finding | best | `prompt_two_stage` | `sample_t10_x3` |
|---|---:|---:|---:|
| Effusion | **0.878** | 0.880 | 0.737 |
| Medial OA | 0.753 | 0.673 | 0.667 |
| Lateral OA | 0.680 | 0.634 | 0.670 |
| Contusion | 0.659 | 0.684 | 0.659 |
| PF OA | 0.644 | 0.607 | 0.631 |
| Lateral Meniscus | 0.600 | 0.576 | 0.550 |
| Synovitis | 0.581 | 0.564 | 0.571 |
| Medial Meniscus | 0.538 | 0.588 | 0.532 |
| Fracture | 0.516 | 0.504 | 0.501 |
| ACL | 0.492 | 0.542 | 0.586 |
| MCL | undefined | 0.480 | 0.580 |
| Baker's | undefined | 0.583 | 0.618 |

"undefined" means the condition emitted a constant score for that finding across all 58 studies —
reported as undefined rather than as 0.5, which would read as chance performance when nothing was
ranked at all.

The ordering is consistent and clinically legible: effusion, a large bright signal filling the
suprapatellar recess on any fluid-sensitive sequence, is the one thing the model reliably sees.
Compartmental osteoarthritis and bone contusion follow, both large-area signal changes. Ligament
and meniscal tears — small, focal, requiring the right slice and plane — are at or near chance.

## What actually mattered

**1. The imaging pipeline barely mattered; the prompt and answer format did.**

Every `ctx_*`, `down_*` and `image_*` condition was eliminated for emitting a constant vector. Slab
MIPs, montages, 8 images versus 190, 448 px versus 896, percentile versus DICOM windowing — all
hedged identically. Twenty-two of the 33 rejections were structural (16 flat, 15 producing no
ranking at all, one low yield); only one was rejected on accuracy alone.

This retires the concern that long-image-context handling was the bottleneck. It is not, because
the model does not use the extra pixels either way. Feeding it more of the study is not what
unlocks it.

**2. Guided decoding on a bare 0–1 probability is the dominant failure mode.**

The reference condition returned exactly the same probability for all twelve findings in every
study, producing no ranking whatsoever. `format_percent_int` collapsed to 0 everywhere and
`format_likert` to `"absent"` everywhere. The hedge is the model's own: it survived the schema, the
parser, and every image spec tried.

**3. Two things break the collapse, both cheap.**

- *Force a commitment before scoring.* Both two-stage conditions — write a structured reading of
  the images, then score that text — rank at the top. Making the model commit to observations in
  prose first appears to prevent the uniform hedge.
- *Use nonzero temperature.* `sample_t10_x3` differs from the fully degenerate reference **only**
  in temperature and sample count, and reaches 0.609 with `flat = 0.00`.

**4. Per-finding requests underperformed joint ones.** Both `binary_*` conditions sit at the bottom
despite never degenerating (`flat = 0.00`), and `prompt_binary_json` is indistinguishable from
chance. Twelve independent looks at the same images cost 12× the requests and lose the
cross-condition context that helps a joint prompt.

## Costs

Measured cost is ~310 prompt tokens per image once instructions and per-image captions are counted,
not the 256 the vision encoder charges. Latency scales steeply with image count: 8 images 1.8s,
64 images 14.8s, 128 images 43.2s, 190 images (58,244 tokens) 53.5s. The server accepted 190 images
in a single request without a per-request image cap or context error, so nothing in the ladder was
blocked by infrastructure — the conditions failed on model behaviour, not capacity.

`ctx_256img_64k` averaged 204s per request and 41,678 prompt tokens to produce a score that ranked
nothing. It is the clearest illustration of the report's main finding.

## Limitations

- **Selection and evaluation used the same 58 studies.** The surviving conditions were chosen by
  looking at these outcomes, then scored on them. The intervals cover case sampling only, not
  prompt-selection uncertainty. A confirmatory estimate needs labeled studies held out from the
  sweep, and this dataset has none beyond these 58.
- **58 studies is small**, with 9 to 35 positives per finding. Per-finding AUCs have wide
  intervals; treat the per-finding table as ordering, not measurement.
- **The screen's AUC rule ran on 15 studies**, where the macro-AUC interval was ~0.22 wide. It may
  have discarded a usable condition by chance. The structural rejections — constant output, low
  yield — are safe at any cohort size, and they account for the overwhelming majority.
- **One condition is unresolved.** `guided_off_two_stage` scored 0.640 on the pilot, nominally the
  best of all, but the screen dropped it on yield (0.867 — 13% of responses unparseable, including
  one that returned a detection box `{"box_2d": [...], "label": "fracture", "score": 0.99}` instead
  of a probability). It therefore only ever ran on 15 studies and is **not comparable** to the
  table above. Settling it costs ~10 minutes:
  `--condition guided_off_two_stage`.

## If continuing

The evidence points at prompting and decoding, not imaging. Worth trying, roughly in order:

1. Two-stage with nonzero temperature and several averaged samples — the two effects that
   independently break the hedge have not been combined.
2. Per-finding two-stage for the findings that need focused attention (ACL, menisci, fracture),
   keeping the joint prompt for effusion and osteoarthritis.
3. Effusion alone is at 0.878 zero-shot. If a usable signal is wanted now, that is where it is.
4. Fine-tuning, or a purpose-built vision model. A 4B general medical VLM being at chance on ACL
   tears is a capability limit, not a prompting one.

## Artifacts

- `artifacts/metrics_macro.csv`, `metrics_by_label.csv` — all scores, with `studies_scored`,
  `request_yield` and degeneracy columns.
- `artifacts/screen.json` — surviving conditions, per-condition rejection reasons, thresholds, and
  the cohort the screen was derived from.
- `artifacts/contrasts_vs_reference.csv` — paired case-bootstrap differences.
- `artifacts/predictions.csv` — every averaged prediction.
- `artifacts/raw/*.jsonl` — raw responses, seeds, usage, and the text of any cell that failed to
  parse.
- `artifacts/pilot.log`, `screen.log`, `full.log`, `final_eval.log` — run logs.

Reproduce with the three commands in `README.md`.
