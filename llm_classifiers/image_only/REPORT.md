# Image-only knee-abnormality experiment report

How accurately can `google/medgemma-1.5-4b-it` identify the twelve findings in
`data/from_host/documentation.md` **from MRI pixels alone**? Two rounds of experiments, 65
conditions, 13,432 logged requests.

## Headline

**Macro ROC AUC 0.689 (95% case-bootstrap CI 0.638–0.739)** on all 58 labeled studies, from the
rank-average of two conditions that fail on different findings:

- `v2_digit` — ask per finding "rate 0–5 how confident you are", generate **one token**, and take
  the expected value over the digit distribution in its logprobs. 0.659 alone, ~1s per request.
- `v2_two_stage_t07_x5` — describe the images in free text, score that description on an ordinal
  word scale, five sampled repeats averaged. 0.659 alone.

The best single condition is either of those at **0.659 (CI 0.603–0.714)**. Round one's best was
0.634, so single-condition gains from round two are real but modest and *not* statistically
separable; the ensemble gain over `v2_digit` is (+0.029, CI +0.004 to +0.056, p=0.020).

Effusion is now at **0.907** and fracture at **0.792**, but MCL remains at chance. For scale: the
text-only pipeline in `../text_only/` reaches 0.909 macro AUC on these same studies. Reports
*state* the findings; images only *show* them.

## Round two: what improved it, and what did not

Round one concluded the bottleneck was decoding and prompting, not pixels. Round two tested five
directions against that. Results on all 58 studies:

| Direction | Best condition | AUC | Verdict |
|---|---|---:|---|
| **Read the answer distribution** | `v2_digit` | **0.659** | **Best single, and cheapest** |
| **Average sampled repeats** | `v2_two_stage_t07_x5` | **0.659** | **Ties for best** |
| Background from reports | `v2_digit_background` | 0.649 | Roughly neutral |
| Structured question read-out | `v2_questions` | 0.633 | No better than free description |
| Word scale, parsed | `v2_words` | 0.611 | Works, but ties cap it |
| Revisit unguided two-stage | `v2_guided_off_two_stage` | — | **Rejected twice on yield** |
| **Few-shot examples** | `v2_fewshot2_yesno` | **0.491** | **At chance — no help at all** |

**Reading the distribution beat emitting a word.** A five- or seven-level verbal answer is mostly
*ties*, and AUC is invariant to how the words are mapped to numbers, so the granularity is the
whole story. Asking for one token on a 0–5 scale and taking the expected value over its logprobs
gives a continuous score from a single greedy request: no sampling, no parsing, and structurally
incapable of the constant-output collapse that dominated round one. It is also the fastest thing
in the sweep at ~1s per request, against ~14s for a two-stage description.

The same trick applied to words (`v2_words`, 0.611) needed a string-enum schema — unconstrained,
MedGemma opens with a markdown `**` and the scale word never lands in the answer position. Digits
need no constraint.

**Averaging sampled repeats worked, and for the same reason**: five samples break ties. It reaches
the identical macro AUC by a much more expensive route.

**Few-shot did not work.** Four variants — one or two labelled positives and equal negatives per
finding, drawn leave-one-out from the other 57 studies, shown in the plane that answers that
finding — all sit at 0.49–0.51 on the full cohort. This is a clean negative, not a noisy one: the
intervals are tight (0.443–0.541) and it held from pilot to full run. Showing MedGemma-4B labelled
example knees does not help it judge a new one.

**Report-derived background was roughly neutral** (+0.02 on two-stage, −0.01 on digit, neither
significant). The 13-question structured read-out worked but beat nothing, and combining it with
the word scale collapsed completely.

**The unguided two-stage is settled**: rejected on yield twice (0.87 both times), even after
quadrupling its token budget. Its failures are empty responses, not truncation.

### Why the ensemble helps

The two leaders fail on *different* findings, which is why averaging their within-label ranks
gains more than either gains alone:

| Finding | pos | `v2_digit` | `v2_two_stage_t07_x5` | ensemble |
|---|---:|---:|---:|---:|
| Effusion | 35 | 0.843 | 0.873 | **0.907** |
| Fracture | 18 | 0.742 | 0.701 | **0.792** |
| Lateral OA | 11 | 0.673 | 0.769 | 0.744 |
| Synovitis | 27 | **0.769** | 0.602 | 0.729 |
| Contusion | 19 | 0.713 | 0.644 | 0.703 |
| Medial OA | 15 | 0.572 | **0.798** | 0.698 |
| Lateral Meniscus | 23 | 0.642 | 0.676 | **0.689** |
| Baker's | 12 | 0.649 | 0.645 | **0.687** |
| Medial Meniscus | 26 | 0.618 | 0.597 | 0.619 |
| PF OA | 21 | 0.584 | 0.615 | 0.607 |
| ACL | 24 | **0.609** | 0.521 | 0.603 |
| MCL | 9 | 0.499 | 0.469 | 0.490 |

The digit scale rescued exactly what round one was worst at — ACL 0.492 → 0.609, fracture
0.516 → 0.742, synovitis 0.581 → 0.769 — while the two-stage description remains better at
osteoarthritis grading. MCL is still chance under everything tried; with 9 positives it is also
the least measurable.

The ensemble members were fixed as "the top condition from each of the two distinct families"
before testing, not searched. A wider search over pairs and triples of the top eight does reach
0.703, but that maximum is selected over 84 combinations on 58 cases and should not be believed.

## Search space

Round one: 43 conditions, each exactly one field's change from a reference condition
(`ref_joint_definitions`: all twelve findings in one guided request, greedy, 16 images). Round two
added 22 more.

| Axis | n | What varied |
|---|---:|---|
| prompt | 6 | joint (plain / definitions / checklist / anti-degeneracy), one request per finding (JSON or single-token Yes-No scored from logprobs), two-stage read-then-score |
| guided | 4 | `response_format: json_schema` on vs off, with tolerant parsing |
| format | 5 | answer scale: 0–1 float, 0–100 integer, seven-level ordinal enum |
| sampling | 5 | temperature 0.3/0.6/1.0, `top_p` 0.80/0.95, 1/3/5 averaged seeded samples |
| image | 7 | series plan, 2/4/8/9 slices per series, 448 vs 896 px, percentile vs DICOM windowing, montage |
| downsample | 9 | fixed vs proportional sampling, slab MIP, slab mean, montage at 16/25, full-depth coverage |
| context | 6 | study-level image budget from 8 to 256 images (2k–65k image tokens) |
| v2 guidance | 5 | report-derived background block; 13-question structured read-out before scoring |
| v2 verbal | 5 | 0–5 digit or 5-word scale read as an expected value over the answer token's logprobs; per-finding view targeting |
| v2 averaging | 4 | five seeded samples at temperature 0.7 across four strategies |
| v2 few-shot | 4 | 1–2 labelled positives and equal negatives per finding, leave-one-out |
| v2 revisit | 1 | unguided two-stage with a quadrupled token budget |

The downsample and context axes exist because this archive's tail is severe: series reach 320
slices and studies 589, so at ~256 tokens per image, sending everything would cost ~40k tokens for
the median study, 151k for the largest, and would overrun the 128k window for about 2% of them.

The `background` block is derived from the 4,349 **unlabeled** training reports; the 58 labelled
studies never contribute to a prompt. Few-shot is the one exception to "no labels in prompts", and
those labels always belong to other patients — see Limitations.

## Method

Both rounds used the same protocol:

1. **Screening pilot.** Every condition on 15 studies chosen greedily to cover as many labels as
   possible — each finding got ≥3 positives and ≥3 negatives, so all twelve contribute to the
   macro average.
2. **Screen.** Conditions kept only if they parsed reliably, produced a non-degenerate ranking,
   and had a macro-AUC bootstrap **upper** bound clearing 0.60 (`--screen-keep-if-ci-clears`).
3. **Full run.** Survivors on all 58 studies. Responses are cached per study and image spec rather
   than per cohort, so the pilot's requests are inherited rather than repeated.

| | conditions | pilot | screen kept | full run | total |
|---|---:|---:|---:|---:|---:|
| Round one | 43 | 1,320 req / 34 min | 10 | 1,806 req / 13 min | 3,126 |
| Round two | +22 | 2,835 req / 37 min | 23 | ~7,500 req / 78 min | 13,432 |

**99.8% parse yield overall** (25 failures in 13,432). Screening avoided roughly 60% of the
unscreened grid across both rounds.

## Results on all 58 studies

| Condition | Macro AUC | 95% CI | flat | s/req |
|---|---:|---:|---:|---:|
| **ensemble: `v2_digit` + `v2_two_stage_t07_x5`** | **0.689** | 0.638–0.739 | — | — |
| `v2_digit` — per finding, 0–5 scale, expected value over one token's logprobs | 0.659 | 0.603–0.714 | 0.00 | 2.2 |
| `v2_two_stage_t07_x5` — describe, score on words, 5 samples averaged | 0.659 | 0.607–0.713 | 0.02 | 6.9 |
| `v2_digit_t07_x5` — digit scale, 5 samples averaged | 0.653 | 0.596–0.707 | 0.00 | 0.4 |
| `v2_digit_background` — digit scale plus report-derived background | 0.649 | 0.600–0.696 | 0.00 | 1.3 |
| `v2_likert_t07_x5` — joint prompt, ordinal words, 5 samples | 0.648 | 0.598–0.697 | 0.03 | 3.5 |
| `v2_two_stage_background` — describe with background, ordinal words | 0.640 | 0.605–0.691 | 0.29 | 13.5 |
| `format_likert_two_stage` — round one's best | 0.634 | 0.600–0.673 | 0.29 | 11.8 |
| `v2_questions` — 13-question read-out, then score | 0.633 | 0.577–0.705 | 0.28 | 11.9 |
| `v2_digit_targeted` — digit scale, only the plane that answers the finding | 0.627 | 0.566–0.684 | 0.00 | 0.6 |
| `v2_words` — 5-word scale via string-enum schema, read from logprobs | 0.611 | 0.555–0.666 | 0.00 | 1.5 |
| `prompt_two_stage` | 0.610 | 0.575–0.657 | 0.29 | 13.6 |
| `v2_words_targeted` | 0.609 | 0.554–0.660 | 0.00 | 0.3 |
| `format_likert_checklist` | 0.609 | 0.572–0.690 | 0.57 | 7.5 |
| `sample_t10_x3` | 0.609 | 0.555–0.664 | 0.00 | 4.7 |
| `prompt_joint_prevalence_free` | 0.597 | 0.541–0.653 | 0.45 | 7.9 |
| `sample_t06_x3` | 0.575 | 0.513–0.638 | 0.31 | 4.7 |
| `sample_t06_x5` | 0.572 | 0.498–0.644 | 0.09 | 3.5 |
| `v2_questions_background` | 0.541 | 0.487–0.602 | 0.07 | 11.7 |
| `prompt_binary_logprob` — Yes/No per finding | 0.540 | 0.482–0.595 | 0.00 | 1.0 |
| `sample_t06_p80_x3` | 0.509 | 0.445–0.569 | 0.43 | 4.8 |
| `v2_fewshot1_yesno` — 1 positive + 1 negative example | 0.492 | 0.443–0.541 | 0.00 | 6.6 |
| `v2_fewshot2_yesno` — 2 positive + 2 negative examples | 0.491 | 0.449–0.534 | 0.00 | 11.8 |
| `prompt_binary_json` — one JSON probability per finding | 0.491 | 0.454–0.530 | 0.00 | 1.1 |

`flat` is the fraction of studies given one identical value for all twelve findings. It is printed
beside AUC deliberately: a condition can look plausible and have ranked almost nothing. Note that
every logprob-scored condition has `flat = 0.00` by construction.

### Paired contrasts against the best single condition

5,000 paired case-bootstrap resamples over studies, all on the same 58 cases.

| vs `v2_digit` | Δ macro AUC | 95% CI | p |
|---|---:|---:|---:|
| ensemble with `v2_two_stage_t07_x5` | **+0.029** | +0.004 to +0.056 | **0.020** |
| `v2_two_stage_t07_x5` | −0.000 | −0.050 to +0.053 | 0.987 |
| `v2_digit_t07_x5` | −0.006 | −0.022 to +0.010 | 0.437 |
| `v2_digit_background` | −0.011 | −0.043 to +0.024 | 0.510 |
| `v2_likert_t07_x5` | −0.011 | −0.063 to +0.044 | 0.681 |
| `v2_two_stage_background` | −0.019 | −0.066 to +0.049 | 0.650 |
| `format_likert_two_stage` (round one's best) | −0.025 | −0.077 to +0.037 | 0.412 |
| `v2_questions` | −0.026 | −0.071 to +0.028 | 0.372 |
| `v2_digit_targeted` | −0.033 | −0.077 to +0.010 | 0.132 |
| `v2_words` | −0.048 | −0.099 to +0.002 | 0.063 |
| `prompt_joint_prevalence_free` | −0.062 | −0.110 to −0.012 | 0.014 |
| `sample_t06_x3` | −0.084 | −0.154 to −0.012 | 0.022 |
| `v2_questions_background` | −0.118 | −0.193 to −0.035 | 0.005 |
| `prompt_binary_logprob` | −0.119 | −0.194 to −0.048 | 0.001 |
| `v2_fewshot1_yesno` | −0.168 | −0.242 to −0.089 | <0.001 |
| `v2_fewshot2_yesno` | −0.168 | −0.237 to −0.101 | <0.001 |
| `prompt_binary_json` | −0.168 | −0.233 to −0.102 | <0.001 |

Read this carefully. **The improvement from round one's best to round two's is not statistically
separable** (−0.025, p=0.41): 58 studies cannot resolve a 0.03 difference in macro AUC. The whole
leading group, spanning 0.63 to 0.66, is one indistinguishable cluster.

Three things *are* separable, and they are the report's real conclusions: the ensemble beats its
own members (p=0.020); few-shot and single-probability-per-finding are decisively worse than the
leaders (p<0.001); and `top_p = 0.80` hurts.

Prefer `v2_digit` within the leading cluster not because it measured highest but because it is
6–30× cheaper, cannot degenerate, and needs no output parsing.

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

**4. Per-finding requests are not inherently worse — the answer format was.** Round one concluded
that twelve independent looks lose the cross-condition context a joint prompt provides, because
both `binary_*` conditions sat at the bottom. Round two refutes that: `v2_digit` is also one
request per finding and is the best single condition. What sank the round-one versions was asking
for a free-form probability (`prompt_binary_json`, 0.491) or a hard yes/no (`prompt_binary_logprob`,
0.540), not the per-finding framing itself.

**5. How you read the answer matters as much as what you ask.** A 5- or 7-level verbal answer is
mostly ties, and AUC is invariant to the word→number mapping, so granularity is the whole story.
The same question, answered on a 0–5 scale and scored as the expected value over the answer token's
logprob distribution, is the best single condition in the sweep and the cheapest. Under a string-enum
schema the same trick works for words, though unconstrained the model opens with a markdown `**` and
the scale never reaches the answer position.

**6. Few-shot did not help this model.** Four variants, 1–2 labelled positives and equal negatives
per finding drawn leave-one-out and shown in the plane that answers that finding, all landed at
0.49–0.51 with tight intervals. This is a clean negative that held from pilot to full run.

**7. Report-derived background is roughly neutral** (+0.02 on two-stage, −0.01 on digit, neither
significant), and the 13-question structured read-out beat nothing it was compared against. Telling
the model what radiologists look at did not make it see better.

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
- **Pilot numbers regress.** `v2_digit` measured 0.741 on the 15-study screening pilot and 0.659 on
  all 58. Conditions are selected on the pilot, so its numbers are optimistic by construction; only
  the full-cohort column should be quoted.
- **The ensemble is exploratory.** Its members were fixed before testing as "the best condition from
  each of the two distinct families", and it beats them at p=0.020 — but it is still a combination
  chosen after seeing which families existed, on the same 58 cases.
- **`v2_digit` compresses effusion.** It scores 0.843 on effusion against the two-stage's 0.873,
  and its advantage is on focal findings. If effusion is the target, do not use it alone.
- **Few-shot is not zero-shot.** Those conditions place other patients' labels in the prompt. The
  test study is always excluded, so it is leave-one-out rather than leakage, but the resulting
  numbers describe few-shot performance *given* a labelled pool. They are reported alongside the
  zero-shot ones because they lost, not because they are the same kind of measurement.

## If continuing

The evidence points at decoding and answer format, not imaging or instruction. Worth trying,
roughly in order:

1. **Per-finding two-stage scored from logprobs.** The two leaders are per-finding-digit and
   joint-two-stage; the cell they define — describe the images for *one* finding, then score that
   description on a digit scale — has not been run, and both parents are top of the table.
2. **A wider ensemble, evaluated honestly.** Rank-averaging three conditions reaches 0.703 here, but
   that maximum was selected over 84 combinations. Fix the members a priori and test them on
   labelled studies held out from all of this.
3. **Calibrate per finding.** The best condition differs by finding — digit for ACL, fracture,
   synovitis and contusion; two-stage for osteoarthritis and effusion. A per-finding pick would beat
   both, but choosing it on these 58 cases would be overfitting; it needs a held-out set.
4. **Effusion is usable now** at 0.907 from the ensemble. Fracture at 0.792 and synovitis at 0.769
   are the round-two surprises — both were near chance before.
5. **MCL is hopeless so far** (0.49 under everything) and has only 9 positives. Either accept it or
   treat it as a targeted-view problem needing something other than prompting.
6. **Fine-tuning, or a purpose-built vision model.** Two rounds of prompt engineering moved macro
   AUC from 0.63 to 0.69. That is real, but the remaining gap to the text-only 0.909 is a capability
   limit, not a prompting one.

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
