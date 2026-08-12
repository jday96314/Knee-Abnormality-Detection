# Image-only zero-shot knee abnormality experiments (MedGemma)

Measures how accurately `google/medgemma-1.5-4b-it`, served by vLLM, can identify the twelve
findings in `data/from_host/documentation.md` **from pixel data alone**. No report text and no
labels ever enter a prompt, and no worked examples are supplied, so every condition is zero-shot.

This is the image counterpart to `../text_only/`, which classifies from the radiology reports.
The two are not comparable: reports state the findings, images only show them.

```bash
python llm_classifiers/image_only/run_medgemma_experiments.py --sweep all
python llm_classifiers/image_only/evaluate.py
```

## The extraction caveat

`rsna-knee-abnormality-detection.zip` is 265 GB and still unpacking, so only some of the 58
labeled studies are on disk. The runner admits a study only when every series listed in
`train_series.csv` is present and non-empty, hashes the resulting cohort, and stamps that hash on
every cached record. `evaluate.py` scores one cohort at a time and ignores records from another,
because a macro AUC over 17 studies and one over 58 are not the same quantity.

By default the runner refuses to start on fewer than 30 studies. Override deliberately:

```bash
python llm_classifiers/image_only/run_medgemma_experiments.py --sweep smoke --min-studies 15
```

With a small cohort several labels have no positives at all; their AUC is reported as undefined
rather than imputed, and the macro average is taken over the labels that are defined.

## Screening: don't pay full price for hopeless configurations

Most of the grid is not worth the whole cohort. Screen on a small label-balanced pilot first,
then run only the survivors:

```bash
# 1. pilot: every condition, 15 studies chosen to cover as many labels as possible
python llm_classifiers/image_only/run_medgemma_experiments.py --sweep all --pilot-studies 15

# 2. screen: writes screen.json and prints why each condition was dropped
python llm_classifiers/image_only/evaluate.py --write-screen

# 3. full run: only the conditions that survived
python llm_classifiers/image_only/run_medgemma_experiments.py --sweep sane
python llm_classifiers/image_only/evaluate.py
```

Nothing is wasted between steps. Cached responses are keyed by study and image spec, not by
cohort, so step 3 inherits every pilot request and only fills the gap.

The screen applies two kinds of rule, and they are not equally trustworthy:

| Rule | Default | Reliability |
|---|---|---|
| `--screen-min-yield` | 0.90 | **Safe at any N.** A condition that cannot be parsed cannot be used. |
| `--screen-max-flat` | 0.50 | **Safe at any N.** Fraction of studies given one value for all twelve findings. |
| `--screen-max-constant-labels` | 6 | **Safe at any N.** Findings whose score never varies across studies. |
| `--screen-min-auc` | 0.60 | **Noisy at N=15.** The 95% interval on macro AUC is roughly ±0.12 there, so a genuinely useful condition can drop below the line by luck. |

The structural rules do the reliable work: a condition emitting a constant vector has no
discriminative power, and no extra data will change that. The AUC rule is the one that can
discard something good. `--screen-keep-if-ci-clears` judges on the bootstrap upper bound instead
of the point estimate, keeping more conditions for the full run; raising `--pilot-studies`
tightens the interval directly. `evaluate.py --write-screen` prints the observed interval widths
so the size of that risk is visible rather than assumed.

`screen.json` records the surviving conditions, a per-condition rejection reason, the thresholds
used, and the cohort it was derived from, so a screening decision can be audited or re-litigated
later without re-running anything.

The pilot subset is deliberately label-enriched — a uniform random 15 studies routinely leaves
several findings with no positive case, and those labels then contribute nothing to macro AUC.
That bias is right for ranking conditions against each other and wrong for quoting an absolute
number, which is why the screen only ever selects conditions and never reports a headline figure.
`evaluate.py` also prints a per-condition `n` column and warns when conditions cover different
numbers of studies, so a pilot-only condition is never silently ranked against a full-cohort one.

## Axes

`--sweep {smoke,prompt,guided,format,sampling,image,downsample,context,sane,all}`, or
`--condition <name>` repeatedly.
`--list-conditions` prints the full grid. Every condition is a one-field change from the reference
condition `ref_joint_definitions`, so contrasts are interpretable.

**Round-two strategies** (`prompts_v2.py`) — added after round one showed the bottleneck was
decoding and prompting rather than pixels:

| Strategy | What it does |
|---|---|
| `binary_digit` | One request per finding: "rate 0–5 how confident you are", one token generated, score = expected value over the digit distribution in the logprobs. Continuous from a single greedy request, no parsing, cannot degenerate. **Best single condition.** |
| `binary_words` | The same read on a five-word scale (`absent … certain`), which needs a string-enum schema — unconstrained, the model opens with a markdown `**` and the scale never reaches the answer position. |
| `two_stage_questions` | A fixed 13-point structured read-out answered from the images, then scored as text. |
| `fewshot_yesno` / `fewshot_digit` | Labelled example studies for the finding in question, drawn leave-one-out from the other 57 and shown in the plane that answers it. **Did not work** — 0.49 on the full cohort. |
| `background=True` | Prepends context mined from the 4,349 *unlabeled* training reports: what readers of this collection comment on, how often, and how findings co-occur. Roughly neutral. |
| `view_targeted=True` | Shows only the plane that can answer the finding (ACL sagittal, MCL coronal, Baker's axial) via `FINDING_PLAN`. |

**Prompting strategy** (`prompts.py`)

| Strategy | What it does |
|---|---|
| `joint_plain` | All twelve findings in one request, label names only, no definitions. |
| `joint_definitions` | Adds an MRI-side definition of each finding and its usual confusers. The reference. |
| `joint_checklist` | Definitions plus an explicit seven-step search pattern to work through first. |
| `joint_prevalence_free` | Definitions plus an instruction forbidding an identical probability for every finding. |
| `binary_json` | Twelve separate requests, one finding each, JSON probability. |
| `binary_logprob` | Twelve separate requests, one-word Yes/No, scored as P(yes)/(P(yes)+P(no)) from the first token's logprobs. Never mis-parses and cannot hedge to a constant. |
| `two_stage` | Request one asks for a free-text structured reading of the images; request two scores that reading as text only. Images are seen only in stage one. |

**Guided decoding** — `guided=True` sends `response_format: json_schema`, which vLLM enforces.
`guided=False` sends nothing and relies on tolerant parsing. Note that the top-level `guided_json`
field is silently ignored by the server in use; `response_format` is the working path.

**Answer scale** — a bare 0-1 number is where the model hedges hardest, so `format_percent_int`
(integer 0-100) and `format_likert` (a seven-level ordinal enum mapped back to probabilities)
ask for the same judgement on scales it may commit to more readily.

**Sampling** — temperature 0/0.3/0.6/1.0, `top_p` 0.80/0.95, and 1/3/5 averaged samples. Repeats
use distinct seeds derived from `sha256(condition|study|label|repeat)`, so they are independent
but exactly reproducible and resumable.

**Rendering** — how a study becomes pictures dominates everything else on an image-only task, so
it is swept rather than assumed: which series are chosen (`sag_fluid`, `tri_fluid`,
`tri_fluid_plus_t1`, `all_series`), 2/4/8/9 slices per series, 448 vs 896 px, percentile vs DICOM
windowing, and individual slices vs one tiled montage per series.

Slices are ordered by `InstanceNumber`, falling back to position projected on the slice normal,
and sampled evenly from the central 70% of the stack; the end slices of a knee series are usually
off-joint. Images are letterboxed rather than stretched, since distorting a non-square series
changes apparent cartilage and meniscus geometry.

**Downsampling long series** — measured over this archive, not the doc's summary:

| | median | p90 | p99 | max |
|---|---|---|---|---|
| series per study | 5 | 7 | 10 | 14 |
| slices per series | 30 | 38 | 144 | 320 |
| total slices per study | 155 | 291 | 524 | 589 |

At 256 tokens per image, sending every slice costs ~40k tokens for the median study and 151k for
the largest — about 2% of studies would not fit the 128k window at all. So the reduction strategy
is a first-class axis, not an implementation detail:

| `sampling` | What it does |
|---|---|
| `fixed` | A set number of slices per series regardless of length. A 20-slice and a 320-slice acquisition get the same treatment — a 16× difference in through-plane coverage. |
| `proportional` | Roughly every `stride`-th slice, so sampling density is constant across studies and long series get proportionally more images, up to a per-series cap. |
| `slab_mip` | Partition the stack into contiguous slabs and take the per-pixel maximum of each. Same image count as `fixed`, but every slice contributes — the standard way to keep a small bright finding (effusion, cyst, marrow oedema on a fluid-sensitive sequence) that a sparse subsample steps straight over. |
| `slab_mean` | The same slabs averaged instead of maximised: less noise, but focal findings get diluted. The control for `slab_mip`. |

`layout="montage"` is the other lever, and the cheapest: one tiled image per series costs 256
tokens whatever it contains, so `down_montage25` shows the model ~100 slices for 1,024 tokens
where the reference shows 16 for 4,096. What it spends instead is per-slice resolution — 25 tiles
in an 896 px canvas leaves each slice ~179 px.

**Context length** — `ctx_8img_2k` through `ctx_256img_64k` take every slice of every series, so
the study-level image budget is the only binding constraint, and step it from 2k to 65k image
tokens (50% of the window). This is the axis that answers whether a 4B model actually uses a long
image context or comes apart well before its advertised limit. `MAX_IMAGES_HARD_CAP` (256) is a
safety rail so no condition can silently overrun the window.

Check the cost of any spec before spending GPU time on it:

```bash
python llm_classifiers/image_only/run_medgemma_experiments.py --sweep context --dry-run
```

This plans every study without decoding a pixel and reports images, prompt tokens, percentage of
context, and slices actually represented per image — the coverage side of what the token count
buys.

## Reading the output

`evaluate.py` writes `metrics_macro.csv`, `metrics_by_label.csv`, `contrasts_vs_reference.csv`,
`predictions.csv` and `summary.json`, and prints a ranked table. Two columns matter as much as AUC:

- **`flat`** — the fraction of studies where the condition gave every finding the same
  probability, and **`const`** — the number of findings whose probability never changed across
  studies. A condition can look reasonable and have produced no ranking at all.
- **`yield`** — the fraction of requests that parsed. A condition that answers 60% of the time is
  not comparable to one that always answers.

A constant score is reported as an undefined AUC, not as sklearn's 0.5, which would read as
chance performance when nothing was ranked.

Confidence intervals and the contrasts use 5,000 paired case-bootstrap resamples over studies.
They capture case-sampling uncertainty only. Conditions are selected by looking at these same
studies, so a confirmatory estimate needs labeled studies held out from the sweep.

If the reference condition is itself degenerate, every contrast against it would be undefined, so
the evaluator falls back to the best-scoring condition and says so. That fallback is a baseline
chosen after seeing the scores; it is not a pre-specified contrast and must not be read as one.

## Observed behaviour on a 5-study pilot

All 28 conditions were run end to end on five studies. The cohort is far too small for any
accuracy claim — several labels have no positives at all — so what follows is about failure
modes, not performance. Do not quote the pilot AUCs.

- Guided joint scoring on a bare 0-1 scale collapses: MedGemma returned exactly 0.5 for all
  twelve findings in all five studies. `format_percent_int` collapsed to 0 everywhere and
  `format_likert` to `"absent"` everywhere. The hedge is the model's own — it survives both the
  schema and the parser, and it appeared under every image spec tested.
- Making the model commit to something before scoring is what breaks the collapse. `two_stage`
  (write a reading, then score the text), `joint_prevalence_free` (forbid an identical value for
  every finding), and the per-finding conditions all produced varying output. `binary_logprob`
  cannot degenerate by construction, since the score is a token probability.
- This is why `flat` and `const` are printed next to AUC. Several conditions with a plausible
  headline number had, on inspection, ranked almost nothing.
- MedGemma 1.5 emits an optional thinking block delimited by `<unused94>`/`<unused95>`. The
  server is not started with a reasoning parser, so it arrives inside `content`; the parser strips
  it. Unguided conditions need a token budget covering the thinking block *and* the JSON, or the
  answer is truncated mid-object and the cell is lost.
- Unguided output is formatted unpredictably: bare fenced scalars (```` ```json\n0.85\n``` ````)
  and invented key names (`{"finding_present": 1.0}`) both occur. The parser accepts these, but
  only where the intent is unambiguous — with several numbers in free prose the cell is failed
  rather than guessed at.

### Context and downsampling, from a 2-study infrastructure check

Every condition on both axes returned successfully, so the ladder is runnable end to end:

| images | prompt tokens | latency |
|---:|---:|---:|
| 8 | 3,016 | 1.8s |
| 32 | 10,249 | 6.3s |
| 64 | 19,891 | 14.8s |
| 128 | 39,384 | 43.2s |
| 190 | 58,244 | 53.5s |

- **The server accepts very large image counts.** 190 images and 58k tokens in one request went
  through without a per-request image cap or a context error, so nothing in the ladder is blocked
  by infrastructure.
- **Cost is ~310 tokens per image, not 256**, once the instructions and per-image captions are
  counted. `--dry-run` reports image tokens only and says so.
- **Latency scales worse than linearly in wall-clock terms** and image preparation dominates at
  high counts — hence the on-disk PNG cache. Repeat requests over an identical image set returned
  in 1.7s against 37.8s cold, which is vLLM prefix caching.
- **More pixels did not fix the degeneracy.** Under guided decoding at temperature 0 every
  condition from 8 to 190 images still returned one constant value for all twelve findings, and
  the constant wandered (0.0, 0.1, 0.5) with no evident relation to image content. On this
  evidence the uniform hedge is a decoding-and-prompting failure, not the model being starved of
  image information — which is what makes the prompt and answer-scale axes the ones worth
  spending the larger cohort on.

## Caching and resumption

Successful records are appended to `artifacts/raw/<condition>.jsonl` immediately, keyed by
study/label/repeat, and reused only when the cohort hash and image spec match. Rendered PNGs are
cached under `artifacts/image_cache/`, so re-running a condition does not re-decode DICOMs.

Unparseable cells are recorded with `ok: false` and the offending text rather than being dropped,
so a failure can be diagnosed from the cache. They are retried on the next run; under greedy
decoding a parse failure is not retried within a run, since the same seed reproduces the same text.

## Requirements

`pydicom`, `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`, `python-gdcm` (the archive
mixes uncompressed, JPEG Lossless and JPEG 2000 transfer syntaxes), plus `pillow`, `openai`,
`pandas`, `numpy`, `scipy` and `scikit-learn`.
