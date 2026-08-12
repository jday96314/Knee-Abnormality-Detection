# Text-only knee abnormality experiments

This directory evaluates four cached Qwen prompt variants on the 58 fully labeled reports.
Each request is sampled three times by default at temperature 0.6 with distinct reproducible
seeds; averaged probabilities are the primary predictions and per-sample results are retained.

- `joint_extract`: all 12 targets at once, close to explicit report evidence, thinking off.
- `joint_latent`: all targets at once, allowing inference through report errors/omissions, thinking off.
- `individual_latent`: one request per target, otherwise similar latent inference, thinking off.
- `two_stage_reasoning`: native reasoning plus a bounded 12-line rationale (10k safety cap,
  temperature 0.7, no repetition penalty), followed by guided JSON at temperature 0.6. Only
  the final rationale is passed to the scorer; native reasoning is retained for audit.

An earlier flag-only reasoning condition is preserved at
`artifacts/raw/joint_reasoning_pre_parser_diagnostic.jsonl`; it returned zero exposed reasoning
because vLLM had not yet been started with `--reasoning-parser qwen3`, so it is excluded from the
active evaluation. The two-stage condition is the valid reasoning test.

Run inference and evaluation from the repository root:

```bash
python llm_classifiers/text_only/run_llm_experiments.py --concurrency 48
python llm_classifiers/text_only/evaluate.py
```

The follow-up raw-ensemble refinement is reproducible with:

```bash
python llm_classifiers/text_only/run_llm_experiments.py \
  --experiments joint_latent --repeats 10 --temperature 0.6 \
  --concurrency 48 --skip-compile
python llm_classifiers/text_only/refine_raw_ensemble.py --repeats 5 --concurrency 48
python llm_classifiers/text_only/refine_raw_ensemble.py \
  --configs image_reviewer_t06 --repeats 10 --concurrency 48
python llm_classifiers/text_only/evaluate_refinement.py
```

Successful raw requests are appended immediately to `artifacts/raw/*.jsonl`, so interrupted
runs resume without repeating them. The evaluation writes per-label and macro metrics, raw and
out-of-fold predictions, bootstrap intervals, and runtime/token summaries under `artifacts/`.
Concurrency 48 is the default for short requests. Use `--concurrency 30` for long reasoning
requests to avoid KV-cache queueing on the referenced two-GPU server.

## Generate predictions

`predict.py` is the minimal production entry point. It predicts every input row, whether labeled
or unlabeled, using the selected cost/performance configuration: all 12 targets in one prompt,
actual-abnormality inference, thinking disabled, temperature 0.6, and the arithmetic mean of five
stochastic samples. The five choices are generated together in one vLLM request per report so the
server can reuse the prompt prefill; 12 concurrent reports was faster than 20 in a production-size
smoke test on the referenced server.

From the repository root:

```bash
python llm_classifiers/text_only/predict.py
```

By default this reads `data/from_host/train.csv` and writes:

- `llm_classifiers/text_only/predictions/text_only_predictions.csv`: one row per report, containing
  `source_row`, `StudyInstanceUID`, the 12 mean probabilities, and a `__sample_std` column for each
  target.
- `llm_classifiers/text_only/predictions/text_only_predictions.jsonl`: append-only raw sample
  cache. An interrupted run can be resumed with the same command.

For another CSV containing unique `StudyInstanceUID` and non-null `Report` columns:

```bash
python llm_classifiers/text_only/predict.py \
  --input path/to/reports.csv \
  --output path/to/predictions.csv
```

The script never reads label columns, so its default invocation includes all 4,349 unlabeled reports
as well as the 58 labeled reports in `train.csv`. `--samples 10 --concurrency 6` can be used for
maximum tested stability, at approximately twice the inference cost; five samples captured most of
the measured benefit.
Pre-specified method contrasts use paired case-level bootstrap tests of macro ROC AUC, while a
separate table reports sampling variation across LLM reruns.

Learned stackers use 10x repeated stratified 5-fold out-of-fold predictions. The primary stack uses only
features for the target condition; a cross-condition stack and a word/character hashing baseline
are reported as explicit generalization checks. Raw LLM methods need no fitting and are evaluated
directly on all labeled rows.
