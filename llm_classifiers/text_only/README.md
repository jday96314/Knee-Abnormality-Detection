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

Successful raw requests are appended immediately to `artifacts/raw/*.jsonl`, so interrupted
runs resume without repeating them. The evaluation writes per-label and macro metrics, raw and
out-of-fold predictions, bootstrap intervals, and runtime/token summaries under `artifacts/`.
Concurrency 48 is the default for short requests. Use `--concurrency 30` for long reasoning
requests to avoid KV-cache queueing on the referenced two-GPU server.
Pre-specified method contrasts use paired case-level bootstrap tests of macro ROC AUC, while a
separate table reports sampling variation across LLM reruns.

Learned stackers use 10x repeated stratified 5-fold out-of-fold predictions. The primary stack uses only
features for the target condition; a cross-condition stack and a word/character hashing baseline
are reported as explicit generalization checks. Raw LLM methods need no fitting and are evaluated
directly on all labeled rows.
