# Text-only knee-label experiment report

## Recommendation

Use the `joint_latent` prompt, which asks for all 12 labels in one request and asks the model to
infer the likely true abnormality despite possible errors or omissions in the report. Run it with
thinking disabled, temperature 0.6, and three independent seeded samples per report. Average the
three probabilities separately for each target. This was the best evaluated predictor at
**0.9088 macro ROC AUC** (case-bootstrap 95% CI 0.8782–0.9347).
The corresponding columns are named `joint_latent_first3_average__<label>` in
`artifacts/all_predictions.csv`.

Do not use a learned stacker on this dataset, and do not pay for long reasoning based on these
results. Five samples also did not improve over the original three.

## Approaches tested

All LLM approaches received the report and the same clinical definitions for the 12 targets.
For each target they returned an evidence status (`present`, `suspected`, `absent`,
`not_mentioned`, or `conflicting`), a probability, and confidence. The approaches differed as
follows:

| Identifier | What was done |
|---|---|
| `joint_extract` | One request predicted all 12 targets. The model was told to stay close to explicit report evidence and use indirect findings only when medically compelling. This approximates careful information extraction, while still producing probabilities rather than hard labels. |
| `joint_latent` | One request predicted all 12 targets. The model was explicitly told that the report could be inaccurate, incomplete, internally inconsistent, or templated, and that the desired probability concerned the actual abnormality—not merely whether the report asserted it. It could use secondary signs, injury mechanism, co-occurring findings, and statement reliability. Here “latent” refers to the unobserved true condition behind a noisy report, not a learned latent representation. |
| `individual_latent` | The same actual-abnormality task as `joint_latent`, but run in 12 separate requests, with each request seeing the full report and focusing on exactly one target. This tested whether joint processing caused interference or instead provided useful cross-condition context. |
| `two_stage_reasoning` | A first request used native Qwen reasoning and produced one bounded rationale per target. A second, thinking-disabled request received the report plus those rationales and returned guided JSON probabilities for all 12 targets. |

Each LLM configuration was sampled three times with distinct reproducible seeds at nonzero
temperature, and its primary prediction was the mean probability. Two additional `joint_latent`
samples tested whether averaging five instead of three helped.

The non-prompt comparisons were:

- **Rank ensembles:** convert each method's scores to within-target ranks, then average ranks.
- **Per-condition stack:** for each target, fit a separate regularized logistic model using only
  that target's probabilities, confidence, evidence status, and across-sample disagreement from
  the LLM approaches.
- **Cross-condition stack:** fit the same type of model but allow outputs for all 12 targets as
  predictors of each target.
- **Hashed-text baseline:** regularized logistic regression on word and character n-gram hashing
  features from the report, without LLM features.

## Data and validation

- `train.csv` has 4,407 reports but only 58 rows have labels; those 58 have all 12 labels.
- Positive counts range from 9 (MCL) to 35 (Effusion).
- Raw LLM methods are evaluated on all 58 labeled rows because they never see labels in prompts.
- Learned methods use 10 repeats of stratified 5-fold out-of-fold prediction. Every scored row is
  excluded from the model fit that scores it.
- Confidence intervals and pre-specified method contrasts use 5,000 paired case bootstrap
  samples. They capture case-sampling uncertainty but not prompt-selection uncertainty.
- This is still an exploratory test set: prompt and method choices were developed against these
  same 58 outcomes, so an untouched labeled set is needed for a confirmatory estimate.

## Main results

| Method | Macro AUC | 95% bootstrap CI | Notes |
|---|---:|---:|---|
| All targets jointly, infer actual abnormality (`joint_latent`), first 3 samples averaged | **0.9088** | 0.8782–0.9347 | Recommended |
| Joint rank ensemble | 0.9043 | 0.8732–0.9319 | No gain |
| All targets jointly, infer actual abnormality (`joint_latent`), 5 samples averaged | 0.9040 | 0.8722–0.9312 | Worse than first 3 |
| All-prompt rank ensemble | 0.9033 | 0.8712–0.9316 | No gain |
| Two-stage native reasoning | 0.8977 | 0.8666–0.9249 | Much slower |
| All targets jointly, stay close to explicit evidence (`joint_extract`) | 0.8947 | 0.8653–0.9217 | Weaker than actual-abnormality prompt |
| One target/request, infer actual abnormality (`individual_latent`) | 0.8829 | 0.8480–0.9141 | 12 requests/report |
| OOF per-condition stack | 0.8330 | 0.7885–0.8726 | Overfits despite target isolation |
| OOF cross-condition stack | 0.8041 | 0.7617–0.8417 | Worse than per-condition stack |
| OOF word/character hashing baseline | 0.5700 | 0.5077–0.6316 | Weak with 58 multilingual reports |

Key paired contrasts:

- Three-sample joint latent vs joint extraction: +0.0140 AUC, CI +0.0023 to +0.0259,
  bootstrap p=0.0212.
- Individual-condition vs three-sample joint latent: −0.0259, CI −0.0431 to −0.0099,
  p=0.0012.
- Two-stage reasoning vs three-sample joint latent: −0.0110, CI −0.0261 to +0.0029,
  p=0.1172.
- Cross-condition vs per-condition OOF stack: −0.0289, CI −0.0562 to −0.0028,
  p=0.0340.
- Five-sample vs first-three average: −0.0048, CI −0.0102 to −0.0003, p=0.0404.
  This sample-count comparison was added adaptively and should not be read as proof that exactly
  three samples is universally optimal.

## Per-target AUC for the recommended predictor

| Target | AUC | Target | AUC |
|---|---:|---|---:|
| ACL | 0.9737 | MCL | 0.9694 |
| Medial Meniscus | 0.9573 | Lateral Meniscus | 0.9112 |
| Medial OA | 0.9612 | Lateral OA | 0.8878 |
| PF OA | 0.9022 | Effusion | 0.8714 |
| Synovitis | 0.7742 | Baker's | 0.9357 |
| Contusion | 0.8839 | Fracture | 0.8771 |

Synovitis is the clearest remaining weakness. The reports' inaccurate or incomplete statements
also particularly affect Effusion and Contusion, but modeling the condition as a latent truth
rather than literal extraction improved overall ranking.

## Reasoning and runtime

The valid reasoning test used native Qwen reasoning with `--reasoning-parser qwen3`, followed by a
bounded 12-line rationale and a separate thinking-disabled guided-JSON scorer. Across 174 calls:

- Mean native reasoning-stage completion was 5,516 tokens (range 3,480–9,075); none hit 10k.
- Mean end-to-end request latency was 161.2 seconds versus 16.3 seconds for three-sample joint
  latent requests.
- Total completion tokens were 1,046,860 versus 86,917 for the three-sample joint latent run—about
  12× as many—while AUC was lower by 0.0110.
- Reasoning improved Effusion AUC (+0.029) but did not improve the macro score.

Native thinking combined directly with guided JSON was also smoke-tested. Although the reasoning
parser separated fields correctly, Qwen degenerated into repetition/whitespace and exhausted a
10k cap with invalid JSON. The two-stage design avoids that failure mode.

## Artifacts

- `artifacts/all_predictions.csv`: labels plus every evaluated prediction.
- `artifacts/metrics_macro.csv` and `metrics_by_label.csv`: aggregate and per-target AUCs.
- `artifacts/paired_bootstrap_tests.csv`: paired differences, intervals, and p-values.
- `artifacts/sampling_variability.csv`: stochastic-run variation.
- `artifacts/runtime_summary.csv`: request latency and token totals.
- `artifacts/raw/*.jsonl`: resumable raw responses, seeds, usage, and reasoning traces.
- `artifacts/raw/*partial.jsonl`: deliberately preserved diagnostics from interrupted/truncated
  reasoning designs; these are not used by the active evaluation.
