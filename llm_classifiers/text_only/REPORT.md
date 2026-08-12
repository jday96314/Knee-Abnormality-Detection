# Text-only knee-label experiment report

## Recommendation

Use the `joint_latent` prompt, which asks for all 12 labels in one request and asks the model to
infer the likely true abnormality despite possible errors or omissions in the report. Keep
thinking disabled and temperature at 0.6.

- **Cost/performance default:** average five independently seeded probabilities per target.
- **Maximum stability tested:** average ten probabilities. If only ROC AUC matters and the whole
  prediction batch is available, averaging each run's within-target percentile ranks was slightly
  better than probability averaging (0.9048 vs 0.9042), but the difference was very small.
- **Faster option:** request probabilities only, omitting evidence status and confidence. This
  reduced mean request latency from about 23.6 to 14.0 seconds and completion tokens by 66%, with
  five-run AUC decreasing from 0.9039 to 0.9007.

The original first-three-seed average scored **0.9088 macro ROC AUC** (case-bootstrap 95% CI
0.8782–0.9347), but the later ten-seed analysis showed that this prefix was unusually favorable.
Across all 120 three-of-ten subsets, mean AUC was 0.9016. It should therefore not be treated as the
expected gain from exactly three reruns. Those original prediction columns remain named
`joint_latent_first3_average__<label>` in `artifacts/all_predictions.csv`.

Do not use a learned stacker on this dataset, and do not pay for long reasoning based on these
results. No alternative prompt or sampling temperature produced a validated improvement.

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
- Logistic feature ablations also report mean AUC within the 50 held-out folds. This avoids
  penalizing a method merely because independently fitted folds place probabilities on slightly
  different scales when their predictions are pooled.
- Confidence intervals and pre-specified method contrasts use 5,000 paired case bootstrap
  samples. They capture case-sampling uncertainty but not prompt-selection uncertainty.
- This is still an exploratory test set: prompt and method choices were developed against these
  same 58 outcomes, so an untouched labeled set is needed for a confirmatory estimate.

## Main results

| Method | Macro AUC | 95% bootstrap CI | Notes |
|---|---:|---:|---|
| All targets jointly, infer actual abnormality (`joint_latent`), favorable first 3 seeds averaged | **0.9088** | 0.8782–0.9347 | Highest observed, but seed-subset analysis shows optimism |
| Joint rank ensemble | 0.9043 | 0.8732–0.9319 | No gain |
| All targets jointly, infer actual abnormality (`joint_latent`), first 5 samples averaged | 0.9040 | 0.8722–0.9312 | Near rerun-count plateau |
| All-prompt rank ensemble | 0.9033 | 0.8712–0.9316 | No gain |
| Two-stage native reasoning | 0.8977 | 0.8666–0.9249 | Much slower |
| All targets jointly, stay close to explicit evidence (`joint_extract`) | 0.8947 | 0.8653–0.9217 | Weaker than actual-abnormality prompt |
| One target/request, infer actual abnormality (`individual_latent`) | 0.8829 | 0.8480–0.9141 | 12 requests/report |
| OOF per-condition full-feature stack | 0.8320 | 0.7881–0.8714 | Overfits despite target isolation |
| OOF cross-condition full-feature stack | 0.7974 | 0.7537–0.8365 | Worse than per-condition stack |
| OOF word/character hashing baseline | 0.5700 | 0.5077–0.6316 | Weak with 58 multilingual reports |

Key paired contrasts:

- Three-sample joint latent vs joint extraction: +0.0140 AUC, CI +0.0023 to +0.0259,
  bootstrap p=0.0212.
- Individual-condition vs three-sample joint latent: −0.0259, CI −0.0431 to −0.0099,
  p=0.0012.
- Two-stage reasoning vs three-sample joint latent: −0.0110, CI −0.0261 to +0.0029,
  p=0.1172.
- Cross-condition vs per-condition OOF stack: −0.0346, CI −0.0620 to −0.0077,
  p=0.0112.
- Five-sample vs first-three average: −0.0048, CI −0.0102 to −0.0003, p=0.0404.
  This sample-count comparison was added adaptively and should not be read as proof that exactly
  three samples is universally optimal.

## Raw-probability refinement study

The follow-up held the basic all-target, thinking-disabled design fixed and varied prompt wording,
temperature, output schema, aggregation, and rerun count. Prompt/temperature comparisons used
five fixed samples per report:

| Configuration | What changed from `joint_latent` | Five-run macro AUC |
|---|---|---:|
| Baseline, temperature 0.6 | Original actual-abnormality prompt | **0.9039** |
| Probability-only, temperature 0.6 | Removed evidence status and confidence from the response | 0.9007 |
| Independent image-review framing, temperature 0.6 | Asked for the label an independent reviewer of the source MRI would assign | 0.8991 |
| Decomposed prompt, temperature 0.6 | Instructed the model to internally separate evidence, sentence reliability, secondary signs, and reconciliation | 0.8979 |
| Skeptical prompt, temperature 0.6 | Restricted indirect inference to diagnostically specific secondary signs | 0.8966 |
| Baseline, temperature 0.2 | Lower sampling temperature | 0.8965 |
| Baseline, temperature 1.0 | Higher sampling temperature | 0.8960 |

None differed significantly from the temperature-0.6 baseline in paired case bootstraps. The
closest was probability-only (−0.0033; CI −0.0155 to +0.0084; p=0.579). Temperature 0.6 remained
the strongest of the tested values.

Ten baseline samples allowed rerun-count estimates over every seed subset rather than a single
prefix:

| Reruns averaged | Number of subsets | Mean subset AUC | SD across subsets |
|---:|---:|---:|---:|
| 1 | 10 | 0.8920 | 0.0037 |
| 2 | 45 | 0.8992 | 0.0038 |
| 3 | 120 | 0.9016 | 0.0036 |
| 5 | 252 | 0.9033 | 0.0031 |
| 8 | 45 | 0.9043 | 0.0019 |
| 10 | 1 | 0.9042 | — |

Most of the gain occurred by five runs. Additional runs mainly reduced seed sensitivity. For ten
runs, aggregation AUCs were: rank mean 0.9048, arithmetic probability mean 0.9042, trimmed mean
0.9035, logit mean 0.9035, and median 0.8983.

The independent-image-review prompt was extended to ten runs to test prompt diversity. A 20-call
ensemble averaging ten baseline and ten image-review probabilities scored 0.9052, only +0.0010
over ten baseline calls; the paired CI was −0.0044 to +0.0063 (p=0.697). Random seed-subset
comparisons likewise showed essentially no diversity advantage at matched request budgets. The
extra prompt is therefore not justified by the present evidence.

All refinements were selected and evaluated on the same 58 labeled reports. Their absolute
rankings and small differences need confirmation on untouched labels.

## Logistic-regression feature ablation

The initial per-condition stack used, for the target being predicted, all prompt variants'
probability estimates, confidence estimates, categorical evidence statuses, probability and
confidence standard deviations across three LLM samples, and status agreement. The following
ablation uses the same class-balanced L2 logistic regression (`C=0.10`) and the same 10× repeated
5-fold splits for every row of the table.

| Inputs to the per-target model | Mean held-out-fold macro AUC | Pooled OOF macro AUC |
|---|---:|---:|
| Recommended LLM probability, no fitted classifier | **0.9074** | **0.9088** |
| Logistic regression: recommended probability only | 0.9074 | 0.8598 |
| Logistic regression: probabilities from all prompt variants | 0.9043 | 0.8484 |
| All probabilities + evidence status | 0.9033 | 0.8457 |
| All probabilities + confidence | 0.8743 | 0.8477 |
| All probabilities + confidence + evidence status | 0.8779 | 0.8484 |
| Full feature set, also including sampling disagreement | 0.8511 | 0.8320 |

The mean held-out-fold AUC is the cleaner feature-impact comparison. A logistic transform of the
single probability preserves essentially the same ranking within each held-out fold (0.9074 vs
0.9074). Combining probabilities from the weaker prompt variants did not help. Evidence status
was nearly neutral, while reported confidence reduced mean held-out-fold AUC by about 0.030.
Adding sample-disagreement features reduced it further. On these 58 cases, the extra features
mostly duplicate the probability or give the classifier opportunities to overfit.

The lower pooled OOF AUCs for fitted models require care: each fold has a separately trained
calibrator, so its output scale differs, and pooling those fold-specific probabilities can disturb
global ranking even when within-fold ranking is unchanged. The conclusion is consistent under
either view—use the averaged raw LLM probability and do not fit a metadata stacker here.

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
- `artifacts/logistic_feature_ablation.csv`: per-label and macro held-out-fold AUCs for the
  probability/confidence/status ablations.
- `artifacts/sampling_variability.csv`: stochastic-run variation.
- `artifacts/runtime_summary.csv`: request latency and token totals.
- `artifacts/raw/*.jsonl`: resumable raw responses, seeds, usage, and reasoning traces.
- `artifacts/raw/*partial.jsonl`: deliberately preserved diagnostics from interrupted/truncated
  reasoning designs; these are not used by the active evaluation.
- `artifacts/refinement/config_metrics.csv`: five-run prompt and temperature comparisons.
- `artifacts/refinement/rerun_count_summary.csv`: seed-subset results by ensemble size.
- `artifacts/refinement/aggregation_methods.csv`: mean, median, trimmed, logit, and rank averaging.
- `artifacts/refinement/prompt_diversity_by_budget.csv` and `extended_prompt_ensembles.csv`:
  matched-budget prompt-diversity results and paired intervals.
- `artifacts/refinement/candidate_predictions.csv`: directly usable five-run, ten-run,
  probability-only, rank-averaged, and mixed-prompt candidate scores.
- `artifacts/refinement/runtime_summary.csv`: latency and completion-token costs by configuration.
