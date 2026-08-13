# Blending the text-only and image-only predictions

Combines `../text_only/predictions/text_only_predictions.csv` and
`../image_only/predictions/image_only_predictions.csv` into calibrated probabilities for
all 4,407 training studies, optimising **binary cross-entropy per finding**. ROC AUC is
reported alongside as an auxiliary metric.

```bash
python blend_experiments.py          # compare strategies, out-of-fold
python make_blended_predictions.py   # write the chosen blend for all 4,407 studies
```

| file | contents |
|---|---|
| `predictions/blended_predictions.csv` | 4,407 rows, same schema as the two inputs |
| `predictions/blended_predictions_meta.json` | fitted weights, out-of-fold metrics per finding |
| `results/methods.csv`, `results/per_label.csv` | every strategy tried |

## Why calibration dominates this problem

AUC is invariant to any monotone rescaling; BCE is not. That single difference reorders
everything measured in the earlier rounds:

| | BCE | AUC |
|---|---:|---:|
| predict the base rate | 0.6088 | — |
| **image raw** | **0.7222** | 0.7012 |
| text raw | 0.4778 | **0.8980** |

The image predictions rank respectably yet score **worse than a constant base-rate
prediction** under BCE, because nothing ever mapped their scores onto a probability scale.
Meanwhile `text raw` has the *best* AUC of any method tried and a mediocre BCE. Calibration
is not a refinement here; it is the main effect.

## The chosen method

A logistic model on the two logits:

```
logit(p) = a·logit(p_text) + b·logit(p_image) + c_finding
```

with **one `a` and one `b` shared across all twelve findings**, and a per-finding intercept
`c`. Fitted on the 58 gold-labelled studies. The fitted weights are `a = 0.733`,
`b = 0.358` — the image channel earns roughly a third of the text channel's influence.

**Out-of-fold: BCE 0.3508, AUC 0.8966**, under 10×5-fold stratified CV.

Sharing the slopes is the point. A separate two-parameter calibration per finding spends
those parameters on as few as 9 positives; pooling estimates them from ~700 rows while the
per-finding intercept still lets each finding keep its own base rate. That shows up
directly in the overfitting gap — 0.011 for the shared fit against 0.041 for the
per-finding one.

### Per finding (out-of-fold)

| Finding | BCE | AUC |
|---|---:|---:|
| MCL | 0.194 | 0.971 |
| Medial OA | 0.227 | 0.958 |
| Baker's | 0.270 | 0.875 |
| ACL | 0.276 | 0.929 |
| Medial Meniscus | 0.292 | 0.941 |
| Lateral OA | 0.315 | 0.836 |
| PF OA | 0.337 | 0.931 |
| Lateral Meniscus | 0.355 | 0.922 |
| Contusion | 0.433 | 0.854 |
| Effusion | 0.442 | 0.912 |
| Fracture | 0.496 | 0.842 |
| Synovitis | 0.574 | 0.790 |

## Everything that was tried

All figures out-of-fold, 10×5-fold stratified, macro-averaged over the twelve findings.

| Strategy | BCE | AUC | overfit gap |
|---|---:|---:|---:|
| per-finding image weight, pooled slope (C=1) | 0.3475 | 0.892 | +0.022 |
| **shared slope, text+image** ← chosen | **0.3508** | **0.897** | **+0.011** |
| per-finding image weight, pooled slope (C=0.3) | 0.3531 | 0.895 | +0.015 |
| blend: weighted logit, then Platt | 0.3534 | 0.880 | +0.041 |
| blend: Platt each channel, then weight | 0.3560 | 0.874 | +0.036 |
| blend: logistic on both logits, per finding (C=1) | 0.3576 | 0.876 | +0.051 |
| per-finding text *and* image weights | 0.3576 | 0.885 | +0.027 |
| shared slope, text only | 0.3590 | 0.883 | +0.010 |
| blend: logistic on both logits, per finding (C=0.1) | 0.3594 | 0.884 | +0.023 |
| text + Platt | 0.3627 | 0.863 | +0.032 |
| per-finding image weight, pooled slope (C=0.1) | 0.3675 | 0.897 | +0.010 |
| text + temperature (1 parameter) | 0.4348 | 0.875 | +0.020 |
| text raw | 0.4778 | 0.898 | — |
| image + Platt | 0.5573 | 0.622 | +0.023 |
| text + isotonic | 0.5947 | 0.856 | **+0.339** |
| prevalence baseline | 0.6099 | — | — |
| image raw | 0.7222 | 0.701 | — |

### What the comparison shows

**Calibration is worth ~0.115; blending is worth ~0.008.** Going from `text raw` (0.4778)
to `text + Platt` (0.3627) is by far the largest single step. Adding the image channel then
moves 0.3590 → 0.3508. If you only do one thing, calibrate the text predictions.

**Adding the image channel is not statistically established.** Shared-weight blend versus
text-only: −0.0082, CI −0.021 to +0.004, **p = 0.19**. With per-finding weights: −0.0116,
CI −0.024 to +0.001, p = 0.07. Both point the right way; neither clears significance at
n = 58. This matches what the AUC analysis found independently.

**Per-finding blend weights do not pay for themselves.** Against the shared weight:
−0.0034, CI −0.011 to +0.005, **p = 0.43**. The entire effect is one finding — Effusion
improves by −0.086 while eight of the other eleven get slightly worse. The fitted weights
are clinically sensible (Effusion +0.95, Medial OA +0.49, MCL +0.43, … Fracture +0.12,
all positive), but spending eleven extra parameters to help one finding is a poor trade on
58 studies, and the shared fit also has the better AUC and half the overfit gap.

*If effusion specifically matters*, the defensible exception is a per-finding weight for
that one finding and the shared weight elsewhere.

**Isotonic regression is the cautionary case.** In-sample 0.2561, out-of-fold 0.5947 — a
gap of +0.339. It looks like the best method available by a wide margin unless it is
cross-validated, and it is in fact worse than doing nothing at all.

**BCE and AUC disagree about the winner.** `text raw` has the best AUC of anything tried
(0.898) and one of the worst BCEs. If the downstream use is ranking rather than calibrated
probability, this whole exercise is close to a no-op — re-optimise for the metric that
matters.

## How the output file is built

Ten repeats of 5-fold CV give 50 fold-models. Every fold-model predicts all 4,407 studies.

* **The 4,349 unlabelled studies** take the mean over all 50 models.
* **The 58 gold-labelled studies** take *out-of-fold* predictions — each is scored only by
  the models that did not train on it. They fitted the blender, so an in-sample value would
  be optimistic and would silently contaminate any later evaluation that reuses this file.
* **`<finding>__sample_std`** is the disagreement between fold-models, which is a statement
  about how well-determined the blender's parameters are, not about the input models'
  own confidence.

## Limitations

* **58 studies, 9–35 positives per finding.** Every figure here carries wide intervals, and
  only the calibration effect is large enough to be unambiguous.
* **Selection and evaluation share a cohort.** The strategy was chosen by comparing these
  same out-of-fold numbers, so the reported 0.3508 omits selection uncertainty and is
  optimistic by an unknown amount. The `C = 1` regularisation was likewise picked by
  inspection rather than nested inside the CV.
* **The intercepts encode this cohort's prevalence.** The competition documentation warns
  explicitly that prevalence differs between the training, public and final splits. Those
  twelve intercepts are the parameters most likely to transfer badly; the two slopes should
  be more portable.
* **Reports do not exist at test time.** `test.csv` carries only `StudyInstanceUID`, so the
  text channel — and therefore this blend — cannot score the real test set. The output is
  useful as pseudo-labels for training an image-only student, which is what it was built
  for, not as a submission path.
* **The image channel inherits its own limitations**, documented in
  `../image_only/PSEUDOLABELS.md` — notably that its teacher is near-noise on ACL, MCL and
  Baker's.
