# Architecture 4 — hierarchical slice-Transformer + series-Transformer

The plan's most elaborate aggregator, and the closest thing in the ladder to the attached
architecture diagram: a small Transformer over slice embeddings within a series, a second
over series within a study, twelve pathology queries at the top, and a fixed-pooling
residual path so the Transformer does not have to rediscover `[mean, p90, max]`.

## Deliberate deviation: built on frozen features

The plan places this after architecture 3 and implies it inherits the fine-tuned encoder.
It is built here on **frozen** MRI-CORE features instead. The reason is that the only
question this architecture asks is *does Transformer aggregation beat fixed pooling* —
and architectures 1 and 2 answered adjacent questions on frozen features, with the same
split, features and augmentations. Building #4 the same way makes it the third point on one
controlled curve rather than a new system differing in two ways at once.

It also buys resolution: on frozen features every slice of every series is affordable
(up to 48 per series), where the fine-tuned path can only afford 4.

## Result

| | val soft BCE | gold AUC |
|---|---:|---:|
| baseline (train mean) | 0.4969 | — |
| architecture 1 — fixed pooling | **0.4146** | 0.8358 |
| **architecture 4 — best (plane head)** | 0.4178 | **0.8432** |
| architecture 2 — learned series attention | 0.4357 | 0.8167 |
| architecture 4 — best query head | 0.4694 | 0.7957 |

9 of 16 configurations completed; the run was stopped after the pattern became unambiguous
and the remaining configurations were competing for the GPU with architecture 3. The seven
missing cells are `slice_layers=2` crossed with query mode and with `series_layers=2` — all
in regions the completed cells already characterise.

| slice_layers | series_layers | mode | consistency | val BCE | gold AUC |
|---:|---:|---|---:|---:|---:|
| 1 | 2 | plane | 0.0 | **0.4178** | 0.8432 |
| 1 | 2 | plane | 1.0 | 0.4183 | 0.8335 |
| 2 | 1 | plane | 0.0 | 0.4200 | 0.8452 |
| 1 | 1 | plane | 1.0 | 0.4216 | 0.8421 |
| 1 | 1 | plane | 0.0 | 0.4219 | 0.8423 |
| 1 | 2 | query | 0.0 | 0.4694 | 0.7957 |
| 1 | 2 | query | 1.0 | 0.4826 | 0.8219 |
| 1 | 1 | query | 1.0 | 0.5059 | 0.7505 |
| 1 | 1 | query | 0.0 | 0.5064 | 0.7942 |

## What it shows

**1. Slice-level Transformer modelling does not pay for itself — but it is close, and it
wins on AUC.** 0.4178 against architecture 1's 0.4146 is a loss on the selection metric,
yet 0.8432 against 0.8358 is the best gold AUC of any frozen model here. Adding ordered
slice position and letting the model attend along the stack extracts something real; it
just does not show up as better-calibrated log-loss on teacher imitation. Given that the
teacher is itself only ~0.72 on gold, a model that imitates slightly worse while ranking
slightly better is not obviously the worse model.

**2. Pathology queries fail a third time, and here they fail catastrophically** — 0.4694 to
0.5064, i.e. two of the four query configurations are *worse than predicting the training
mean*. The evidence across the ladder is now consistent and one-directional:

| where | fixed/plane head | 12 pathology queries | gap |
|---|---:|---:|---:|
| architecture 2 (frozen) | 0.4146 (arch 1) | 0.4357 | 0.021 |
| architecture 4 (frozen, this) | 0.4178 | 0.4694 | 0.052 |
| earlier `frozen_backbones` MIL, 58 studies | — | — | ~0.09 AUC |

The plan expected the queries to be the winning ingredient. On this dataset they are the
single most reliably harmful choice tested. Note the one exception, in architecture 3:
once the *encoder* is trainable the queries stop losing — so the failure is specific to
learning per-finding routing on top of features that cannot adapt, not to the idea itself.

**3. Consistency regularisation does nothing.** Median plane-head BCE is 0.41995 with it
and 0.42000 without — a difference of 5e-5, and gold AUC is slightly *worse* with it
(0.8378 vs 0.8432). This was the plan's most distinctive training-side proposal for this
architecture, motivated by wanting the diagnosis invariant to slice sampling and missing
series. The likely reason it is inert: the augmentations it is meant to enforce invariance
to were already measured to be nearly irrelevant here (architecture 1's largest augmentation
effect was 0.0038). Penalising disagreement between two views is pointless when the two
views barely differ in the model's eyes. It also doubles training cost.

**4. Depth is irrelevant, in both Transformers.** slice_layers 1 vs 2: 0.41995 vs 0.42000.
series_layers 1 vs 2: 0.42160 vs 0.41805. The second is the larger of the two and still an
order of magnitude below the plane-vs-query gap.

## Running it

```bash
python train.py --backbone mri_core --sweep --epochs 30
python train.py --backbone mri_core --save-preds arch4   # single config, for common/ensemble.py
```

~9 s/epoch on an idle 4090; ~4.5 min per 30-epoch configuration.

## Limitations

- **Nine of sixteen cells.** The conclusions rest on marginals computed over an unbalanced
  design, so the "depth is irrelevant" claim in particular is weaker than the plane-vs-query
  claim, which is visible in every cell.
- **One backbone.** Architectures 1 and 2 were replicated on OrthoFoundation; this was not.
- **Slice position is the index, not physical z.** Millimetre spacing is not in the cache
  index, so a series acquired with 2 mm gaps and one with 5 mm look identical to the
  position embedding. That is a real handicap for the one component meant to exploit
  through-plane structure, and worth fixing before concluding the idea itself fails.
