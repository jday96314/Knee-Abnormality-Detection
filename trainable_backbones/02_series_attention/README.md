# Architecture 2 — frozen slice pooling, learned pathology attention over series

Keeps architecture 1's fixed `[mean, p90, max]` slice pooling and replaces only the
study-level aggregation: twelve learned query tokens, one per finding, cross-attend over
the study's 3-15 series tokens.

```
slices -> fixed [mean, p90, max] -> series token
       + plane / fluid-sensitivity / fat-suppression embeddings
       -> 12 pathology queries, cross-attention -> 12 logits
```

The motivation is that each finding can select its own sequence — ACL from sagittal, MCL
from coronal, Baker's from axial — which fixed pooling must average away. The plan rated
this "probably the best frozen-feature model".

## What was tested

A full grid: **2^4 = 16 configurations, run twice (once per backbone) = 32 runs.** The slice
pooling, features, split and augmentations are byte-identical to architecture 1 — only the
study-level aggregation differs, which is what makes the comparison a controlled one.

| Axis | Values tried | Values *not* tried |
|---|---|---|
| **backbone** | MRI-CORE ViT-B/16, OrthoFoundation ViT-L/16 | — |
| **`d_model`** | 128, 256 | 64, 512 |
| **`n_layers`** (attention depth) | 1, 2 | 3+ |
| **`dropout`** | 0.1, 0.3 | 0.0, 0.5 |
| **`series_dropout`** | 0.0, 0.15 | 0.3+ |
| `n_heads` | 4 | 1, 8 |
| number of queries | 12 (one per finding) | shared/grouped queries |
| slice pooling | fixed `[mean, p90, max]` | learned (that is architecture 4) |
| LR / weight decay / batch | 1e-3 / 1e-2 / 64 | — |
| `slice_keep` / `slice_dropout` / `noise` | 1.0 / 0.1 / 0.0 | — |
| epochs / seeds | 30 (cosine), 1 seed | multiple seeds |

Note that `slice_keep` and `mixup` — two of architecture 1's swept axes — are held fixed
here, so this grid is not a superset of that one.

## Results (MRI-CORE `cls`, 224 px, 16 configurations)

| | val soft BCE | gold AUC |
|---|---:|---:|
| baseline | 0.4969 | — |
| **best configuration** | **0.4357** | 0.7956 |
| best gold AUC in the sweep | 0.4389 | **0.8167** |
| worst configuration | 0.4664 | — |
| *architecture 1, for comparison* | *0.4146* | *0.8358* |

## It loses to fixed pooling, and not narrowly

**Architecture 2's best configuration is worse than architecture 1's worst** — 0.4357
against 0.4278. The two sweeps do not overlap.

That contradicts the plan's expectation, but it reproduces the earlier `frozen_backbones`
finding where attention-based MIL scored 0.09 macro AUC below fixed pooling on the 58 gold
studies. The notable part is that **60x more training signal did not reverse it**: the
earlier result could be dismissed as 58 studies being too few to fit attention, and that
explanation no longer applies at 3,479.

Two readings, not mutually exclusive:

1. Attention over 3-15 series is still a harder estimation problem than it looks, because
   the useful variation is between *studies* (which sequences exist at all) rather than
   within them, and 3,479 studies is not obviously many for learning twelve separate
   attention patterns.
2. The fixed `[mean, p90, max]` prior is simply strong. It already encodes "take the most
   extreme evidence anywhere in the study", which is most of what a focal-finding detector
   needs, and leaves learned attention little to add.

Within the sweep, smaller is better: `d_model=128` beat 256 everywhere, and the best
configurations used one attention layer rather than two — consistent with capacity being
the problem rather than the solution.

## What would change the verdict

This tests learned attention over *series*. The plan's architectures 3-5 add a partially
fine-tuned encoder and learned slice-level pooling, where the aggregation is trained
jointly with the features rather than bolted onto frozen ones. That is a materially
different proposition and this result does not speak to it.
