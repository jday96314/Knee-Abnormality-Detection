# Architecture 1 — frozen encoder, fixed hierarchical pooling

The plan's reference model, and the best of the frozen architectures — later beaten by
architecture 3, which fine-tunes the encoder (0.4064 vs 0.4146). Nothing is learned here
except a shallow head: slices become a series descriptor by fixed statistics, series are
grouped by imaging plane, and the head maps the result to twelve independent logits.

It remains a useful ensemble member: `arch 1 + arch 3 + arch 4` is the best combination
found (0.3993 BCE), better than architecture 3 alone.

```
slice -> frozen encoder -> [mean, p90, max] within series
      -> mean and max within plane -> concat over planes (+ presence flags)
      -> LayerNorm -> shallow head -> 12 logits
```

## What was tested

A full grid: **2^5 = 32 configurations, run twice (once per backbone) = 64 runs.**

| Axis | Values tried | Values *not* tried |
|---|---|---|
| **backbone** | MRI-CORE ViT-B/16, OrthoFoundation ViT-L/16 | any non-medical encoder |
| **`hidden`** (head width) | 0 (linear), 256 | 512+, or >1 hidden layer |
| **`dropout`** | 0.1, 0.3 | 0.0, 0.5 |
| **`slice_keep`** (slice jitter) | 1.0, 0.7 | 0.5 or lower |
| **`series_dropout`** | 0.0, 0.15 | 0.3+ |
| **`mixup`** | 0.0, 0.4 | other alphas; CutMix |
| encoder feature head | `cls` | `patch_mean`, `patch_max` (extracted but unused here) |
| pooling statistics | fixed `[mean, p90, max]` | any other quantile, or learned pooling (that is architecture 2) |
| LR / weight decay / batch | 1e-3 / 1e-2 / 64 | — |
| `slice_dropout` / `noise` | 0.1 / 0.0 | — |
| epochs / seeds | 60 (cosine), 1 seed | multiple seeds |
| input resolution | 224 px | 336, 448 |

Everything is feature-level: the encoder is frozen and its per-slice outputs are cached, so
a configuration costs ~6 minutes and the full grid is affordable. No pixel-space
augmentation is possible in this design — that is what architecture 3 exists to test.

## Results (MRI-CORE `cls`, 224 px, 32 configurations)

| | val soft BCE | gold AUC |
|---|---:|---:|
| baseline: predict train mean | 0.4969 | — |
| **best configuration** | **0.4146** | 0.8260 |
| best gold AUC in the sweep | 0.4170 | **0.8358** |
| worst configuration | 0.4278 | — |

Best config: 256-unit hidden layer, dropout 0.1, no slice jitter, MixUp 0.4.

## Design notes

**Why all three pooling statistics.** The findings want different things from a stack. A
joint effusion is diffuse and shows up in the mean; a fracture line or a collateral
ligament tear is focal and only shows up in `p90`/`max`. Concatenating all three lets the
head choose per finding, which costs almost nothing and reproduces what the earlier
frozen-backbone sweep found (`meanmax` beat either statistic alone).

**Why absent planes carry a flag.** A study with no axial series and a study whose axial
series is unremarkable would otherwise produce identical zeros. The presence flags let the
head distinguish "not imaged" from "imaged and normal", which matters for Baker's cysts in
particular since those are an axial finding.

**Why the head is deliberately small.** The descriptor is already 13,827-dimensional
against 3,479 training studies. Capacity here buys overfitting, and the sweep bears that
out — a 256-unit hidden layer is worth 0.0038 BCE and nothing larger was tried because
nothing larger is justified.

## Augmentations

All feature-level, since the encoder is frozen and the features are cached. Measured
marginal effect on validation BCE:

| Augmentation | Spread | Verdict |
|---|---:|---|
| slice jitter (keep 70%) | 0.0029 | below noise floor — no measured effect |
| dropout 0.1 vs 0.3 | 0.0028 | below noise floor — no measured effect |
| series dropout 0.15 | 0.0014 | below noise floor — no measured effect |
| MixUp 0.4 | 0.0005 | below noise floor — no measured effect |

The "Verdict" column previously read these as slightly harmful / preferred / marginally
positive. That was over-reading: all four spreads sit at or under the ~0.003
reproducibility floor, so the directions are not distinguishable from run-to-run variation.

The plan expected slice-sampling jitter and series dropout to be the most valuable
augmentations available. On frozen features they are worth almost nothing, and jitter is
mildly negative — with fixed pooling over the whole stack there is no encoder being
regularised, so discarding slices only removes evidence. This should not be read as a
verdict on those augmentations for the fine-tuned architectures, where an encoder does
exist to regularise.

**Read the whole table as "no measured effect."** The reproducibility floor for this script
is about 0.003 BCE (see the top-level README): the best configuration scores 0.4146 inside
the sweep and 0.417628 standalone, the latter reproducing bit-for-bit across three repeats.
Every spread in the table above is at or below that, so the ordering of these four
augmentation axes is not information. What *is* information is that no augmentation setting
escapes a 0.004 band while fine-tuning the encoder moves the metric by 0.012.
