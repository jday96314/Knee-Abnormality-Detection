# Architecture 1 — frozen encoder, fixed hierarchical pooling

The plan's reference model and, so far, the best. Nothing is learned except a shallow
head: slices become a series descriptor by fixed statistics, series are grouped by imaging
plane, and the head maps the result to twelve independent logits.

```
slice -> frozen encoder -> [mean, p90, max] within series
      -> mean and max within plane -> concat over planes (+ presence flags)
      -> LayerNorm -> shallow head -> 12 logits
```

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
| slice jitter (keep 70%) | 0.0029 | slightly **harmful** |
| dropout 0.1 vs 0.3 | 0.0028 | prefer 0.1 |
| series dropout 0.15 | 0.0014 | marginally positive |
| MixUp 0.4 | 0.0005 | no effect |

The plan expected slice-sampling jitter and series dropout to be the most valuable
augmentations available. On frozen features they are worth almost nothing, and jitter is
mildly negative — with fixed pooling over the whole stack there is no encoder being
regularised, so discarding slices only removes evidence. This should not be read as a
verdict on those augmentations for the fine-tuned architectures, where an encoder does
exist to regularise.
