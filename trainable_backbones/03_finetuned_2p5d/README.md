# Architecture 3 — partially fine-tuned 2.5D encoder + hybrid pooling

**The best architecture in the ladder.** Three changes from architecture 1: the encoder is
unfrozen from the top down, through-plane context is added by a Conv1D over adjacent slice
*embeddings*, and the series representation concatenates the fixed `[mean, p90, max]`
statistics with a learned attention pool.

| | val soft BCE | gold AUC |
|---|---:|---:|
| baseline (train mean) | 0.4969 | — |
| architecture 1 — frozen, fixed pooling, **all** slices | 0.4146 | 0.8358 |
| architecture 4 — frozen, hierarchical Transformer | 0.4178 | 0.8432 |
| **architecture 3 — best (unfreeze 8, plane head)** | **0.4064** | **0.8630** |

## Configurations

| config | mode | unfreeze | val soft BCE | gold AUC | best epoch | minutes |
|---|---|---:|---:|---:|---:|---:|
| plane head, unfreeze 8 | plane | 8 | **0.4064** | **0.8630** | 8 | 30.1 |
| query head, unfreeze 4 | query | 4 | 0.4074 | 0.8599 | 9 | 22.4 |
| plane head, unfreeze 4 | plane | 4 | 0.4081 | 0.8469 | — | 32.0 |
| plane head, frozen encoder | plane | 0 | 0.4185 | 0.8283 | 14 | 14.9 |

The last row is the controlled ablation: identical sampling, augmentation, context module,
pooling and head, with only the encoder's gradients switched off.

## What it shows

**1. Fine-tuning is what buys the improvement, and it is the largest single effect in the
whole ladder.** 0.4185 frozen against 0.4064 with eight blocks trainable — **0.0121 BCE**,
and +0.035 gold AUC. For scale, every augmentation axis in architecture 1's 32-config sweep
moved the metric by at most 0.0038, and the architecture-1-vs-2 gap was 0.021. Unfreezing
more helps monotonically across the three settings tested (0 → 4 → 8 blocks:
0.4185 → 0.4081 → 0.4064), with diminishing returns, so 8 is a floor rather than an optimum.

**2. Fine-tuning wins *despite* seeing far less of each study.** This architecture reads 6
series x 4 slices = 24 slices per study per epoch; architecture 1 pools over every slice of
every series. The frozen ablation isolates that cost: 0.4185 here against 0.4146 for
architecture 1, so restricting to 24 sampled slices costs about 0.004. Fine-tuning then
buys back three times that. An adaptable encoder on a sample beats a fixed encoder on
everything — which is the opposite of the trade-off the frozen sweeps implied.

**3. The pathology queries stop losing once the encoder can move.** Query attention lost by
0.021 in architecture 2 and by 0.052 in architecture 4, both on frozen features. Here it is
*ahead* of the matched plane head (0.4074 vs 0.4081 at unfreeze 4) and well ahead on gold
AUC (0.8599 vs 0.8469). That reframes the earlier result: twelve queries learning per-finding
routing are not a bad idea, they are an idea that cannot work when the features underneath
them are frozen and were never trained to be routed. The difference is within one seed's
noise on BCE, but the AUC gap of 0.013 points the same way.

This is why `StudyHead` has a `mode` switch instead of inheriting the plan's instruction to
"keep the 12 pathology queries from model #2". The frozen evidence said drop them; testing
both is what surfaced that the frozen evidence did not generalise.

**4. Best epoch moves early.** 8-9 epochs for the fine-tuned configurations against 14 for
the frozen one. A trainable encoder reaches its best validation loss roughly twice as fast
and then overfits, which matters for anyone budgeting a longer run.

## Design notes

- **Through-plane context is a Conv1D over slice embeddings** (the plan's variant B), not
  neighbouring slices stuffed into the encoder's RGB channels. The encoder was pretrained
  with three identical channels; feeding it three different slices asks it to reinterpret
  an input convention it never learned.
- **Hybrid pooling concatenates rather than replaces.** The learned attention pool sits
  beside `[mean, p90, max]`, so the learned path can only add to a prior that architecture 1
  already validated.
- **Geometry is synchronised within a series and mild** (±6° rotation, ±5% translation,
  0.95-1.05 scale, gamma 0.8-1.25, occasional light noise). **No flips** — the VLM
  experiments measured flip TTA costing 0.061 AUC on ACL, because flipping a sagittal knee
  exchanges anterior for posterior.
- **Two learning rates.** The pretrained blocks get 0.1x the head's rate; one shared rate
  either wrecks the features or starves the head.

## Running it

```bash
python train.py --compare --epochs 20 --workers 6      # the four configurations
python train.py --subset 2,3,4 --epochs 20             # a subset of them
python train.py --mode query --unfreeze 8 --save-preds arch3   # single run + predictions
```

~1 min/epoch on an idle 4090 at 24 slices/study.

## Operational note

The first attempt hung: with `num_workers=10` and a fresh DataLoader per epoch, ~800 forks
alongside a live CUDA context deadlocked after 16 epochs — the main thread spinning at 99%
GPU with its workers idle and no output for 40 minutes. `persistent_workers=True` and 6
workers fixed it. Worth knowing before scaling this script up.

## Limitations

- **Four configurations, one seed.** The top three sit within 0.0017 BCE of each other,
  which is almost certainly inside seed noise; only the frozen-vs-fine-tuned gap (0.012) is
  comfortably outside it. The plane-vs-query ordering in particular should not be trusted
  from this evidence alone.
- **`unfreeze` and `mode` are not crossed.** The best configuration is `unfreeze=8, plane`
  and the runner-up is `unfreeze=4, query`; `unfreeze=8, query` was never run and is the
  obvious next cell.
- **24 slices per study per epoch** is a compute budget, not a modelling choice. Inference
  uses the same sparse sampling, where averaging several sampling passes would be the
  natural improvement.
- **Gold AUC rests on 58 studies**, where the earlier bootstrap interval was ±0.10 — wider
  than every AUC gap in the table.
