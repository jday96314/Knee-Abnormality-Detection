# Vision models distilled from the pseudo-labelled studies

Trains image-only models to imitate the blended text+image soft targets from
`../llm_classifiers/blend`, following the architecture ladder in `planning/plan.md`.

One subdirectory per architecture family; everything shared lives in `common/` so the
evaluation protocol cannot drift between them.

## Protocol

Fixed before any results existed, and identical for every architecture:

| | |
|---|---:|
| train | 3,479 pseudo-labelled studies (80%) |
| validation | 870 pseudo-labelled studies (20%) |
| objective | **soft binary cross-entropy** — the selection metric |
| gold set | 58 studies, ROC AUC **reported only, never tuned against** |
| baseline | predict the training mean: **0.4969** val soft BCE |

The 58 gold studies are excluded from *both* halves, not merely from validation. They cost
1.3% of the training data and buy a measurement that no architecture choice has touched.
The plan asks for exactly this, and for good reason: the teacher itself only reaches 0.723
on those studies, so imitating it well is not the same as diagnosing well.

## Headline result

**A small head on frozen features reaches 0.836 gold ROC AUC from pixels alone.**

| | gold AUC | what it is |
|---|---:|---|
| **architecture 1 (best config)** | **0.836** | this work — pixels only |
| blended teacher it learned from | ~0.92 | text + image, needs reports |
| image-only VLM teacher | 0.723 | 2 models, TTA, ensembled |
| frozen probe trained on gold labels | 0.706 | 58 hard labels, no distillation |

Two things follow. First, the distillation premise holds: report-derived knowledge, which
does not exist at test time, has been transferred into a model that sees only images —
+0.11 AUC over the image-only VLM ensemble that helped produce the labels. Second,
**3,479 noisy soft targets beat 58 clean hard ones by a wide margin** (0.836 vs 0.706).

*Caveat.* The blender's parameters were fitted on the gold labels, so gold information
indirectly shaped the pseudo-targets. For AUC the effect should be nil — those parameters
are monotone within each finding and AUC is monotone-invariant — but it is not literally
zero, and a clean estimate would need studies held out from the blender too.

## Architecture comparison (MRI-CORE `cls`, 224 px)

| Architecture | configs | best val BCE | best gold AUC |
|---|---:|---:|---:|
| baseline (train mean) | — | 0.4969 | — |
| **1. fixed hierarchical pooling** | 32 | **0.4146** | **0.8358** |
| 2. learned pathology attention over series | 16 | 0.4357 | 0.8167 |

**Fixed pooling wins decisively. Architecture 2's *best* configuration is worse than
architecture 1's *worst*** (0.4357 against 0.4278) — the two sweeps do not overlap at all.

This contradicts the plan's expectation that per-finding series attention would be "the
best frozen-feature model", but it reproduces what the earlier `frozen_backbones` work
found on 58 studies, where attention-based MIL scored 0.09 below fixed pooling. The
interesting part is that 60x more training signal did not reverse it. Twelve query tokens
learning to attend over 3-15 series is evidently still the harder estimation problem, even
with 3,479 studies, and the fixed `[mean, p90, max]` prior is strong enough that learned
aggregation has little left to add.

## What actually moved the metric

Marginal medians over the 32 architecture-1 configurations:

| Axis | Setting | Median BCE | Spread |
|---|---|---:|---:|
| head capacity | 256 hidden vs linear | 0.4188 / 0.4226 | 0.0038 |
| slice jitter | keep 100% vs 70% | 0.4192 / 0.4221 | 0.0029 |
| dropout | 0.1 vs 0.3 | 0.4192 / 0.4220 | 0.0028 |
| series dropout | 0.15 vs 0.0 | 0.4207 / 0.4221 | 0.0014 |
| MixUp | 0.4 vs 0.0 | 0.4215 / 0.4210 | 0.0005 |

Read in order of effect size, and note the scale:

1. **Every augmentation axis is nearly irrelevant here.** The largest spread of any single
   choice is 0.0038, against an architecture gap of 0.021 and a gap to baseline of 0.082.
   The plan expected slice-sampling jitter and series dropout to be the most valuable
   augmentations; on frozen features with 3,479 studies they are worth almost nothing.
2. **Slice jitter actively hurts slightly** (0.4221 keeping 70% vs 0.4192 keeping all).
   With fixed pooling over the whole stack, discarding slices only removes evidence — there
   is no encoder being regularised, so the usual argument for it does not apply.
3. **A hidden layer helps a little**, which is the one place capacity earns its keep.
4. **MixUp does nothing** (0.0005), despite being the augmentation that most changes the
   training distribution.

The honest summary is that at this scale the architecture and the frozen features decide the
result, and the regularisation knobs are noise. That is worth knowing before spending
effort tuning augmentation for the fine-tuned architectures, where the picture may differ
because there is then an encoder to regularise.

## Layout

```
common/       dataset, protocol, feature extraction, augmentation
01_frozen_pooling/     architecture 1 — fixed [mean, p90, max] hierarchy
02_series_attention/   architecture 2 — 12 pathology queries over series
features/     per-slice frozen features (memory-mapped float16)
```

Slice cache: 819,076 slices at 224 px across all 4,407 studies, on local NVMe
(`KNEE_SLICE_CACHE`, default `/mnt/data01/knee_cache`). It is not beside the code because
the repository sits on a CIFS share that is near capacity and slower to read, and CIFS
cannot hold a symlink to redirect it.

## Reproducing

```bash
python common/dataset.py 224 40          # build the slice cache (I/O bound, ~2.5 h)
python common/features.py mri_core       # per-slice features (~5 min on one 4090)
python common/features.py orthofoundation
cd 01_frozen_pooling && python train.py --sweep
cd 02_series_attention && python train.py --sweep
```

## Limitations

- **One split, one seed.** Configurations differ by less than 0.005 BCE across most of the
  sweep, which is plausibly within seed noise. The architecture-1-vs-2 gap (0.021) is
  large enough to trust; the ranking *within* architecture 1's top ten is not.
- **Val BCE and gold AUC do not agree.** Architecture 1's best-BCE configuration is not its
  best-AUC configuration. Selection follows BCE as instructed; the AUC column is reported
  so the divergence is visible rather than hidden.
- **Gold AUC rests on 58 studies** with 9-35 positives per finding. The earlier frozen work
  measured a bootstrap interval of 0.10 AUC on this cohort, which is wider than every gap
  in the table above.
- **Only frozen architectures so far.** Plan items 3-6 (partial fine-tuning, hierarchical
  2.5D transformers, 3D encoders) need pixel access rather than cached features and are not
  yet run. The plan's own prior is that those win as single models.
