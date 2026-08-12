# Frozen-backbone probes for knee MRI findings

Feeds features from two frozen MRI foundation models into logistic regression and
asks which pooling strategy turns a bag of slices into the best study-level
descriptor. Scored by macro-averaged ROC AUC over pooled out-of-fold predictions
from 5-fold stratified CV on the 58 fully-labeled studies.

## Headline

| | macro AUC |
|---|---|
| Best configuration, 20 held-out CV seeds | **0.665 ± 0.015** |
| Same, seed-averaged OOF predictions | **0.685**, 95% CI **[0.630, 0.740]** (bootstrap over studies) |
| Best flat-bag (non-hierarchical) configuration | 0.661 ± 0.013 |
| Simplest thing that works (`mri_core/cls`, plain mean) | 0.652 ± 0.014 |
| Labels shuffled (permutation baseline) | 0.493 ± 0.026 |

Best configuration: **OrthoFoundation @ 224px, `patch_std` read-out,
`meanmax-plane-axmean-ctr`** — over the central half of each stack, mean⊕max
within each plane, then averaged across planes.

**The honest reading.** The bootstrap CI over studies spans 0.11 of AUC. Every
pooling strategy tested falls inside that interval. Pooling choice is worth about
0.02 macro AUC end to end; the 58-study cohort is worth five times that in
uncertainty. The rankings below are real -- the matched hierarchical comparisons
hold at p<0.005 across 20 CV partitions -- but they are rankings over partitions
of one small cohort, not claims about new data.

## What actually moved the metric

Marginal medians over the 480 flat-bag configurations (`python analyze.py sweep.csv`):

| Axis | Winner | Median | Runner-up | Median |
|---|---|---|---|---|
| Read-out head | `cls` | 0.627 | `patch_max` | 0.608 |
| Resolution | 224px | 0.607 | 448px | 0.597 |
| Reducer | `meanmax` | 0.610 | `mean` | 0.596 |
| Grouping | per-plane | 0.610 | whole study | 0.598 |
| Central half only | yes | 0.613 | no | 0.599 |
| Backbone | OrthoFoundation | 0.606 | MRI-CORE | 0.595 |
| Slice L2 | no | 0.602 | yes | 0.597 |

Read in order of effect size:

1. **The read-out head matters more than the pooling.** `cls` beats `patch_mean`
   by 0.04 in median — four times the spread of the entire pooling axis.
2. **Higher resolution does not help.** 448px is consistently worse than 224px for
   both backbones, including for OrthoFoundation, whose RoPE handles it natively.
   Knee findings at these scales appear to be resolved at 14x14 patches already.
3. **Concatenating mean with max beats either alone**, which is the one pooling
   result that holds up across backbones and heads: a study descriptor wants both
   "what is generally here" and "what is the most extreme thing here".
4. **Conditioning on plane helps a little**; conditioning on sequence type helps
   less. Note that `Fluid_Sensitive` and `Fat_Suppression` are perfectly
   correlated in this cohort, so they are one binary variable, not two. Whether
   the per-plane descriptors are concatenated or reduced across makes no
   difference on average (see the hierarchical section), so the cheaper
   dimension-preserving `across` form is preferable.
5. **Trimming to the central half of each stack helps**, consistent with the ends
   of a knee series being off-joint.
6. **Fusion hurts.** On the same five seeds, concatenating read-out heads (0.635)
   or both backbones (0.633) scores *below* the best single head (0.669). At n=58
   the extra columns cost more than the extra signal pays.

## Hierarchical pooling: which level wants which reduction

A study is a three-level object -- slices in a series, series in a plane, planes
in a study -- and `hier.py` sweeps the reduction used at each level separately
(288 configurations over the three leading backbone/head combinations):

| Level | Options | Winner (median) |
|---|---|---|
| `inner` — collapse each series | none, mean, max, `p90` | **p90** 0.642, vs none 0.636, max 0.634 |
| `reduce` — combine series within a plane | mean, max, `mean⊕max` | **mean⊕max** 0.640 |
| `across` — combine planes | concat, mean, max | wash (0.636 / 0.638 / 0.634) |
| `central` — trim stack ends | on/off | **on** 0.643 vs 0.631 |

**Adding a series level helps, but only if it preserves within-series extremes.**
Matched pairs, identical except for the inner reduction, 20 held-out seeds,
`orthofoundation/224/cls`:

| Hierarchical | Flat control | Δ | p |
|---|---|---|---|
| `max-all-inp90-ctr` | `max-all-ctr` | **+0.022** | <0.001 |
| `meanmax-all-inp90-ctr` | `meanmax-all-ctr` | **+0.015** | <0.001 |
| `meanmax-plane-inp90-axmax-ctr` | `meanmax-plane-axmax-ctr` | **+0.009** | 0.003 |
| `meanmax-all-bal` (inner=**mean**) | `meanmax-all` | **−0.009** | 0.012 |

A soft-max (p90) within series helps; a *mean* within series actively hurts,
which is why the first sweep -- whose only two-level option was `bal` -- found
hierarchy useless. Averaging within a series and then again across them discards
slice-level variation twice.

**The gain is exactly where the mechanism predicts.** Per-label effect of adding
p90-within-series (`max-all-inp90-ctr` vs `max-all-ctr`):

| Focal findings | Δ AUC | | Diffuse findings | Δ AUC |
|---|---|---|---|---|
| MCL | +0.084 | | Medial OA | +0.005 |
| Contusion | +0.051 | | Lateral Meniscus | +0.005 |
| PF OA | +0.049 | | Effusion | +0.001 |
| Lateral OA | +0.042 | | Synovitis | −0.003 |
| Baker's | +0.029 | | ACL | −0.007 |
| Fracture | +0.022 | | Medial Meniscus | −0.011 |

Findings visible on a handful of slices in one acquisition gain; findings visible
on most slices of most series do not. Collapsing a series by a soft-max before
pooling preserves focal evidence that a flat reduction dilutes into 1/180th of a
study average.

**A cautionary note on the sweep's top line.** The best single-seed configuration
out of the 288 scored 0.692; on 20 held-out seeds the same configuration scores
**0.648** -- below six others in the shortlist. Picking the maximum of a few
hundred noisy estimates buys about 0.03 of pure selection bias. Only the matched
comparisons above, and the held-out re-scoring in `confirm.py`, are load-bearing.

## Per-label

Leading configuration, 20 held-out seeds:

| Finding | Positives | AUC | Finding | Positives | AUC |
|---|---|---|---|---|---|
| Medial OA | 15 | 0.786 | Lateral Meniscus | 23 | 0.665 |
| Lateral OA | 11 | 0.750 | Contusion | 19 | 0.635 |
| ACL | 24 | 0.736 | MCL | 9 | 0.627 |
| Effusion | 35 | 0.704 | PF OA | 21 | 0.622 |
| Synovitis | 27 | 0.693 | Baker's | 12 | 0.543 |
| Medial Meniscus | 26 | 0.681 | Fracture | 18 | 0.543 |

Across the 480 flat-bag configurations, Medial OA and Effusion beat chance in
essentially every run; Baker's cyst was above 0.5 in under half of them and
Contusion in 59%. Hierarchical pooling rescued Contusion (0.635) but not Baker's
or Fracture, both of which remain at chance and should be read as undetected.

## Backbone notes

Both checkpoints needed handling their own repos get wrong or leave undocumented.

**MRI-CORE** is a DINO-pretrained plain **ViT-B/16 at 224px**, not a SAM model.
The checkpoint is the DINO teacher: `pos_embed` is `[1, 197, 768]`, there is no
layerscale, and the blocks sit in DINOv2 `BlockChunk`s. `backbones.py` loads it
into a plain timm ViT (0 missing / 0 unexpected keys) rather than through the
README's `sam_model_registry` path, because that path:

- never loads `pos_embed`. `_build_sam` interpolates it and then writes it back
  under the *source* key `backbone.pos_embed` instead of the remapped
  `image_encoder.pos_embed`, so the position embedding silently stays at random
  init. Loading that way reports `image_encoder.pos_embed` missing and
  `backbone.pos_embed` unexpected.
- wraps the weights in windowed attention (`window_size=14`, all but four blocks)
  and a SAM neck for which the checkpoint contains no weights at all.

**OrthoFoundation** is DINOv3 ViT-L/16 and loads strict into the reference
architecture. It must run in **bfloat16** — in fp16 every feature comes back NaN.
`extract.py` asserts finiteness after extraction, because that failure is
otherwise silent until a scaler raises much later.

## Method

- **Cohort**: the 58 studies in `train.csv` with all twelve labels present; all are
  on disk. 336 series, 10,528 slices.
- **Decoding**: every slice of every series, once, to a 448px uint8 cache
  (`data.py`). Slices are ordered by `InstanceNumber` when unique, else by position
  projected onto the slice normal. Intensity is windowed to the 0.5-99.5 percentile
  rather than min-max, which MR magnitude images need; images are letterboxed, not
  stretched. Nothing is subsampled, so slice selection stays a pooling decision.
- **Features**: four read-out heads per slice (`cls`, and patch-grid mean / max /
  std), at 224px and 448px, per backbone. Stored per slice with study, series,
  plane, sequence type and normalized through-plane position, so pooling can be
  re-run without a GPU.
- **Probe**: `StandardScaler` -> row-space rotation -> L2 logistic regression with
  `class_weight="balanced"`. C is selected by *repeated* stratified CV inside each
  training fold, never against the out-of-fold predictions being scored.
- **Splits**: per-label `StratifiedKFold(5)`. A split shared across all twelve
  labels cannot keep MCL (9 positives) balanced.

Two details that changed the numbers materially and are worth knowing if you
extend this:

- **C selection is not a formality.** On one split, C moved a label's AUC from
  0.31 to 0.88. A grid topping out at C=1 leaves real AUC unclaimed.
- **Single inner-CV C-selection is noisy enough to invert results.** With one
  4-fold inner estimate, a *wider* C grid scored *worse* — the extra candidates
  just gave noise more chances to win. `RepeatedStratifiedKFold(n_repeats=3)`
  fixed it, after which the wider grid wins and is stable.

### The row-space rotation

`probe.RowSpaceSVD` projects each fold's design onto the training right-singular
vectors. This is exact, not an approximation: an L2-penalized linear model's
solution lies in the span of the training rows, so any component of a test point
orthogonal to that span is multiplied by zero, and the rotation is orthonormal so
the penalty is unchanged. It shrinks designs of up to 21,504 columns to at most 46
and makes the sweep tractable. `tests/test_probe.py` asserts prediction-level
equivalence against an unreduced fit.

## Reproducing

```bash
python data.py 448                                   # decode cache (~2 min)
for b in mri_core orthofoundation; do
  for s in 224 448; do python extract.py --backbone $b --size $s; done
done                                                 # features (~5 min, GPU)
python sweep.py --out sweep.csv                      # 480 flat-bag configs (~2 h)
python hier.py --out hier.csv                        # 288 hierarchical configs (~1 h)
python analyze.py sweep.csv                          # marginal effect of each axis
python analyze.py hier.csv
python stage2.py --seeds 5 --top-k 6                 # L2, head/backbone fusion
python confirm.py --first-seed 100 --n-seeds 20      # held-out seeds + permutation
python -m pytest tests/                              # pooling and speedup correctness
```

Set `OMP_NUM_THREADS=2` for the sweep; the probe parallelizes over labels.

## Where the ceiling probably is

Nothing here trains the backbone or lets the classifier choose *which* slices
matter — every strategy is a fixed reduction over the whole bag. The obvious next
step is attention-based MIL over the per-slice features, which are already cached
and would cost no new GPU time. Beyond that, the binding constraint is 58 labeled
studies; `train.csv` ships ~4,350 unlabeled studies with reports, and deriving
weak labels from those is likely worth more than any further pooling work.
