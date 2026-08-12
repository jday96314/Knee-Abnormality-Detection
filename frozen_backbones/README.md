# Frozen-backbone probes for knee MRI findings

Feeds features from two frozen MRI foundation models into logistic regression and
asks which pooling strategy turns a bag of slices into the best study-level
descriptor. Scored by macro-averaged ROC AUC over pooled out-of-fold predictions
from 5-fold stratified CV on the 58 fully-labeled studies.

## Headline

| | macro AUC |
|---|---|
| Best configuration, 20 held-out CV seeds | **0.661 ± 0.013** |
| Same, seed-averaged OOF predictions | **0.674**, 95% CI **[0.619, 0.727]** (bootstrap over studies) |
| Simplest thing that works (`mri_core/cls`, plain mean) | 0.652 ± 0.014 |
| Labels shuffled (permutation baseline) | 0.496 ± 0.018 |

Best configuration: **OrthoFoundation @ 224px, `patch_std` read-out,
`mean-plane-ctr-l2` pooling** — per-plane mean over the central half of each
stack, with each slice L2-normalized first.

**The honest reading.** The bootstrap CI over studies spans 0.11 of AUC. Every
pooling strategy tested falls inside that interval. Pooling choice is worth about
0.01 macro AUC; the 58-study cohort is worth ten times that in uncertainty. The
ranking below is real but small, and it is a ranking over partitions of one small
cohort, not a claim about new data.

## What actually moved the metric

Marginal medians over all 480 configurations (`python analyze.py`):

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
   correlated in this cohort, so they are one binary variable, not two.
5. **Trimming to the central half of each stack helps**, consistent with the ends
   of a knee series being off-joint.
6. **Fusion hurts.** On the same five seeds, concatenating read-out heads (0.635)
   or both backbones (0.633) scores *below* the best single head (0.669). At n=58
   the extra columns cost more than the extra signal pays.

## Per-label

Leading configuration, 20 held-out seeds:

| Finding | Positives | AUC | | Finding | Positives | AUC |
|---|---|---|---|---|---|---|
| Medial OA | 15 | 0.789 | | Lateral OA | 11 | 0.678 |
| Effusion | 35 | 0.719 | | MCL | 9 | 0.651 |
| Lateral Meniscus | 23 | 0.717 | | Contusion | 19 | 0.630 |
| ACL | 24 | 0.697 | | Baker's | 12 | 0.572 |
| Medial Meniscus | 26 | 0.687 | | PF OA | 21 | 0.563 |
| Synovitis | 27 | 0.680 | | Fracture | 18 | 0.552 |

Across all 480 configurations, Medial OA and Effusion beat chance in essentially
every run; Baker's cyst is above 0.5 in under half of them and Contusion in 59%.
Those two are not being detected — whichever configuration tops a ranked list has
picked them up by chance.

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
python sweep.py --out sweep.csv                      # 480 configs (~2 h)
python analyze.py                                    # marginal effect of each axis
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
