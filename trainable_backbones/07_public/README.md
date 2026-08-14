# Architecture 7 — the public notebooks, measured here

Six community notebooks sit in `planning/public_notebooks`. They carry two model families
and two datasets that nothing else in this sweep has tried, and they report their quality
as a Kaggle score against a hidden test set — a number that cannot be compared with
anything in `trainable_backbones`. This directory re-fits them on the local corpus and
scores them with `common/protocol.py`, so their numbers land in the same table as
architectures 1 and 2.

## The approaches

### The dataset: what a "slot" is

Architectures 1 and 2 read *everything* — every slice of every series, pooled. These
notebooks read a fixed, small, deliberately chosen subset instead, and the unit of that
choice is a **slot**: one named combination of imaging plane and acquisition type, filled
by at most one series per study.

The six slots are the three planes crossed with the sequence types a knee protocol
actually contains:

| slot | plane | fluid-sensitive | fat-suppressed | filled |
|---|---|---|---|---:|
| `AX_FLUID_FS` | axial | yes | yes | 98.5% |
| `COR_FLUID_FS` | coronal | yes | yes | 95.5% |
| `SAG_FLUID_FS` | sagittal | yes | yes | 93.5% |
| `COR_T1` | coronal | no | no | 64.1% |
| `SAG_FLUID_NOFS` | sagittal | yes | no | 62.6% |
| `SAG_T1` | sagittal | no | no | 42.5% |

The three fat-suppressed fluid-sensitive slots are nearly always present; the structural
sequences are the scarce ones, and a sagittal T1 is missing from more than half the
corpus.

A study contributes at most one series to each slot — where several match, the one with
the most slices wins — and a slot no series matches stays **empty and is masked out**,
never imputed. On this corpus 4.57 of the six are filled on average, which is why the
presence mask matters: "not acquired" and "acquired and unremarkable" must not look
alike to the model.

From each filled slot, nine slices are taken, spread evenly across the central 20–80% of
the stack in *geometric* order (position projected on the slice normal — file order here
is a SOP UID and uncorrelated with anatomy). They are cropped to a constant **130 mm of
patient anatomy** rather than a constant pixel box, intensity-normalised per series to
its 1st–99th percentile, and right knees are mirrored onto a left-knee convention —
horizontally for coronal and axial, by reversing slice order for sagittal.

So one study becomes **a bag of at most six images plus a six-element presence mask**.
The nine slices per slot are read as three groups of three consecutive slices; each
training step picks one group at random and stacks it as the three colour channels of a
single image (the usual "2.5D" trick), and evaluation averages logits over all three
groups. A study is therefore ~18 images where architecture 1 sees ~300.

### Family A: fine-tuned ViT + slot attention

```
6 slot images                 (3 channels = 3 adjacent slices, 224 or 336 px)
  -> DINOv2 ViT-S/14          last 6 of 12 blocks trainable, rest frozen
  -> [CLS ; mean of patches]  one 768-d vector per slot
  -> LayerNorm/Linear/GELU    -> 256-d, plus a learned per-slot embedding
  -> 12 learned queries       dot-product attention over the <=6 slots, absent ones masked
  -> per-finding read-out     -> 12 logits
```

The head — `SlotHead` in `model.py` — is one layer of single-head cross-attention with
twelve learned query vectors, one per finding. Each query scores every present slot,
softmaxes over them, takes the weighted sum of slot embeddings, and is read out against
its own output row. The argument for it is anatomical: cruciates are read sagittally,
collateral ligaments and the meniscal body coronally, patellar cartilage axially, so
pooling the six slots identically would dilute whichever one carries the evidence. Three
variants of this head were tested — the plain version, one with a **fixed anatomy prior**
adding a per-(finding, slot) constant to the attention logits before the softmax, and one
where the slot vector gains a third part (`cls_mean_focal`, the mean of the top eighth of
each channel across the patch grid, to stop a focal finding being averaged away).

The control is `MeanSlotHead`: identical projection, identical slot embedding, identical
dropout, with the attention replaced by a masked mean. Any difference between the two is
the attention and nothing else.

Only the last `k` transformer blocks train, plus the final LayerNorm and the head; `k` is
a swept axis (0, 2, 6, 12). The encoder is DINOv2 ViT-S/14 or DINOv3 ViT-S/16, both
self-supervised on natural images, at 224 or 336 px.

### Family B: frozen RadImageNet + query head

```
3 fat-suppressed slots x 8 slices = 24 images  (full frame, 224 px)
  -> RadImageNet ResNet-50, frozen             one 2048-d vector per SLICE
  -> Linear -> 512, + plane embedding + within-stack position embedding
  -> 12 learned queries, 8-head MultiheadAttention over the 24 tokens
  -> fuse [attended ; mean ; |attended - mean| ; attended * mean]
  -> per-finding read-out -> 12 logits
```

The notebooks run this as an independent arm whose diversity comes from the *pretraining
corpus* — a ResNet-50 trained on radiology images rather than on ImageNet or on natural
photographs. Three differences from family A matter: the encoder never trains, so its
features are computed once and cached; a token is a **slice** rather than a slot, so the
twelve queries can select a depth as well as a sequence (24 tokens instead of 6); and the
notebooks read it at *full frame*, deliberately skipping the physical crop to match how
RadImageNet was pretrained. That last decision turned out to be the arm's largest single
handicap.

### What was varied

| axis | values |
|---|---|
| encoder | DINOv2 ViT-S/14, DINOv3 ViT-S/16, RadImageNet ResNet-50 (frozen) |
| trainable blocks | 0, 2, 6, 12 |
| resolution | 224 px, 336 px |
| head | slot attention, slot attention + anatomy prior, masked mean, query head |
| slot vector | `cls_mean`, `cls_mean_focal` |
| slot scheme | six "recovered" slots, six "public" slots, three fat-suppressed |
| crop | 130 mm physical, full frame |
| slice band | central 20–80%, central 12–88% |
| augmentation | rigid jitter + intensity, or none |
| schedule | 10, 24, 30, 60 epochs |
| **teacher** | this repo's blend; the notebooks' three public report-label sets |

The last axis is a dataset rather than a model. The notebooks train on labels read out of
the radiology reports by three independent public efforts (`report_labels_v2`,
`llm_labels_v2`, `labels_llm_gpt56sol`), averaged, with a per-cell weight that falls when
the three disagree or when the average sits near a half. Swapping that for
`llm_classifiers/blend` while holding everything else fixed is the only way to tell
whether these image models are better than this repository's or merely better taught.

Evaluation never varies: soft BCE against this repository's blend on the held-out fifth,
and ROC AUC over the 58 gold studies, both from `common/protocol.py`.

## Results

Selection is val soft BCE, as everywhere else; gold ROC AUC over the 58 annotated studies
is reported and never optimised against. `folds = 5` rows are out-of-fold over all 4,349
pseudo-labelled studies, with gold AUC as the **mean of the five per-fold AUCs** — not the
AUC of their averaged prediction, which is an ensemble and flatters by about 0.005.

### Cross-validated (5 folds)

| | val soft BCE | gold AUC | GPU s |
|---|---:|---:|---:|
| *probability mean of the two families below* | *0.3828* | *0.8960* | — |
| **DINOv2-224, unfreeze 6, 30 epochs** | **0.3892** | 0.8723 ± 0.0065 | 10,493 |
| RadImageNet, 130 mm crop, band 0.20-0.80 | 0.3932 | 0.8794 ± 0.0047 | 518 |
| DINOv2-336, unfreeze 6, 10 epochs | 0.3942 | 0.8529 ± 0.0036 | 8,419 |
| RadImageNet, 130 mm crop, lr 1e-4 | 0.3950 | **0.8832 ± 0.0065** | 82 |
| DINOv2-224, unfreeze 6, 10 epochs | 0.3970 | 0.8477 ± 0.0036 | 3,426 |
| RadImageNet, full frame — *the notebooks' own configuration* | 0.3989 | 0.8574 ± 0.0090 | 90 |
| DINOv2-224 frozen + slot head | 0.4332 | 0.7536 ± 0.0039 | 2,303 |

### Single split, the protocol's own 80/20 — comparable to the other architectures

| | val soft BCE | gold AUC |
|---|---:|---:|
| baseline: predict train mean | 0.4969 | — |
| architecture 2 — learned series attention, best | 0.4357 | 0.7956 |
| architecture 1 — frozen pooling, best | 0.4146 | 0.8260 |
| slot model, DINOv2-224, 10 epochs | 0.4051 | 0.8520 |
| slot model, DINOv2-336, 10 epochs | 0.4024 | 0.8609 |
| slot model, DINOv2-224, unfreeze 12, 10 epochs | 0.4024 | 0.8622 |
| **slot model, DINOv2-224, 30 epochs** | 0.3972 | **0.8856** |
| **slot model, DINOv2-336, 30 epochs** | **0.3958** | 0.8797 |
| **slot model, DINOv2-224, unfreeze 12, 30 epochs** | **0.3958** | 0.8828 |

The three thirty-epoch rows are a tie, not a ranking: 0.0014 BCE and 0.006 AUC separate
them, against a fold-to-fold standard deviation of 0.005 and 0.007 for this model. Which
of the three prints the lowest number on one split is not information.

The five-fold split is the same shuffle the protocol uses, cut into fifths, so its last
fold *is* the protocol's validation set. Fold 4 is a slightly harder fifth than average —
0.4051 against an out-of-fold 0.3970 for the same configuration — which is worth knowing
before reading any single-split difference of that size as a result.

## What actually pays

**Fine-tuning the encoder at all.** This is the one large effect. A frozen encoder scores
0.4425; opening two blocks gives 0.4187, six gives 0.4051, all twelve 0.4024. Everything
else in this section is small next to the first step away from frozen.

**Training to convergence — and it subsumes the other two.** Ten epochs had not converged.
Thirty takes the single split from 0.4051 to 0.3972 and gold AUC from 0.852 to 0.886,
with the trace flattening around epoch 22. The notebooks train for ten because they share
a nine-hour budget with a full decode of the hidden test set: a platform constraint, not
a modelling finding, and the single largest thing they leave on the table. Cross-validated
it holds — 0.3892 out-of-fold against 0.3970 at ten epochs, on every fold.

**Resolution and unfreezing depth turn out to be the same finding wearing a hat.** At ten
epochs both look real: 336 px is worth 0.0028 BCE and 0.009 AUC over 224 px, and twelve
blocks 0.0027 BCE over six. At thirty epochs all three configurations land on top of each
other:

| at 30 epochs | val soft BCE | gold AUC |
|---|---:|---:|
| DINOv2-224, unfreeze 6 | 0.3972 | 0.8856 |
| DINOv2-336, unfreeze 6 | 0.3958 | 0.8797 |
| DINOv2-224, unfreeze 12 | 0.3958 | 0.8828 |

A spread of 0.0014 BCE and 0.006 AUC, against a fold-to-fold standard deviation of 0.005
and 0.007. What more resolution and more trainable blocks were buying at ten epochs was
mostly *faster convergence*, not a better model — which is worth knowing before paying
three times the compute for 336 px. The notebooks' own resolution argument (a 130 mm crop
at 224 px samples at 0.58 mm, above the 0.5 mm a 1 mm tear needs, while 336 px clears it)
is sound sampling theory that this corpus does not reward once the schedule is long
enough.

**Physical normalisation, for the RadImageNet arm.** Those notebooks deliberately disable
the 130 mm crop there, to match the full-frame images RadImageNet was pretrained on.
Turning it back on is worth 0.0036 BCE and **0.026 gold AUC** — second only to the epoch
budget among everything tried here, and it comes from deleting a line. Matching the
pretraining field of view costs that arm more than the domain mismatch it avoids.

## What does not pay

Every refinement the notebooks make *above* the encoder is worth nothing here.

| | val soft BCE | gold AUC |
|---|---:|---:|
| 12 learned queries attending over slots (as published) | 0.4051 | 0.8520 |
| a masked mean over slots, same capacity otherwise | 0.4049 | 0.8496 |
| queries + the fixed per-finding anatomy prior | 0.4041 | 0.8489 |
| focal patch pooling (top-eighth of each channel) | 0.4106 | 0.8489 |
| no augmentation at all | 0.4060 | 0.8501 |
| the simpler plane x fat-suppression slot scheme | 0.4048 | 0.8569 |

**Per-diagnosis attention is the architectural claim of these notebooks, and it is a
null.** Replacing twelve learned queries with a masked mean changes val BCE by 0.0002.
This is the third time the same result has appeared here: attention-based MIL lost to
fixed pooling in `frozen_backbones`, architecture 2's learned series attention lost to
architecture 1's fixed pooling by a margin wider than either sweep's spread, and now
learned slot attention ties a mean. Sixty times more training signal has not reversed it.
The reading that survives all three is that the useful variation is *between* studies —
which sequences were acquired at all — rather than in how a study's sequences should be
weighted, and a presence mask already carries that.

**The anatomy prior, in particular, is a null.** It encodes real anatomy — cruciates
sagittally, collaterals coronally — as a fixed tilt on the attention logits, and it moves
nothing (0.4041 against 0.4051, inside the fold-to-fold spread of 0.005). Correct domain
knowledge, expressed where the model had already found it or did not need it.

**Conditioning on both sequence axes changes nothing, even though the delivered flags are
broken.** `Fluid_Sensitive` and `Fat_Suppression` in `train_series.csv` are byte-identical
on all 24,371 series — one flag published twice — so both axes have to be recovered from
the DICOM headers, which the notebooks do and this port reproduces. The question that
leaves open is whether the second axis is worth conditioning on at all. The "recovered"
scheme uses both (plane x fluid-sensitivity x fat-suppression) and scores 0.4051/0.8520;
the "public" scheme keeps only plane x fat-suppression, mimicking what a single honest
flag would have supported, and scores 0.4048/0.8569. No difference, and the simpler
scheme fills more slots (4.82 of six against 4.57) because its predicates are looser.
Recovering the flags is necessary; splitting on the recovered second axis is not.

## Encoders

| | val soft BCE | gold AUC |
|---|---:|---:|
| DINOv2 ViT-S/14, 336 px | 0.4024 | 0.8609 |
| DINOv2 ViT-S/14, 224 px | 0.4051 | 0.8520 |
| DINOv3 ViT-S/16, 336 px | 0.4169 | 0.8301 |
| DINOv3 ViT-S/16, 224 px | 0.4196 | 0.8236 |

The `bend-the-knee` notebook's contribution to the community pipeline is precisely one
DINOv3 ViT-S/16 rank-blended into twenty DINOv2 models. As a member measured on its own,
DINOv3 is behind DINOv2 at both resolutions and by a margin (0.014 BCE, 0.03 AUC) larger
than most of this sweep's effects. That does not make the blend wrong — a weaker member
with decorrelated errors can still improve a rank mean, which is the case that notebook
argues — but the member is not an upgrade on its own terms, and this sweep cannot confirm
the ensemble claim from one member.

## The frozen RadImageNet arm is the efficient answer

It is the notebooks' side experiment: a ResNet-50 pretrained on radiology images, frozen,
one embedding per slice, and a small query head over the 24 resulting tokens. With the
crop restored it reaches **0.3932 BCE / 0.879 gold AUC**, against 0.3942 / 0.853 for a
fine-tuned DINOv2-336. Five folds of it take about 90 seconds of head training against
that model's 8,419 — the 518 seconds in the table above is a cold run that also encoded
the frozen features for a pixel configuration nobody had built yet, which happens once.

That is roughly 90x less compute for the same BCE and better AUC, and it makes the
comparison the one against architecture 1's frozen features rather than against the
fine-tuned models. Both are frozen encoders with a trained head; the differences are the
pretraining corpus (radiology against MRI-CORE's MRI), the read-out (one vector per slice
against pooled statistics per series) and the dataset (24 chosen slices against every
slice). It wins by 0.021 BCE and 0.053 AUC.

The arm is also almost unresponsive to its own hyperparameters — learning rate from 5e-5
to 2e-4 spans 0.3950 to 0.3953, and 60 epochs is identical to 24 — which is what a head
over frozen features being the easy part looks like.

## The slot dataset is not what wins

The single most useful control here is the frozen one. If the six-slot dataset were a
better *representation* of a study, a frozen encoder over it should beat a frozen encoder
over whole stacks. It does not, and not narrowly:

| | val soft BCE | gold AUC |
|---|---:|---:|
| architecture 1 — frozen MRI-CORE, every slice, fixed pooling | 0.4146 | 0.8260 |
| this — frozen DINOv2, six slots x three slices, slot head | 0.4425 | 0.7530 |

Reading eighteen images instead of three hundred throws away evidence, and the throwing
away is visible whenever the encoder cannot adapt to compensate. What the slot dataset
buys is that eighteen images per study *can* be fine-tuned on a single GPU in twenty
minutes, where three hundred cannot. It is a compute decision that pays for itself
through the fine-tuning it enables, not a better view of a knee — and the frozen
RadImageNet result says even that framing is generous, since the same 24 slices support a
better model with no fine-tuning at all when the encoder is pretrained on the right
domain.

## The labels are yours, and it matters

| training teacher | val soft BCE | calibrated | gold AUC |
|---|---:|---:|---:|
| `llm_classifiers/blend` | 0.4051 | 0.4048 | 0.8520 |
| public 3-teacher mean, agreement-weighted | 0.4439 | 0.4216 | 0.8137 |
| public 3-teacher mean, unweighted | 0.4465 | 0.4220 | 0.8102 |
| `report_labels_v2` alone | 0.4583 | 0.4251 | 0.8026 |

The evaluation always reads this repository's blend, so a model taught by a different
teacher is scored partly on a marginal-rate offset — the public labels call a finding
present about half again as often. The `calibrated` column removes that: two parameters
per finding, fitted on 512 *training* studies, never on the validation fifth, and
monotone so gold AUC is untouched. About half the raw gap is calibration and half is
real. The ordering is the same in all three columns.

Same result on the frozen arm, where it is cheap enough to be sure: 0.3989 → 0.4396 raw,
0.3994 → 0.4068 calibrated, 0.857 → 0.836 AUC. The agreement/certainty weighting the
notebooks put on the public labels is worth a little (0.003 BCE, 0.004 AUC) and does not
close the gap.

## Ensembling

The notebooks' submission is not a model, it is a rank mean of twenty-odd members, and
that operation does work here too.

| | val soft BCE | gold AUC |
|---|---:|---:|
| DINOv2-224, 30 epochs | 0.3892 | 0.879 |
| RadImageNet, 130 mm crop | 0.3950 | 0.887 |
| **probability mean of the two** | **0.3828** | **0.896** |
| rank mean of the two — *their operation* | — | 0.894 |
| probability mean of three | 0.3819 | 0.895 |

Gold AUC in this table is over each family's five fold models averaged together, so these
are ensemble numbers on both axes and are not comparable to the per-fold numbers above.
A rank is not a probability, so a rank mean has no meaningful BCE and none is quoted.

The two families are worth combining and a third member is not: one fine-tuned ViT over
six slot summaries and one frozen ResNet-50 over twenty-four individual slices disagree
in useful ways, while two RadImageNet variants that differ only in slice band do not.
That is the structural argument for the notebooks' twenty-member rank mean, and it is
the part of their approach this sweep can least well evaluate — it measures members, and
members are what transfer to a different evaluation.

## Files

| | |
|---|---|
| `pixels.py` | the notebooks' slot dataset over the local DICOMs; builds a memmapped cache |
| `model.py` | `SlotHead` / `MeanSlotHead` + fine-tuned ViT, and `FoundationQueryHead` |
| `targets.py` | the teacher choice: this repo's blend or the public report labels |
| `train.py` | training and evaluation under `common/protocol.py`; `--folds 5` for CV |
| `blend.py` | rank / probability combination of saved predictions |
| `results/` | one CSV per sweep, plus per-configuration predictions for blending |

Reproducing from scratch is a 19-minute header-and-decode pass over the 4,407 studies
(`python pixels.py`), then `python train.py --sweep main`. The slice order is remembered
in `cache/slice_order.json`, so later configurations skip the expensive half.
