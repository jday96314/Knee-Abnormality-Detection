# Vision models distilled from the pseudo-labelled studies

Trains image-only models to imitate the blended text+image soft targets from
`../llm_classifiers/blend`, following the architecture ladder in `planning/llm_plans/plan.md`.

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

**A single tuned image-only model reaches 0.884 gold ROC AUC from pixels alone.**

| | gold AUC | what it is |
|---|---:|---|
| **architecture 3, tuned (OrthoFoundation + slot head)** | **0.884** | this work — pixels only |
| architectures 3 + 4 ensembled (pre-tuning) | 0.878 | this work — pixels only |
| architecture 3, hand-picked configs | 0.863 | this work — pixels only |
| architecture 1 (best frozen config) | 0.836 | this work — frozen features |
| blended teacher it learned from | ~0.92 | text + image, needs reports |
| image-only VLM teacher | 0.723 | 2 models, TTA, ensembled |
| frozen probe trained on gold labels | 0.706 | 58 hard labels, no distillation |

Two things follow. First, the distillation premise holds: report-derived knowledge, which
does not exist at test time, has been transferred into a model that sees only images —
**+0.155 AUC over the image-only VLM ensemble that helped produce the labels**, closing
roughly three quarters of the gap between that ensemble and the text-informed teacher.
Second, **3,479 noisy soft targets beat 58 clean hard ones by a wide margin** (0.884 vs
0.706).

*Caveat.* The blender's parameters were fitted on the gold labels, so gold information
indirectly shaped the pseudo-targets. For AUC the effect should be nil — those parameters
are monotone within each finding and AUC is monotone-invariant — but it is not literally
zero, and a clean estimate would need studies held out from the blender too.

## The ladder

296 configurations across all six architectures the plan proposes: 32 + 32 + 16 + 16 for
architectures 1-2 on two backbones, 9 for architecture 4, 4 for the volumetric family
(#5 and #6), and for architecture 3 both the original 4 hand-picked configurations and a
later round of **15 Optuna studies / 183 trials** over 5 backbones x 3 heads
(see `03_finetuned_2p5d/TUNING.md`).

| # | Architecture | encoder | configs | best val BCE | gold AUC |
|---:|---|---|---:|---:|---:|
| | baseline (train mean) | — | — | 0.4969 | — |
| **3** | **partially fine-tuned 2.5D, tuned** | **OrthoFoundation ViT-L, 19/24 blocks** | 183 | **0.4000** | **0.8836** |
| 3 | partially fine-tuned 2.5D, hand-picked | MRI-CORE, 8/12 blocks | 4 | 0.4064 | 0.8630 |
| 1 | fixed hierarchical pooling | MRI-CORE, frozen | 32 | 0.4146 | 0.8260 |
| 1 | fixed hierarchical pooling | OrthoFoundation, frozen | 32 | 0.4146 | 0.8177 |
| 4 | hierarchical slice+series Transformer | MRI-CORE, frozen | 9 | 0.4178 | 0.8432 |
| 2 | learned pathology attention over series | MRI-CORE, frozen | 16 | 0.4357 | 0.7956 |
| 6 | 3D video encoder per series | R3D-18 Kinetics, 1 stage | 2 | 0.4427 | 0.8089 |
| 2 | learned pathology attention over series | OrthoFoundation, frozen | 16 | 0.4478 | 0.7793 |
| 5 | pretrained 3D MRI encoder per series | MedicalNet, 1 stage | 2 | 0.4555 | 0.7529 |

### What was actually searched

The ladder is not searched uniformly, and the asymmetry is a compute artefact rather than a
judgement: a frozen-feature configuration costs 3-9 minutes, a pixel-level one 7-30 minutes.
So architectures 1, 2 and 4 got full factorial grids and architectures 3, 5 and 6 got a
handful of hand-picked cells — **including the architecture that won.**

| # | Search | Backbones | Swept axes | Held fixed (not ablated) |
|---:|---|---|---|---|
| 1 | 2^5 grid x2 backbones (64 runs) | MRI-CORE, OrthoFoundation | head width, dropout, slice jitter, series dropout, MixUp | pooling statistics, LR, batch, resolution, epochs |
| 2 | 2^4 grid x2 backbones (32 runs) | MRI-CORE, OrthoFoundation | `d_model`, attention depth, dropout, series dropout | n_heads, 12 queries, slice pooling, LR |
| 3 (first pass) | 4 hand-picked cells | **MRI-CORE only** | `unfreeze` (0/4/8), head mode | **2.5D context module, hybrid pooling**, LR, `d_model`, slice budget, augmentation |
| 3 (tuned) | 15 Optuna studies, 183 trials | MRI-CORE, OrthoFoundation, RadImageNet, DINOv2-S, DINOv3-S | `unfreeze`, LR, encoder-LR ratio, wd, batch, epochs, schedule, dropout, `kernel`, 8 augmentation dials | hybrid pooling, `d_model`, slice budget, resolution |
| 4 | 9 of 16 grid cells | **MRI-CORE only** | slice depth, series depth, head mode, consistency | `d_model`, n_heads, slice cap, position encoding |
| 5,6 | 4 hand-picked cells | n/a (3D encoders) | encoder, `unfreeze` (0/1) | **resolution, depth**, study head, LR, augmentation |

**The three gaps most likely to change conclusions — all since closed for architecture 3**
(see `03_finetuned_2p5d/TUNING.md`), and each closed against the original expectation:

1. **`unfreeze` was never pushed past 8 of 12 blocks.** *Closed:* tuned studies pick 6-10 of
   12 for the ViT-B/S encoders and 19 of 24 for OrthoFoundation. 8/12 was indeed below the
   optimum, and the prior that fitting 86M parameters to 3,479 labels would overfit is
   unsupported — heavy augmentation is what makes heavy unfreezing safe.
2. **Architecture 3 ran on one backbone.** *Closed:* five backbones. The four smaller ones
   land within 0.003 of each other, so MRI-CORE is **not** special; only OrthoFoundation's
   ViT-L separates, and on 3 trials per study.
3. **Architecture 3's defining components were never ablated.** *Partly closed:* `kernel=1`
   (which neuters the Conv1D slice-context module) wins **10 of 15** tuned studies, so the
   "2.5D" part is not contributing. Hybrid pooling remains unablated.

Still open: the volumetric family ran at 112 px against everything else's 224, confounding
resolution with architecture; and architectures 1, 2 and 4 have had no equivalent tuning
round, so their numbers are hand-set-hyperparameter numbers being compared against a tuned
one.

### Three findings that survive the noise

**1. Letting the encoder adapt is worth more than every aggregation, pooling and
augmentation choice combined.** Architecture 3's own controlled ablation — same sampling,
augmentation, context module, pooling and head, encoder gradients on vs off — is
0.4185 → 0.4064, a gain of **0.0121**. Against that, architecture 1's entire 32-config
augmentation sweep spans 0.0038, and the best-vs-worst architecture gap among frozen models
is 0.021. The same effect reappears in the volumetric family, where unfreezing one stage is
worth 0.012-0.016. This is the single most reliable lever found.

**2. Fixed pooling beats learned aggregation — but only while the encoder is frozen.**
Architecture 2's *best* configuration is worse than architecture 1's *worst* (0.4357 vs
0.4278 on MRI-CORE; 0.4478 vs 0.4429 on OrthoFoundation), and architecture 4's pathology
queries do worse still, two of four configurations landing *below the predict-the-mean
baseline*. Yet in architecture 3, with a trainable encoder, the gap vanishes: across 15
tuned studies three different heads take the top slot on five backbones, and on RadImageNet
all three land within 0.0007. The failure is not the idea; it is asking learned per-finding
routing to operate on features that cannot move. Once the encoder adapts, **the study-level
aggregator stops mattering at all** — which also means the frozen-regime head comparisons
should not be read as evidence about heads in general.

**3. Pretraining corpus size beats pretraining domain match.** In the volumetric family a
Kinetics-400 *video* network beats a MedicalNet *3D medical* network at both unfreeze levels
(0.4427 vs 0.4555; gold AUC 0.809 vs 0.753). The plan called the MRI-specific 3D encoder the
"main wildcard with a realistic chance of beating the hierarchical 2D system" and cast the
video model as a diversity branch. The opposite happened, and both lost to 2D anyway.

### Ensembling: the diversity branch does not pay

The plan asks that the 3D/video branch be "judged largely on ensemble gain, not solo AUC",
because different volumetric inductive biases might be valuable even if the model is weaker.
Each architecture's best configuration was retrained with `--save-preds` and every subset
averaged in logit space (`common/ensemble.py`):

| members | val soft BCE | gold AUC |
|---|---:|---:|
| **arch 1 + arch 3 + arch 4** | **0.3993** | 0.8711 |
| arch 3 + arch 4 | 0.3997 | **0.8775** |
| arch 1 + arch 3 | 0.4001 | 0.8648 |
| arch 1 + arch 3 + arch 4 + arch 5 | 0.4031 | 0.8697 |
| arch 3 alone (best single) | 0.4064 | 0.8630 |
| arch 5 alone | 0.4427 | 0.8089 |

**Ensembling is worth +0.0070 BCE and +0.008 gold AUC over the best single model** — a real
gain, and roughly half of what fine-tuning bought. The two-member `arch 3 + arch 4` pair
reaches the best gold AUC of the whole project, **0.8775**, which is the number to quote for
an image-only system.

**The volumetric branch fails its own test.** Adding architecture 5 makes *every* ensemble it
joins worse, without exception:

| base | + arch 5 | change |
|---|---|---:|
| arch 1 + arch 3 + arch 4 → 0.3993 | 0.4031 | +0.0038 |
| arch 3 + arch 4 → 0.3997 | 0.4048 | +0.0051 |
| arch 1 + arch 3 → 0.4001 | 0.4048 | +0.0047 |
| arch 1 + arch 4 → 0.4093 | 0.4118 | +0.0025 |

So the hypothesis that a weaker model with a different inductive bias earns its place through
diversity is **falsified here**. It is not merely that the 3D branch is weak; it is that its
errors do not complement the 2D hierarchy enough to offset being 0.036 behind. Note this is
the opposite of the earlier `frozen_backbones` finding that prediction-level ensembling was
the largest single improvement — that held among *comparable* members, and does not extend
to a member this far off the pace.

*These ensembles use the pre-tuning architecture 3 (0.4064 / 0.8630). The tuned single model
now beats the best ensemble on gold AUC (0.8836 vs 0.8775), so the ensemble has not been
rebuilt from tuned members — doing so is the obvious next step and would likely improve on
both.*

The pairing that does work is architectures 3 and 4 — the fine-tuned 2.5D model and the
frozen hierarchical Transformer. They differ in encoder adaptation, in slice coverage
(24 sampled vs up to 48 per series) and in aggregation, which is evidently the useful kind
of diversity: two strong models that reach similar accuracy by different routes.

### Backbone replication

Architectures 1 and 2 were run on two independently pretrained encoders. The conclusions
hold under both, and two details make that more than a repeat:

- The two backbones' best BCEs agree to four decimals (0.41458 vs 0.41459) — coincidence,
  confirmed at full precision, not a caching bug.
- **The 32 configurations rank almost identically across backbones (r = 0.949).** The
  hyperparameter landscape is essentially backbone-independent, further evidence that these
  sweeps measure the head and the aggregation rather than the features. The one disagreement
  is head capacity: MRI-CORE prefers `hidden=256`, OrthoFoundation a bare linear head.

This contradicts the plan's expectation that per-finding series attention would be "the
best frozen-feature model", but it reproduces what the earlier `frozen_backbones` work
found on 58 studies, where attention-based MIL scored 0.09 below fixed pooling. The
interesting part is that 60x more training signal did not reverse it. Twelve query tokens
learning to attend over 3-15 series is evidently still the harder estimation problem, even
with 3,479 studies, and the fixed `[mean, p90, max]` prior is strong enough that learned
aggregation has little left to add.

The backbone replication is what makes this safe to build on. A result that held for one
feature set could be a quirk of those features; holding for two independently pretrained
encoders makes it a property of the *aggregation problem*. Two supporting details:

- The two backbones' best BCEs agree to four decimals (0.41458 vs 0.41459) — coincidence,
  confirmed by inspecting the full precision, not a caching bug.
- **The 32 configurations rank almost identically across backbones (r = 0.949).** The
  hyperparameter landscape is essentially backbone-independent, which is further evidence
  that the head and the aggregation, not the features, are what these sweeps are measuring.
  The one disagreement is head capacity: MRI-CORE prefers `hidden=256`, OrthoFoundation
  prefers a bare linear head.

## What actually moved the metric

Marginal medians over the 32 architecture-1 configurations:

| Axis | Setting | Median BCE | Spread |
|---|---|---:|---:|
| head capacity | 256 hidden vs linear | 0.4188 / 0.4226 | 0.0038 |
| slice jitter | keep 100% vs 70% | 0.4192 / 0.4221 | 0.0029 |
| dropout | 0.1 vs 0.3 | 0.4192 / 0.4220 | 0.0028 |
| series dropout | 0.15 vs 0.0 | 0.4207 / 0.4221 | 0.0014 |
| MixUp | 0.4 vs 0.0 | 0.4215 / 0.4210 | 0.0005 |

**Read this table as a single negative result, not as a ranking.** Every spread in it —
including the largest, 0.0038 — is at or below the ~0.003 reproducibility floor measured
later (see Limitations). Individually, "a hidden layer helps", "slice jitter hurts" and
"MixUp does nothing" are not claims this evidence can support, and an earlier version of
this file stated the first two as findings; they are withdrawn.

What the table does support is the aggregate: **no setting of any augmentation or
regularisation axis escapes a 0.004 band**, against an architecture gap of 0.021, a
fine-tuning gap of 0.012, and a gap to baseline of 0.082. The plan expected slice-sampling
jitter and series dropout to be the most valuable augmentations available; on frozen
features with 3,479 studies, none of these knobs matter enough to measure.

There is a mechanical reason to expect exactly that here, and it is worth separating from
the noise argument: with fixed pooling over the whole slice stack there is **no encoder
being regularised**, so most of these augmentations can only remove evidence rather than
prevent memorisation. That reasoning does not extend to architectures 3, 5 and 6, where a
trainable encoder does exist — and where, notably, augmentation was never ablated at all.

## Layout

```
common/       dataset, protocol, feature extraction, augmentation, ensembling
01_frozen_pooling/     architecture 1 — fixed [mean, p90, max] hierarchy      (frozen)
02_series_attention/   architecture 2 — 12 pathology queries over series      (frozen)
03_finetuned_2p5d/     architecture 3 — partial fine-tune + 2.5D context      (pixels)
04_hierarchical/       architecture 4 — slice + series Transformers           (frozen)
05_volumetric/         architectures 5 & 6 — 3D medical and video encoders    (pixels)
features/     per-slice frozen features (memory-mapped float16)
```

Architectures 1, 2 and 4 read cached frozen features and see *every* slice; 3, 5 and 6 read
pixels and can only afford a stratified sample per epoch. That distinction explains most of
the runtime differences and is why architecture 4 was deliberately built on frozen features
(see its README).

Slice cache: 819,076 slices at 224 px across all 4,407 studies, on local NVMe
(`KNEE_SLICE_CACHE`, default `/mnt/data01/knee_cache`). It is not beside the code because
the repository sits on a CIFS share that is near capacity and slower to read, and CIFS
cannot hold a symlink to redirect it.

## Reproducing

```bash
python common/dataset.py 224 40          # build the slice cache (I/O bound, ~2.5 h)
python common/features.py mri_core       # per-slice features (~5 min on one 4090)
python common/features.py orthofoundation
cd 01_frozen_pooling  && python train.py --sweep                    # ~5 min/config
cd 02_series_attention && python train.py --sweep                   # ~3 min/config
cd 04_hierarchical    && python train.py --sweep --epochs 30        # ~5 min/config
cd 03_finetuned_2p5d  && python train.py --compare --workers 6      # ~25 min/config
cd 05_volumetric      && python train.py --compare --workers 6      # ~7 min/config
python common/ensemble.py                # after any trainer run with --save-preds

cd 03_finetuned_2p5d && ./run_studies.sh # 15 Optuna studies, ~26 h; then:
cd 03_finetuned_2p5d && python summarize.py --importance
```

**Run one at a time.** Two pixel-level trainers sharing the 4090 do not merely halve
throughput — the first attempt deadlocked, and `persistent_workers=True` plus 6 workers was
needed to make the fine-tuning script stable at all.

Two environment quirks for the tuning scripts: the conda `sqlite3` needs
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib`, and the Optuna database must live on local disk
(`KNEE_OPTUNA_DIR`, default `/mnt/data01/knee_optuna`) because SQLite locking does not work
on the CIFS share this repository sits on.

## Limitations

- **One split, one seed**, with a measured reproducibility floor of about **0.003 BCE**.
  When each architecture's best configuration was retrained standalone for the ensemble,
  architectures 3, 4 and 5 reproduced their sweep numbers *exactly*, but architecture 1's
  best configuration read 0.4176 instead of 0.4146 — and three standalone repeats then
  returned 0.417628 bit-for-bit. So a run is perfectly reproducible given identical process
  state, and shifts by ~0.003 when it inherits a different one (the plausible mechanism is
  allocator and kernel-selection state left by earlier configurations in a sweep process,
  which changes reduction order).

  **This is larger than most of the effects the sweeps measured.** Architecture 1's entire
  32-configuration augmentation sweep spans 0.0038; architecture 4's depth and consistency
  marginals span 0.00005. Those are at or below the floor and should be read as "no
  measured effect", not as rankings. The conclusions that clear it comfortably are the ones
  reported above: fine-tuning (0.0121), fixed-vs-learned aggregation on frozen features
  (0.021-0.052), 2D-vs-volumetric (0.028+), and the ensemble gain (0.0070).
- **Val BCE and gold AUC do not agree.** Architecture 1's best-BCE configuration is not its
  best-AUC configuration. Selection follows BCE as instructed; the AUC column is reported
  so the divergence is visible rather than hidden.
- **Gold AUC rests on 58 studies** with 9-35 positives per finding. The earlier frozen work
  measured a bootstrap interval of 0.10 AUC on this cohort, which is wider than every gap
  in the table above.
- **Architecture 3's four configurations are not a sweep.** Its top three sit within 0.0017
  BCE, which is inside seed noise; only its frozen-vs-fine-tuned gap is comfortably outside.
  `unfreeze` and head `mode` were never crossed, so the best cell (`unfreeze=8, query`) is
  untested.
- **Architecture 4 is 9 of 16 cells**, stopped once the pattern was clear and the GPU was
  contended. Its depth marginals come from an unbalanced design.
- **The volumetric family ran at 16 x 112 x 112**, against 224 px in-plane for the 2D
  architectures. Part of its deficit is resolution rather than architecture, and this
  experiment cannot separate the two.
- **Val BCE and gold AUC disagree, consistently.** Architecture 4 loses to architecture 1 on
  BCE (0.4178 vs 0.4146) while beating it on gold AUC (0.8432 vs 0.8358); the same inversion
  appears inside architectures 1 and 3. Selection follows BCE as instructed, but the two
  metrics are answering different questions — imitating the teacher well is not the same as
  diagnosing well, and the teacher is itself only ~0.72 on gold from images.
