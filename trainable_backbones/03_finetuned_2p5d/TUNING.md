# Architecture 3, tuned: 15 Optuna studies over 5 backbones x 3 heads

The first pass at architecture 3 was four hand-picked configurations on one backbone with a
single hard-coded augmentation bundle. This round replaces that with **one Optuna study per
(backbone, head) pair — 15 studies, 183 trials (91 run to completion, 92 pruned),
~26 GPU-hours** — searching optimisation and
augmentation jointly.

The headline: **the things that looked like findings in the four-configuration comparison
mostly dissolve, and the things that looked fixed turn out to matter.**

| | four hand-picked configs | 15 tuned studies |
|---|---|---|
| best val soft BCE | 0.4064 | **0.4000** |
| best gold AUC | 0.8630 | **0.8836** |
| backbones | 1 | 5 |
| heads | 2 | 3 |
| augmentation | 1 fixed bundle | 8 searched dials |

## Leaderboard

| backbone | head | val soft BCE | gold AUC | completed | pruned | total |
|---|---|---:|---:|---:|---:|---:|
| **ortho** | **plane** | **0.4000** | 0.8798 | 3 | 0 | 3 |
| ortho | slot | 0.4010 | **0.8836** | 3 | 0 | 3 |
| dinov2s | plane | 0.4024 | 0.8612 | 8 | 9 | 17 |
| mri_core | plane | 0.4024 | 0.8662 | 5 | 8 | 13 |
| ortho | query | 0.4026 | 0.8772 | 3 | 0 | 3 |
| radimagenet | query | 0.4035 | 0.8612 | 7 | 6 | 13 |
| radimagenet | plane | 0.4036 | 0.8700 | 8 | 6 | 14 |
| radimagenet | slot | 0.4042 | 0.8754 | 8 | 6 | 14 |
| mri_core | slot | 0.4049 | 0.8799 | 6 | 6 | 12 |
| dinov3s | slot | 0.4056 | 0.8719 | 5 | 10 | 15 |
| mri_core | query | 0.4070 | 0.8693 | 5 | 6 | 11 |
| dinov3s | plane | 0.4083 | 0.8570 | 8 | 6 | 14 |
| dinov3s | query | 0.4108 | 0.8509 | 8 | 7 | 15 |
| dinov2s | slot | 0.4111 | 0.8533 | 6 | 14 | 20 |
| dinov2s | query | 0.4153 | 0.8563 | 8 | 8 | 16 |

*Baseline (predict train mean) 0.4969. Trial counts are in the table because these studies
are budget-capped, not convergence-capped. **Only the `completed` column is a search
budget** — pruned trials were stopped after 2-4 epochs and cannot be a study's best. So the
real depth is 3-8 configurations per pair, not the 11-20 the totals suggest.*

## 1. The head choice is a non-finding

The original comparison found `plane` beating `query` and read it as a modelling insight.
Tuned, the head is worth almost nothing, and which head wins depends on the backbone:

| backbone | best head | head spread (best - worst) |
|---|---|---:|
| radimagenet | query | **0.0007** |
| ortho | plane | 0.0026 |
| mri_core | plane | 0.0046 |
| dinov3s | slot | 0.0052 |
| dinov2s | plane | 0.0130 |

Three different heads win on five backbones, and on RadImageNet all three land inside
0.0007 — a fifth of the reproducibility floor. The `slot` head ported from `07_public` is
neither the breakthrough nor a dud: it wins on dinov3s, ties on radimagenet and mri_core,
and clearly loses on dinov2s.

The honest reading is that **the study-level aggregator is not where the accuracy is**, and
the earlier plane-vs-query gaps were tuning artefacts — each head was being evaluated at a
learning rate and augmentation strength chosen for a different configuration.

## 2. Backbone barely matters either — until you get to ViT-L

The four well-searched backbones' best configurations span **0.4024 to 0.4056**, which is
the reproducibility floor. MRI-CORE (86M, MRI/DINO), DINOv2-S (22M, natural images) and
RadImageNet (24M, radiology, supervised) are mutually indistinguishable on the selection
metric. A 22M natural-image encoder matching an 86M MRI-pretrained one is the sharpest
negative result here, and it agrees with the volumetric family, where Kinetics video
pretraining beat MedicalNet.

OrthoFoundation's ViT-L is the exception: it takes the top three BCE slots and the best gold
AUC — **on three trials per study**. That is the least-searched backbone winning, which is
the strong form of the claim: 0.4000 is roughly a *floor* for a barely-sampled ViT-L, not a
tuned ceiling. It is also the only backbone whose search never pruned a trial, because
`MedianPruner` needs completed trials to form a median and three is not enough.

Capacity therefore does look like it helps, but the evidence is 9 trials total and confounded
with OrthoFoundation being musculoskeletal-pretrained. This is the one place where more
compute is clearly worth spending.

## 3. Augmentation is where the accuracy actually is

On frozen features every augmentation axis sat below the noise floor. With a trainable
encoder that reverses completely. fANOVA importances, top axis per study:

| study | top axes |
|---|---|
| dinov2s / plane | **intensity 0.29**, erase 0.19, bias 0.09 |
| dinov2s / query | **bias 0.23**, enc_lr_scale 0.16, erase 0.13 |
| dinov2s / slot | **bias 0.16**, intensity 0.12, lr 0.11 |
| radimagenet / plane | **intensity 0.25**, bias 0.14, unfreeze 0.12 |
| radimagenet / slot | **bias 0.23**, enc_lr_scale 0.13, intensity 0.13 |
| radimagenet / query | intensity 0.18, enc_lr_scale 0.18, bias 0.15 |
| dinov3s / plane | blur 0.13, noise 0.11, series_dropout 0.11 |
| mri_core / slot | **enc_lr_scale 0.20**, phase 0.10, epochs 0.09 |
| mri_core / plane | schedule 0.13, phase 0.13, epochs 0.11 |

Pixel-intensity augmentation — `intensity` (gamma/brightness/contrast) and `bias` (the smooth
multiplicative coil-inhomogeneity field) — is the top axis in 6 of 12 attributable studies.
That is a satisfying result rather than a generic one: **the two transforms that dominate are
the two that simulate MRI acquisition variation**, not the generic computer-vision ones. Both
were absent from the original bundle.

The winners choose *strong* settings. dinov3s/slot picked blur 0.98, bias 0.87, intensity
0.83, erase 0.80; mri_core/plane picked erase 0.99, geom 0.83, phase 0.77. The original
bundle was roughly geom 0.4, intensity 0.5, noise 0.3, phase 0.33 and nothing else — mild by
comparison, and missing bias, blur and erase entirely.

The exception is the two MRI-CORE studies where `enc_lr_scale`, `schedule` and `epochs` lead
instead. MRI-CORE is the one encoder already trained on this modality, so it plausibly needs
less synthetic acquisition variation and more careful control of how fast it moves.

## 4. The 2.5D context module does not earn its place

`kernel` was in the search precisely because `kernel=1` reduces `SliceContext` to an
identity-plus-projection — the ablation the original run never did.

**10 of 15 winning trials chose `kernel=1`**; only 2 chose 5. The Conv1D over adjacent slice
embeddings — the component that makes this architecture "2.5D" rather than 2D — is not
contributing. Architecture 3's advantage over architecture 1 should be attributed to
fine-tuning, which was cleanly isolated, and not to through-plane context.

## 5. Schedules and unfreezing

`onecycle` and `cosine` tie at 6 wins each, `cosine_warm` takes 3, and plain `step` never
wins. One-cycle was worth adding — it is half the winners — but there is no evidence it beats
cosine.

`unfreeze` lands high everywhere: 6-10 of 12 blocks for the ViT-B/S encoders, **19 of 24** for
OrthoFoundation, 3-4 of 4 stages for RadImageNet. The original run's cap of 8/12 was indeed
below the optimum, as its monotone trend suggested, and the prior that fitting 86M parameters
to 3,479 labels would overfit is not supported: the search reliably wants *more* adaptation,
not less, provided augmentation is strong enough to regularise it. Those two findings are the
same finding — heavy augmentation is what makes heavy unfreezing safe.

## Method

```bash
python tune.py --backbone mri_core --head slot --trials 40 --timeout-hours 2
python tune.py --list
python summarize.py --importance
./run_studies.sh          # all 15 pairs, sequentially
```

- **Searched (12 axes):** `unfreeze`, `lr`, `enc_lr_scale`, `wd`, `batch`, `epochs`,
  `schedule`, `dropout`, `kernel`, and 8 augmentation dials (geom, intensity, noise, bias,
  blur, erase, phase, series_dropout).
- **Fixed:** `n_series=6`, `n_slices=4`, `d_model=256`, 224 px, split and protocol per
  `common/protocol.py`. The first three set the cost of a trial; searching them would have
  spent the budget on speed rather than on the axes under study, and 6x4 is the original
  run's budget, which keeps these numbers comparable.
- **Sampler/pruner:** TPE with 8 startup trials; `MedianPruner` (4 startup, 2 warmup steps).
  Pruning stopped 92 of 183 trials early, which is what made 11-20 launched trials per
  study affordable inside the time cap.
- **Budget:** 1.5 h per study for the cheap backbones, 2 h for `mri_core` and `ortho`.

## Limitations

- **Budget-capped, not converged, and thinner than the totals suggest.** Only 3-8 trials
  per study ran to completion over a 12-dimensional space; the rest were pruned within a
  few epochs. OrthoFoundation's 3 trials are barely a search at all. No study was run to
  convergence and none should be treated as having found its optimum.
- **One seed per study, and the same seed across studies.** Every study seeds TPE with
  `seed=0`, so all studies evaluate the same startup candidates. This aids comparability but
  means cross-study agreement in chosen hyperparameters is an artefact, not convergence —
  visible in the table where dinov3s/query and dinov3s/slot report byte-identical parameters.
- **Best-of-N is optimistically biased.** Each reported number is the minimum over 11-20
  trials on the same validation set, so the leaderboard overstates what these configurations
  would score on fresh data. The bias is larger for the studies with more trials, which
  works *against* OrthoFoundation (3 completed) and in favour of dinov2s/plane (8 completed) — so the ViT-L result is, if
  anything, understated by this table.
- **Gold AUC is reported, never optimised.** Selection is by val soft BCE throughout. The two
  disagree materially: the best-BCE model (ortho/plane, 0.4000) is not the best-AUC model
  (ortho/slot, 0.8836), and mri_core/slot is 0.0025 worse on BCE than the joint leaders while
  scoring 0.8799 AUC. Since BCE measures imitation of a teacher that is itself only ~0.72 on
  gold from images, the AUC column is arguably the more meaningful one.
- **58 gold studies**, bootstrap interval ~±0.10 AUC — wider than every AUC gap in the table.
- **The OOM cascade cost real trials.** Before it was fixed, one out-of-memory trial left its
  model alive in the traceback frame, so every subsequent trial in that study died on a few
  MiB. The first `mri_core/plane` study was discarded and rerun; no reported number comes
  from a poisoned study.
