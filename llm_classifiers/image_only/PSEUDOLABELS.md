# Image-only pseudo-labels: what was generated, and where it is weaker than it could be

Soft targets for all 4,407 training studies, produced from MRI pixels alone by
`google/medgemma-1.5-4b-it`, for pretraining a lightweight image-only model. Written by
`predict_all_studies.py`.

This document exists to make the *accuracy* compromises explicit. The run itself was clean
— 42,298 cells, zero ending in failure, 4,407/4,407 rows complete, 2.14 h wall-clock. The
compromises below are design choices, most of them made under time pressure, and several
are worth revisiting before these targets are trusted too far.

## What was generated

| | |
|---|---|
| Files | `predictions/image_only_predictions.csv`, `..._rank.csv`, `..._meta.json` |
| Rows | 4,407 (every training study; 58 have gold labels, 4,349 do not) |
| Schema | identical to `../text_only/predictions/text_only_predictions.csv` — `source_row`, `StudyInstanceUID`, `<finding>`, `<finding>__sample_std` |
| Teacher | mean of `v3_two_stage_tta_slices3_t07` and `v2_two_stage_t07_x5` |
| Measured accuracy | **0.7012 macro AUC** on the 58 gold studies |

Both members describe the images in free text, then score that description on an ordinal
word scale; they differ in slice-selection TTA (3 phase-shifted views) versus sampling
(5 seeded samples at T=0.7). `_rank.csv` holds the same ordering rescaled to within-cohort
percentiles.

Reproduce with:

```bash
python predict_all_studies.py --config best \
  --image-cache-dir artifacts_pseudolabel/image_cache \
  --endpoint http://mlserver3:8000/v1,32 --endpoint http://mlserver2:8000/v1,24
```

## Per-finding reliability — read this before training

The teacher is very unevenly good. Treating all twelve findings alike is the single most
likely way to misuse this file.

| Finding | AUC | Verdict |
|---|---:|---|
| Effusion | 0.919 | Trustworthy |
| Medial OA | 0.860 | Trustworthy |
| Lateral OA | 0.746 | Usable |
| PF OA | 0.742 | Usable |
| Contusion | 0.711 | Usable |
| Lateral Meniscus | 0.709 | Usable |
| Synovitis | 0.697 | Weak |
| Fracture | 0.643 | Weak |
| Medial Meniscus | 0.633 | Weak |
| MCL | 0.596 | **Near noise** |
| Baker's | 0.590 | **Near noise** |
| ACL | 0.568 | **Near noise** |

Weight the distillation loss per finding, or drop the bottom three. At 0.57 the ACL column
is close to a random ranking, and training against it teaches the student to imitate noise.

## Where this is suboptimal, roughly by how much it cost

### 1. The teacher was chosen on macro AUC, which was the wrong criterion

The largest and most avoidable error. Two candidate ensembles measured as statistically
identical on macro AUC — 0.7093 versus 0.7092, p=0.97 — so the cheaper one was shipped
(35,256 requests instead of 330,525). They are *not* equivalent per finding:

| Finding | shipped | dropped alternative |
|---|---:|---:|
| Fracture | 0.766 | **0.821** |
| Contusion | 0.713 | **0.771** |
| PF OA | 0.740 | **0.792** |
| ACL | 0.532 | 0.572 |
| Effusion | **0.908** | 0.843 |
| Medial OA | **0.797** | 0.722 |
| Baker's | **0.719** | 0.659 |

The offsetting strengths cancel in the macro average and hide a real difference. It
matters concretely: fracture is one of the few findings where the image channel beats the
report channel, so trading away 0.055 of fracture AUC directly shrank the measured blend
benefit against the text predictions (+0.0066 delivered, against +0.0110 projected from
the dropped config).

**If revisiting:** select the teacher on per-finding complementarity with whatever the
targets will be blended against, or ship both ensembles as separate channels and let the
student weight them. A macro-equivalent swap is not a blend-equivalent swap.

### 2. Everything was selected and evaluated on the same 58 studies

There is no held-out set anywhere in this pipeline. The condition sweep, the ensemble
choice, the blend weight and the reported AUC all come from the same 58 cases. The
intervals quoted are case-bootstrap only and do not include selection uncertainty, so
0.7012 is optimistic by an unknown amount.

With 9–35 positives per finding, per-finding AUCs carry roughly ±0.1 intervals. Treat the
reliability table above as an ordering, not a measurement.

### 3. The image channel is the weaker teacher for nine of twelve findings

Reports exist for all 4,349 unlabelled studies, and the text-only pipeline reaches ~0.90
macro AUC against this teacher's 0.70. Blending helps only slightly (+0.0066 at 10% vision
weight, p=0.002) because vision beats text on **effusion alone**.

**If revisiting:** the strongest pretraining targets are almost certainly per-finding
blends — text for ACL, MCL, the menisci and OA; image for effusion, and for fracture if
the dropped ensemble is restored. Per-finding blend weights were measured and were stable
across folds (0.53 effusion, 0.49 fracture, 0.00 ACL), though they gained only +0.003 over
a global weight on macro AUC.

### 4. Qwen3.6-27B was never run at scale

The cross-model ensemble (MedGemma ×2 + Qwen) reached **0.723** against MedGemma's 0.709 on
the 58 studies, and Qwen was distinctly better on findings this teacher is worst at: MCL
0.630 vs 0.544, Baker's 0.734 vs 0.666, fracture 0.847 vs 0.815. It was excluded purely on
cost — its share of the earlier plan was ~75% of the compute. The gain was not significant
(+0.014, p=0.29) but it targets exactly the weakest columns.

### 5. Sampling variance was not amortised

Both members sample at T=0.7, and the production run split across two GPU types. The
shipped predictions score 0.7012 where the sweep measured 0.7093 — a −0.008 drift
attributable to sampling and hardware nondeterminism. More samples per study would shrink
this; 3 and 5 were chosen for cost. The predictions are therefore not bit-reproducible from
a single machine, though ordering is stable.

### 6. The probability scale is uncalibrated

Values span 0.02–0.88 and are not calibrated to any prevalence. Nothing maps them to
P(finding present); they are a monotone score. `_rank.csv` at least gives a uniform
distribution. A student trained with soft-BCE on the raw column is fitting an arbitrary
scale — calibrating against the 58 gold labels first, or training on ranks, is safer.

Note the earlier digit-based teacher was far worse here (everything compressed into
0.79–0.90); the ordinal two-stage scale is a genuine improvement in usable spread, and was
an accidental benefit of the cost optimisation rather than a designed one.

### 7. The image spec was fixed early and never re-tuned

4 slices per series, 896 px, sagittal/coronal/axial fluid-sensitive plus one T1. The
downsample and context sweeps found rendering choices barely mattered — but those sweeps
ran when the baseline was degenerate (constant outputs), so they may have been measuring a
floor rather than a real insensitivity. Slice count and resolution have not been re-tested
against a teacher that actually discriminates.

### 8. Known-harmful TTA is still in the dropped config

Flip TTA damages ACL by −0.061, because a left-right flip on sagittal images swaps
anterior for posterior and puts the ACL where the PCL belongs. The shipped config uses
slice-phase TTA only and is unaffected, but `v3_digit_tta_slices_flip6` — the ensemble
member recommended for restoration in §1 — does include flips. Restoring it for fracture
means accepting that ACL cost, or building a flip-free variant.

### 9. Study metadata is unused

`PatientSex` is in `train.csv`, and `Laterality` is an allowlisted DICOM tag (present for
27 of the 58 gold studies). Neither enters the prompt. Laterality in particular determines
whether medial is on the image's left or right, which is exactly the distinction the model
fails at — medial vs lateral OA and meniscus are all mid-table or worse.

## Suggested order of work if revisiting

1. Restore the fracture/contusion signal — run `v3_digit_tta_slices_flip6` over all studies
   (~11 h across both servers) and ship it as a third channel rather than averaging it in.
2. Hold out gold studies before any further selection, even at the cost of a smaller
   selection set. Nothing here is confirmatory.
3. Blend per finding against the text predictions rather than globally.
4. Add Qwen for MCL, Baker's and fracture specifically, where it beats MedGemma.
5. Feed laterality into the prompt and re-test medial vs lateral discrimination.

## Related documents

- `REPORT.md` — the full condition sweep across three rounds and two models
- `README.md` — how the experiment harness works, and its throughput characteristics
- `predictions/image_only_predictions_meta.json` — machine-readable teacher AUCs, corrected
  to the shipped config
