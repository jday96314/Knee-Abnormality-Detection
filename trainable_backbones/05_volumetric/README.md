# Architectures 5 and 6 — pretrained 3D encoders per series

The plan lists these separately — #5 "pretrained 3D MRI encoder", #6 "3D/video/CoPAS-style
diversity branch" — but structurally they are one family: a volumetric encoder turns a whole
series into one vector and the study head is unchanged. They differ only in which
pretraining the encoder carries, so they share a directory and a `--encoder` flag.

Each series becomes a fixed 16 x 112 x 112 volume: depth is *resampled* through the stack
rather than cropped, because knee series are strongly anisotropic and cropping to a fixed
depth would discard a different fraction of the knee in every study.

The study head is architecture 1's fixed per-plane mean/max, not the plan's 12 pathology
queries — reusing the settled head keeps the encoder the only thing under test.

## Result

| encoder | pretraining | unfreeze | val soft BCE | gold AUC | best epoch | minutes |
|---|---|---:|---:|---:|---:|---:|
| **R3D-18** | **Kinetics-400 video** | **1** | **0.4427** | **0.8089** | 4 | 7.3 |
| 3D ResNet-18 | MedicalNet (3D medical) | 1 | 0.4555 | 0.7529 | 11 | 6.6 |
| R3D-18 | Kinetics-400 video | 0 | 0.4584 | 0.7688 | 8 | 6.8 |
| 3D ResNet-18 | MedicalNet (3D medical) | 0 | 0.4678 | 0.6709 | 10 | 6.3 |
| *baseline (train mean)* | — | — | *0.4969* | — | — | — |
| *architecture 3, for scale* | — | — | *0.4064* | *0.8630* | — | — |

## What it shows

**1. The whole family is well behind the 2D hierarchy.** The best volumetric model (0.4427)
loses to architecture 3 by 0.036 and to frozen-feature architecture 1 by 0.028 — gaps far
larger than any within-architecture tuning effect measured anywhere in this repository. On
this dataset, per-slice 2D encoding with hierarchical pooling is simply the better use of
the same pixels. That matches the plan's expectation that #5 deserves "one serious run, not
an entire initial sweep", and this is that run.

**2. Kinetics video pretraining beats MedicalNet 3D medical pretraining — at both unfreeze
levels.** 0.4427 vs 0.4555 unfrozen, 0.4584 vs 0.4678 frozen, and by a large margin on gold
AUC (0.809 vs 0.753). This inverts the plan's framing, which identified the MRI-specific 3D
encoder as "the main wildcard with a realistic chance of beating the hierarchical 2D system"
and cast the video model as a mere diversity branch judged on ensemble gain.

The plausible reading is that *domain* match matters less than *representation quality*.
MedicalNet was trained on a small, heterogeneous collection of 3D medical volumes; Kinetics
is enormous. Whatever a video network learns about spatial texture and continuity along a
third axis apparently transfers to slice-to-slice continuity better than a weak
medical-domain 3D prior does. The frozen MedicalNet result (0.671 gold AUC) is barely above
chance-adjusted usefulness and is the weakest number in the entire ladder.

**3. Partial unfreezing helps both, and it helps the weaker one more.** MedicalNet gains
0.0123 BCE from unfreezing one stage, Kinetics 0.0157. Consistent with architecture 3, where
unfreezing was the single largest effect measured. The pattern across the whole project is
now unambiguous: **letting the encoder adapt is worth more than any aggregation, pooling or
augmentation choice tested.**

**4. Best epoch is early and the curves are flat.** The best Kinetics configuration peaked at
epoch 4 of 12 and drifted worse afterwards; frozen MedicalNet needed 10. With ~4M trainable
head parameters over 3,479 studies, these overfit quickly once the encoder joins in.

## Judged on ensemble gain

The plan asks that this branch be judged on ensemble contribution rather than solo score,
since "different volumetric inductive biases may be valuable even if the model itself is
weaker". `common/ensemble.py` evaluates logit-space averages over the saved per-architecture
predictions; see the top-level README for the outcome.

## Running it

```bash
python train.py --compare --epochs 12 --workers 6
python train.py --encoder r3d18 --unfreeze 1 --save-preds arch5
```

~35 s/epoch on an idle 4090; ~7 min per 12-epoch configuration. MedicalNet weights come from
MONAI (`pip install monai`, downloaded on first use); R3D-18 from torchvision.

## Limitations

- **One resolution, one depth.** 16 x 112 x 112 was chosen to fit the compute budget, and it
  is a real handicap: architecture 3 sees 224 px in-plane. Some of the gap is resolution
  rather than architecture, and this experiment cannot separate the two.
- **`unfreeze` only tested at 0 and 1 stage.** Both encoders improved with 1; neither was
  pushed further, and architecture 3 kept improving out to 8 blocks.
- **No 3D-specific augmentation.** The plan suggests bias fields, coherent 3D affines and
  resolution degradation. Only mild in-plane affine plus intensity was used, matching
  architecture 3 so that the encoder comparison stays clean.
- **MedicalNet may be handicapped by preprocessing.** It expects intensity conventions from
  its own pipeline; slices here are 8-bit windowed PNG-equivalents scaled to [0, 1]. That is
  a plausible partial explanation for its poor showing and was not investigated.
