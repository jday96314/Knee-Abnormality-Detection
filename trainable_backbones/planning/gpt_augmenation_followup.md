Yes. For this problem I would make the augmentation pipeline **MRI-aware and hierarchical**, rather than using a generic ImageNet recipe. With only ~4,000 independently labeled studies, augmentation could matter quite a lot; knee-MRI work has found carefully tuned augmentation important, and recent MRI foundation-model work commonly uses spatial plus MRI-specific intensity/artifact perturbations. ([arXiv][1])

My rough priority would be:

| Augmentation                       |      Priority | Typical strength         |
| ---------------------------------- | ------------: | ------------------------ |
| Random slice/subvolume sampling    | **Very high** | substantial              |
| Series dropout                     | **Very high** | 10–30%                   |
| Slice dropout                      |      **High** | 5–20%                    |
| Small rotation/translation/scale   |      **High** | mild                     |
| Gamma/contrast/brightness          |      **High** | moderate                 |
| Bias-field augmentation            |      **High** | mild–moderate            |
| Gaussian/Rician-like noise         |   Medium-high | mild                     |
| Blur/sharpening/resolution changes |   Medium-high | mild                     |
| Motion/ghosting/Gibbs artifacts    |        Medium | mild                     |
| Anatomically correct flips         |        Medium | carefully                |
| MixUp                              |        Medium | probably embedding-level |
| CutMix                             |           Low | probably avoid initially |
| Synthetic/pathology generation     | Low initially | experimental             |

## 1. Random slice sampling may be your best augmentation

With series ranging from ~10 to several hundred slices, I wouldn't simply resize every series to a deterministic 32 slices.

During training, if the series contains more slices than your budget, randomly sample a **spatially contiguous or stratified subset**.

For example, for a 32-slice input:

```text
epoch 1:  slices  4,  7, 10, ... 97
epoch 2:  slices  1,  5,  8, ... 94
epoch 3:  slices  7, 11, 14, ... 99
```

This gives you multiple views of the same study and reduces dependence on exactly where the acquisition begins and ends.

I'd experiment with two schemes.

**Stratified jittered sampling:** divide the series into 32 equal spatial bins and randomly draw one slice from each. This preserves whole-knee coverage.

**Contiguous crop:** take, say, 70–100% of the physical coverage and resample it to your target number of slices.

I particularly like mixing them:

```text
70% stratified whole-series sampling
30% random contiguous crop
```

The contiguous crop encourages the MIL/attention mechanism to recognize abnormalities from incomplete acquisitions rather than expecting every anatomical landmark.

I would probably make this stronger than the conventional image augmentations.

---

## 2. Series dropout should be extremely useful

Your studies contain anywhere from **3–15 series**, so the model can easily learn brittle rules like:

> "I only recognize ACL tears when this exact sagittal PD-FS protocol is present."

During training, randomly remove entire series before study-level aggregation.

Something approximately like:

```text
p(drop each series) = 0.15
```

or randomly retain 70–100% of the available series.

This effectively trains an ensemble over possible MRI protocols.

I'd also test **targeted modality dropout**. If a study has several highly redundant sagittal sequences, dropping one is low risk and teaches useful invariance.

You can additionally mask the metadata embedding:

```text
series image embedding + [UNKNOWN_SEQUENCE]
```

occasionally, which prevents excessive dependence on perfectly normalized DICOM metadata.

Given the real-world variability of multiparametric MRI protocols, I think **series dropout could be one of the highest-value augmentations in the entire project**. MRI sequence heterogeneity is substantial enough that recent MRI-specific representation work explicitly targets robustness across different sequences and protocols. ([arXiv][2])

---

## 3. Slice dropout

Within a series, randomly remove perhaps 5–20% of slices.

I'd distinguish two variants:

```text
random individual slice dropout
```

and

```text
random contiguous slice dropout
```

The latter simulates incomplete anatomical coverage more realistically.

For example:

```text
original:

1 2 3 4 5 6 7 8 9 10 11 12 ...

augmented:

1 2 3 4 . . . . 9 10 11 12 ...
```

This makes pathological attention less reliant on a single canonical slice.

Don't interpolate the missing embeddings; simply mask them from attention.

---

# 4. Geometric augmentation: mild and synchronized

For every slice within a series, use **exactly the same spatial transformation**.

That's important. Independently rotating/cropping each slice would manufacture an incoherent 3D anatomy.

Reasonable starting ranges might be:

```text
rotation:      ±5–10°
translation:   ±5%
scale:         0.90–1.10
aspect ratio:  very little or none
```

Small rotations/shifts are commonly used successfully in medical-image learning, including knee MRI. ([arXiv][3])

I would avoid aggressive affine transforms. A 25° rotation or giant random-resized crop creates knees that aren't representative of acquisition variation and can remove precisely the small pathology you're trying to detect.

### Be very careful with horizontal/vertical flips

This matters more for your labels than for ordinary knee classifiers.

You have separate things such as:

* **medial** meniscus;
* **lateral** meniscus;
* medial OA;
* lateral OA.

A geometric reflection can interchange medial and lateral anatomy.

Therefore:

> **Do not blindly apply horizontal flips.**

A flip is only label-preserving if you've worked out what image coordinate axis it represents for that acquisition orientation.

You could make flipping a powerful augmentation if you simultaneously transform the labels:

```text
medial meniscus tear ↔ lateral meniscus tear
medial OA            ↔ lateral OA
```

and correctly transform orientation information.

But given sagittal/coronal/axial series and arbitrary DICOM presentation conventions, I would implement that from patient-space coordinates rather than assuming `HorizontalFlip()` means left-right anatomy.

---

# 5. MRI intensity augmentation is especially attractive

MRI intensities are **not standardized physical measurements** in the way CT Hounsfield units are. Sequence, scanner, coil, reconstruction and acquisition parameters can cause major appearance changes.

I'd therefore use moderate:

### Gamma

Something like:

```python
gamma = log_uniform(0.7, 1.4)
```

or initially even narrower:

```text
0.8–1.25
```

This changes tissue contrast without changing anatomy.

### Brightness/gain

Multiplicative scaling:

```text
0.8–1.2×
```

provided your preprocessing doesn't immediately normalize it away.

### Contrast

Perturb around the series mean/median.

### Intensity offset

Small offsets can also help depending on your normalization.

MRI work specifically targeting cross-contrast generalization has found substantial value in generating diverse nonlinear intensity transformations rather than assuming a fixed MRI appearance. ([arXiv][4])

I would apply these **consistently across a complete series**, not independently to individual slices.

---

# 6. Bias-field augmentation

This one I particularly like for MRI.

Generate a smooth multiplicative low-frequency field such as:

```text
             brighter
                ↓

0.8 ───────────── 1.25
       smooth
      gradient
```

and multiply the image volume by it.

That approximates coil sensitivity / B1-related intensity nonuniformity.

Current MRI robustness work routinely includes bias-field perturbation, and recent evaluations show that bias fields remain a meaningful failure mode even for MRI foundation models. ([arXiv][5])

I'd use it fairly often but keep most perturbations mild:

```text
p ≈ 0.2–0.4
```

with occasional stronger cases.

Again, make the bias field **spatially coherent through the series**.

---

# 7. Noise and acquisition-quality augmentation

I'd add small amounts of:

* Gaussian noise;
* Rician/noncentral-χ-like magnitude MRI noise if convenient;
* Gaussian blur;
* sharpening;
* resolution degradation.

For example:

```text
downsample to 60–100% resolution
→ upsample back
```

This should help scanner/reconstruction robustness without substantially altering anatomy.

Recent MRI pretraining pipelines have used Gaussian noise, smoothing, sharpening, gamma changes and bias fields together. ([arXiv][5])

### Don't overdo noise

Meniscal tears, cartilage defects and subtle ligament abnormalities can be extremely fine features.

I'd bias the distribution heavily toward:

```text
no corruption
or
barely noticeable corruption
```

rather than making half your images look awful.

---

# 8. MRI artifacts are interesting but second-wave experiments

Once you have a strong baseline, I'd try mild simulations of:

* motion;
* Gibbs ringing;
* ghosting;
* k-space spike artifacts;
* partial Fourier / resolution degradation.

These correspond to genuine MRI failure modes rather than arbitrary computer-vision corruptions.

Recent robustness testing suggests different MRI foundation encoders can be surprisingly sensitive to bias field, Gibbs ringing, ghosting and k-space spikes. ([arXiv][6])

But I would not start here. Correctly simulating artifacts takes more work, and unrealistic artifacts can easily hurt.

A simple mild blur + noise + bias-field pipeline probably gets most of the easy gain.

---

# 9. I'd be cautious with RandomResizedCrop

This is one of the places I'd depart substantially from ImageNet training.

Something like:

```python
RandomResizedCrop(scale=(0.08, 1.0))
```

would be horrifying here.

A tiny ACL tear or meniscal lesion can simply disappear.

If you're using DINO/ConvNeXt with a fixed input size, I'd instead do:

```text
resize preserving aspect ratio
+ modest random crop
```

perhaps retaining at least:

```text
90–95% of the original field of view
```

unless you've established that your scans include lots of irrelevant border.

---

# 10. MixUp: more appealing than CutMix

Your soft labels actually make MixUp conceptually natural.

For two studies:

[
x' = \lambda x_A + (1-\lambda)x_B
]

[
y' = \lambda y_A + (1-\lambda)y_B
]

But pixel-space mixing of MRI volumes creates an anatomically impossible double knee.

I'd therefore try **representation-level MixUp** instead:

```text
series/study embedding A
          +
series/study embedding B
          ↓
         MixUp
          ↓
classification head
```

Something like:

```text
Beta(0.2, 0.2)
```

would mostly produce samples close to one real example, rather than 50/50 Franken-knees.

This is especially interesting because your teacher targets are already soft probabilities.

I'd test it, but probably after the basic pipeline.

---

## CutMix worries me more

Suppose you paste a rectangular region from a positive ACL tear study into a negative study.

The pasted region may:

* not contain the ACL;
* contain only part of the lesion;
* come from anatomically incompatible slice positions;
* produce a positive interpolation target anyway.

The relationship between crop area and pathology label simply doesn't make much sense.

There is evidence that CutMix/MixUp can improve some medical classification problems, but the effects vary by task. ([arXiv][7])

For this workload I'd expect:

> representation MixUp > pixel MixUp > CutMix.

---

# 11. Series-level MixUp/CutMix is probably a bad idea

I'd avoid constructions such as:

```text
study A:
    sagittal A
    coronal A
    axial B
```

with interpolated labels.

If B has an ACL tear, what should the resulting study's ACL target be?

The answer depends on whether the transplanted series actually depicts the ACL pathology, so the usual MixUp assumption breaks.

Series dropout is much cleaner.

---

# 12. Preserve correlations between slices

One rule I'd enforce throughout the pipeline:

### Series-consistent augmentations

Use one sampled transform for:

* rotation;
* translation;
* crop;
* gamma;
* bias field;
* blur parameters.

This yields:

```text
slice 1 ─┐
slice 2  │
slice 3  ├── SAME augmentation parameters
...      │
slice N ─┘
```

Some noise can reasonably be independently realized per slice, but its **strength** should generally be common to the series.

Otherwise you're training on rapidly changing contrast/geometry that isn't physically representative of an MRI acquisition.

---

# 13. Don't necessarily augment every series identically

There's an important second level here.

I would make geometry coherent **within a series**, but intensity perturbations can differ **between series**.

For example:

```text
sagittal PD-FS:       gamma 1.15
coronal T1:           gamma 0.87
axial PD-FS:          gamma 1.04
```

That's plausible: different sequences and acquisitions genuinely have different contrast characteristics.

A single study-wide gamma transform would not capture this.

Likewise, series dropout naturally occurs independently.

---

# 14. Random protocol metadata corruption

Given the hierarchical architecture we discussed, I'd also augment the **metadata**:

```text
10%: hide SeriesDescription-derived sequence category
5%:  hide TE/TR
5%:  perturb continuous DICOM values slightly
```

I'd be especially willing to hide free-text-derived protocol metadata.

The idea is:

> metadata should help the model interpret images, not permit it to memorize scanner/site/protocol → diagnosis shortcuts.

This can be valuable if the dirty-label training set contains institutional correlations.

---

# What I'd actually start with

My first augmentation recipe would be quite restrained:

```text
STUDY
├── Series dropout                           p=0.20
│
└── EACH RETAINED SERIES
    ├── jittered stratified slice sampling   always
    ├── contiguous slice crop                p=0.25
    ├── random slice dropout                 p=0.20
    │
    ├── rotation ±7°                         p=0.50
    ├── translation ±5%                      p=0.50
    ├── scale 0.95–1.05                      p=0.30
    │
    ├── gamma 0.8–1.25                       p=0.40
    ├── mild contrast/gain                   p=0.30
    ├── smooth bias field                    p=0.25
    ├── mild Gaussian/Rician noise           p=0.20
    └── mild blur/resolution loss            p=0.15
```

I would **not initially use**:

```text
horizontal/vertical flip
CutMix
large crops
large rotations
elastic deformation
perspective transforms
strong color jitter
synthetic pathology
```

Then I'd ablate in groups rather than individual transforms:

1. **baseline**
2. * slice sampling/dropout
3. * series dropout
4. * mild geometry
5. * MRI intensity augmentation
6. * acquisition/artifact augmentation
7. * embedding MixUp

My prior is that **slice sampling/dropout + series dropout + MRI intensity variation** will account for most of the augmentation benefit. The common ImageNet-style spatial augmentations are probably secondary.

One other experiment I'd be particularly interested in with your setup is **consistency regularization**: present two differently augmented versions of the *same study* and penalize disagreement between their 12 predictions. With noisy teacher labels and only 4k studies, that may be more valuable than simply making the augmentation stronger, because it lets the model explicitly learn that protocol/slice/intensity changes should not change the diagnosis.

[1]: https://arxiv.org/abs/2010.01947?utm_source=chatgpt.com "A Comparative Study of Existing and New Deep Learning Methods for Detecting Knee Injuries using the MRNet Dataset"
[2]: https://arxiv.org/html/2508.07165v2?utm_source=chatgpt.com "Large-scale Multi-sequence Pretraining for Generalizable ..."
[3]: https://arxiv.org/pdf/2403.16499?utm_source=chatgpt.com "Self-Supervised Learning for Medical Image Data with ..."
[4]: https://arxiv.org/html/2411.17845v2?utm_source=chatgpt.com "CABLD: Contrast-Agnostic Brain Landmark Detection with ..."
[5]: https://arxiv.org/html/2604.11679v1?utm_source=chatgpt.com "Towards Brain MRI Foundation Models for the Clinic"
[6]: https://arxiv.org/html/2608.06613v1?utm_source=chatgpt.com "Do 3D Medical Foundation Models See Through MRI ..."
[7]: https://arxiv.org/pdf/2308.11902?utm_source=chatgpt.com "arXiv:2308.11902v1 [eess.IV] 23 Aug 2023"
