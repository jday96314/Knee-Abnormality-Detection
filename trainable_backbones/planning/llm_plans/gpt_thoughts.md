With this dataset size, I would **not** expect a large 3D network trained end-to-end from scratch to win. My first bet would be a **strong pretrained 2D image encoder + hierarchical learned aggregation over slices and series**, with explicit series metadata.

That is unusually well matched to your problem: the natural hierarchy is image → series → study, the number of images/series is variable, different abnormalities depend on different MRI sequences and orientations, and 4,000 studies is enough to train a modest aggregation network but not enough to learn MRI visual features from scratch.

### What I would test

| Priority | Architecture                                                               | Expected performance                              |
| -------- | -------------------------------------------------------------------------- | ------------------------------------------------- |
| **1**    | Pretrained 2D ViT/ConvNet → slice Transformer/MIL → series Set Transformer | **Best bet**                                      |
| **2**    | Pretrained MRI-specific 3D encoder per series → series Set Transformer     | Potentially best if good weights are available    |
| **3**    | 2.5D ConvNeXt/EfficientNet/DINO + attention MIL                            | Very strong, simpler baseline                     |
| **4**    | Video model per series (VideoMAE/Hiera/MViT) → study fusion                | Worth testing                                     |
| **5**    | MRNet-style slice CNN + max/attention pooling                              | Essential baseline                                |
| **6**    | Full 3D CNN/ViT trained from scratch                                       | I would expect this to lose                       |
| **7**    | VLM directly predicting the 12 labels                                      | Interesting auxiliary model, not my primary model |

## 1. The architecture I'd expect to win

Conceptually:

```text
                                   ┌─ Series 1 ─ slices ─┐
                                   │                    │
MRI study ─────────────────────────┼─ Series 2 ─ slices ─┼─ ...
                                   │                    │
                                   └─ Series N ─ slices ─┘


slice image
    │
    ▼
pretrained 2D vision encoder
    │
    ▼
slice embedding
    │
    │  + slice position / spacing
    ▼
small Transformer across slices
    │
    ▼
series embedding
    │
    │  + orientation
    │  + sequence type
    │  + fat saturation
    │  + TR/TE/etc.
    ▼
Set Transformer / cross-attention across series
    │
    ▼
12 pathology query tokens
    │
    ▼
12 sigmoid outputs
```

Recent HLIP work is especially relevant here. It explicitly organizes attention around the **slice → scan → study** hierarchy rather than treating an entire study as a flat pile of tokens. Their data include studies with variable scan counts and scans with highly variable numbers of slices; they report that hierarchical attention outperforms naive whole-study ViTs. ([arXiv][1])

### Slice encoder

I would benchmark at least:

* **DINOv3 ViT-S/B**
* **ConvNeXt V2 Base/Large**
* an MRI/medical encoder such as **MedImageInsight**
* possibly RAD-DINO
* **Decipher-MR**, if its released encoder is convenient for your data

DINOv3 is worth taking surprisingly seriously despite being natural-image pretrained. A 2026 evaluation across 2D and 3D medical tasks found it an exceptionally strong transferable representation and in several settings superior to medical-specific foundation encoders. Importantly, bigger DINOv3 models did *not* invariably do better, so I would test B/S rather than automatically going huge. ([arXiv][2])

MedImageInsight was explicitly trained across MRI as well as CT, X-ray and other medical modalities, so it is another sensible initialization. ([arXiv][3])

I suspect **pretraining quality will matter more than whether the backbone says CNN or ViT on the box**.

### Series encoder

For each series, preserve slice order. Something like a 2–4 layer Transformer with perhaps 512–768 dimensions is sufficient.

Give every slice:

```text
image_embedding
+ normalized_slice_position_embedding
+ physical_z_position_embedding, if available
```

Relative positional attention would be particularly appropriate because:

* one series might contain 20 slices;
* another might contain 200;
* slice spacing varies;
* absolute slice number itself has little meaning.

I would initially process perhaps **32–64 slices per series**, randomly/substantially oversampling during training if the series is longer. At validation you can run more/all slices or ensemble several subsamples.

For the 30-slice median, processing everything is perfectly reasonable.

---

# 2. Use pathology-specific attention

I think this is particularly important for your 12-label task.

Don't force the model to produce just:

```text
study -> one embedding -> Linear(12)
```

Instead introduce **12 learned abnormality query tokens**.

Each query can attend differently to the available series:

```text
ACL query
    -> strongly weights sagittal fluid-sensitive series

medial meniscus query
    -> sagittal + coronal

patellofemoral OA query
    -> axial + sagittal

effusion query
    -> multiple fluid-sensitive series

...
```

You don't need to encode those rules manually. Give it the metadata and let cross-attention learn them.

This should be considerably better suited to the task than ordinary mean pooling, because **the useful series is pathology-dependent**. Even old MRNet demonstrated this phenomenon: it trained separate models by series and abnormality and then learned pathology-specific combinations of the series predictions. MRNet itself used a 2D AlexNet on individual slices, max-pooled over slices, and combined three series at the examination level. ([PLOS][4])

Your proposed system can effectively be thought of as a much more capable modern MRNet.

---

## 3. I would test attention MIL versus a slice Transformer

These are close enough that I'd run both.

### A. Attention MIL

```text
DINO/ConvNeXt
    ↓
[slice embeddings]
    ↓
gated attention pooling
    ↓
series embedding
```

Very few parameters, robust with 4k examples.

You could even have 12 independent attention pools:

```python
slice_embeddings -> 12 attention distributions -> 12 series embeddings
```

This lets an ACL classifier and an OA classifier select entirely different slices.

I'd expect this to be an extremely difficult baseline to beat.

### B. Transformer across slices

```text
[slice embeddings + positions]
        ↓
2–4 Transformer layers
        ↓
pathology queries
```

It gains the ability to reason across adjacent slices.

That should particularly help findings where **persistence or changing morphology across consecutive slices is diagnostic**.

My prior would be:

> Transformer > attention MIL, but probably not by a gigantic margin with 4,000 studies.

The MIL model may actually win on rare labels because its effective capacity is much lower.

---

# 4. 2.5D is absolutely worth trying

A very cheap alternative is to give the 2D encoder neighboring slices:

```text
channel R = slice i - 1
channel G = slice i
channel B = slice i + 1
```

or perhaps 5–7 slices followed by a learned projection into three channels.

Then:

```text
2.5D encoder
→ attention across central-slice embeddings
→ series attention
→ labels
```

That gives every feature extractor some immediate through-plane information without paying for a real 3D network.

I'd probably test:

* 1 slice;
* 3 slices;
* 5 slices.

There is a potential disadvantage to abusing pretrained RGB filters by putting adjacent MRI slices into RGB channels. A cleaner variation is:

```text
each slice → same 2D encoder
3/5 adjacent embeddings → small temporal Conv1D/MLP
```

That preserves the pretrained encoder exactly.

---

# 5. MRI-specific 3D foundation models are the major wildcard

This is the architecture that I think has the highest chance of beating #1, **provided the pretrained representation transfers well to knee MRI**.

There has been substantial progress here recently.

PRISM was pretrained on **336,476 volumetric MRI scans across many sequences and anatomical regions**, specifically attempting to make its representations robust to sequence/acquisition variation. It reports strong performance across classification, segmentation and other MRI tasks. ([arXiv][5])

Decipher-MR was trained on **200,000 MRI series from >22,000 studies**, using self-supervised and report supervision, and is explicitly designed so downstream tasks can use lightweight heads on a frozen encoder. ([arXiv][6])

So a very compelling experiment is:

```text
MRI series
     ↓
pretrained MRI 3D encoder
     ↓
series vector
     ↓
Set Transformer over all series
     ↓
12 pathology queries
```

If PRISM/Decipher-MR transfer cleanly to musculoskeletal MRI, this could be excellent.

But I would distinguish that sharply from:

> "Train a 3D Swin/ResNet from scratch on 4,000 knees."

I have little confidence in the latter.

You would be asking it to learn:

* MRI textures;
* anatomy;
* slice continuity;
* protocol invariance;
* pathology;

from only 4,000 noisy study labels.

The pretrained 2D model already knows a huge amount of useful low/mid-level visual structure.

---

# 6. Video architectures

Your series are semantically rather similar to short videos, so I'd include one video model as an experiment:

```text
series
→ VideoMAE / MViT / Hiera
→ series embedding
→ study-level Transformer
```

The advantage is natural spatiotemporal modeling.

The disadvantage is that standard video pretraining has peculiar priors:

```text
time ≠ MRI slice axis
```

Motion and temporal correspondences in video are rather different from moving anatomically through a volume.

Consequently I would expect a natural-video-pretrained VideoMAE to **lose to DINOv3 + learned slice aggregation**, unless you can substantially adapt it.

It could make a valuable ensemble member even if slightly weaker.

---

# 7. Don't concatenate the entire study into a giant ViT

Suppose you have:

* 5 series;
* 30 slices/series;
* ~400 patch tokens/slice.

You're already at something like:

```text
5 × 30 × 400 = 60,000 tokens
```

at the median study.

And your long-tail studies get absurd.

More importantly, the model is being asked to discover structure you already know:

```text
pixels belong to slice
slices belong to series
series belong to study
```

Recent hierarchical 3D imaging work makes essentially this same observation: whole-study tokenization is computationally expensive, while explicit slice/scan/study structure provides a useful inductive bias. ([arXiv][1])

So I would compress at each level.

---

# 8. Series metadata may be disproportionately valuable

I would feed as much trustworthy DICOM information as you have into the series aggregation model:

```text
orientation:
    axial / sagittal / coronal

sequence:
    T1 / T2 / PD / etc.

fat suppression:
    yes / no

TR
TE
flip angle
slice thickness
slice spacing
pixel spacing
field strength
```

I'd turn continuous values into small MLP embeddings and categorical values into learned embeddings.

If `SeriesDescription`/`ProtocolName` is available, I would probably normalize it into explicit categories rather than simply treating arbitrary scanner text as categorical IDs.

For this problem, these aren't incidental metadata. They tell the network **what information a series is capable of showing**.

PRISM's results are also evidence that explicitly dealing with sequence variation matters substantially for MRI representation learning. ([arXiv][5])

---

# 9. Your noisy labels change what model size I would use

The training set is an interesting size:

> 4,000 studies isn't really 4,000 images.

At the median it is roughly:

```text
4,000 × 5 × 30
≈ 600,000 slices
```

So there is plenty of visual data.

But you only have **4,000 independent study-level supervision events**.

That distinction favors:

**large pretrained image encoder + small trainable aggregation network**

rather than:

**large randomly initialized study network**.

Initially I might:

1. freeze most/all of the image encoder;
2. train slice/series/study aggregation;
3. unfreeze the upper 25–50% of the encoder at low LR;
4. optionally fully fine-tune near the end.

Your 0.92-AUROC teacher is also good enough that I would use the **actual soft probabilities**, rather than thresholding them to create binary labels.

For example, BCE against:

```text
teacher probability = 0.84
```

contains considerably more information than converting it to:

```text
positive = 1
```

It also naturally makes dubious examples less aggressive supervision.

---

# 10. I would be very conservative with the 58 gold studies

Fifty-eight studies are not enough to train a sophisticated second-stage model.

They're much more valuable as a clean measure of whether an architectural change is genuinely improving the thing you care about.

You effectively have:

```text
58 × 12 = 696
```

binary decisions, but some abnormalities presumably have very few positives.

I would therefore be reluctant to repeatedly tune architectures against those 58; you'll overfit your research process to them extraordinarily quickly.

If possible, I'd use the noisy 4k for CV/model development and reserve most or all of the 58 as a **locked gold evaluation set**.

Once the architecture is settled, they could potentially be used for final calibration or very low-LR fine-tuning.

---

# The concrete sweep I'd run

If I wanted maximum information from ~6 experiments, I'd do:

### Model A — MRNet++

```text
DINOv3-B
→ per-slice embedding
→ max pooling within series
→ MLP over series
```

This establishes how much sophisticated aggregation matters.

### Model B — MIL

```text
DINOv3-B
→ per-slice embeddings
→ pathology-specific gated attention
→ pathology-specific series features
→ pathology-specific attention across series
```

**I expect this to be very strong.**

### Model C — hierarchical Transformer

```text
DINOv3-B
→ slice embeddings
→ 2-layer slice Transformer
→ series representations
→ 2-layer Set Transformer
→ 12 learned pathology queries
```

**My predicted winner.**

### Model D — ConvNeXt diversity model

Same as C but ConvNeXt V2.

This gives you substantially different inductive biases for an ensemble.

### Model E — 2.5D

```text
DINO/ConvNeXt + neighboring slices
→ attention MIL
→ study attention
```

### Model F — pretrained 3D MRI

```text
PRISM/Decipher-MR
→ series embedding
→ pathology-specific study Transformer
```

This is the one with the best chance of surprising Model C.

---

## My expected ordering

If I had to put probabilities on what ultimately gives your best single model:

**~40%**

> DINOv3/strong 2D foundation encoder + hierarchical slice/series Transformer

**~30%**

> MRI-specific pretrained 3D encoder + hierarchical series fusion

**~15%**

> 2D encoder + relatively simple pathology-specific attention MIL

**~10%**

> 2.5D CNN/ViT + MIL

**~5%**

> video architecture / other

And I would expect the eventual best submission/model to be an **ensemble of 2–3 of these**, probably something like hierarchical DINOv3 + MRI-specific 3D encoder + ConvNeXt/2.5D model.

The single architectural feature I would be most reluctant to omit is **pathology-specific attention over series**. With 12 abnormalities and a heterogeneous mixture of axial/sagittal/coronal and fluid/fat-suppressed sequences, the optimal evidence is sufficiently different for each finding that a single shared study representation seems unnecessarily restrictive.

[1]: https://arxiv.org/html/2505.21862v3 "Towards Scalable Language-Image Pre-training for 3D Medical Imaging"
[2]: https://arxiv.org/abs/2509.06467 "[2509.06467] Does DINOv3 Set a New Medical Vision Standard? Benchmarking 2D and 3D Classification, Segmentation, and Registration"
[3]: https://arxiv.org/abs/2410.06542 "[2410.06542] MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging"
[4]: https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699&utm_source=chatgpt.com "Deep-learning-assisted diagnosis for knee magnetic ..."
[5]: https://arxiv.org/html/2508.07165v2 "Large-scale Multi-sequence Pretraining for Generalizable MRI Analysis in Versatile Clinical Applications"
[6]: https://arxiv.org/abs/2509.21249 "[2509.21249] Decipher-MR: A Vision-Language Foundation Model for 3D MRI Representations"