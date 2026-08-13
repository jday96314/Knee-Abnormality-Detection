A few of these are specific to your setup in ways that generic medical-imaging augmentation advice misses.

## The tension with your frozen-encoder plan

Worth resolving first, since it constrains everything else. If you cache frozen encoder features to make aggregation experiments fast, you've also frozen out every pixel-level augmentation — re-extracting features per epoch defeats the whole point of caching.

Two ways through:

- **Extract N augmented copies offline** (say 4–8 per study), sample among them at train time. Costs 4–8× storage on ~600k slice embeddings, which is still cheap, and preserves the fast iteration loop.
- **Augment only in the aggregation stack** during the search phase — series dropout, slice sampling jitter, embedding-level mixup, feature-space noise — then bring pixel-level augmentation back when you fine-tune end-to-end.

I'd do the second during architecture search and the first once you've picked a design.

## Highest value, in rough order

**Slice sampling jitter.** You're sampling a fixed number of slices from series ranging 30 to a few hundred. Jitter the sampling offset and stride every epoch instead of taking a deterministic uniform grid. This is free, generates enormous diversity, and costs nothing in label fidelity. On a 200-slice series it's effectively unlimited augmentation. If you do only one thing from this list, do this.

**Structured series dropout.** Targets your known failure mode directly — CoPAS lost ~0.09 AUC moving to centers with fewer sequences, and you have 3–15 series in-distribution. But drop *structurally*, not uniformly at random: sample which (plane × contrast) cells to keep rather than which individual series to keep. Dropping all coronals is genuinely destructive for MCL and LCL, whose evidence is largely coronal, and that's injecting label noise, not regularization. Keep at least one series per plane where available, and ramp the rate up over training rather than starting aggressive.

**MRI-physics intensity augmentation.** This is where MRI differs most from natural images and where generic pipelines underdeliver. The valuable ones:

- Random bias field (simulates B1 inhomogeneity — smooth multiplicative shading, very realistic, arguably the single best MRI-specific transform)
- Rician noise rather than Gaussian, since that's the correct model for magnitude images
- Gamma / contrast jitter
- Gibbs ringing and k-space spike artifacts, if your data has them

MONAI (`RandBiasField`, `RandRicianNoise`, `RandGibbsNoise`, `RandKSpaceSpikeNoise`) and TorchIO (`RandomMotion`, `RandomGhosting`) both cover these; torchvision doesn't. On motion artifacts specifically: MRNet deliberately kept noisy exams in training to match clinical reality, while CoPAS excluded them. Simulate motion only if your deployment population will include it.

**Mild affine.** Rotation ±10°, translation ±8%, scale 0.9–1.1. Apply the *same* spatial transform to every slice in a series — independent per-slice rotation produces anatomically incoherent volumes and quietly destroys the through-plane structure your 2.5D encoder is there to exploit. Bias fields should also be sampled per-volume, not per-slice, since the real artifact is spatially smooth in 3D.

**Mixup at the embedding level.** Pixel-space mixup across studies is incoherent when the studies have different series counts and types. Manifold mixup on the study embedding sidesteps that entirely and pairs well with soft targets — you're interpolating two teacher probability vectors, which is exactly the regime mixup was designed for.

## Avoid or handle carefully

**Horizontal flip.** Worth restating because it's the one people reflexively enable: it swaps medial and lateral, which is the entire distinction between two of your twelve classes. Normalizing laterality via `ImageLaterality` fixes your *input* consistency but does not make flipping safe — it reintroduces exactly the ambiguity you normalized away. Same for reversing sagittal slice order.

**Vertical flip** is anatomically meaningless here (superior/inferior). Skip.

**Heavy elastic deformation.** Cartilage thinning and meniscal tear morphology are small-scale findings. Low-magnitude elastic is a reasonable regularizer; anything aggressive can plausibly create or erase the finding you're trying to detect.

**Cutout / CutMix.** Poor fit for MIL. Erasing the region on the one slice carrying the evidence corrupts the label, and you have no way to know when that happened.

## One counterintuitive point

You're distilling from a teacher at ~0.92, so your targets already carry meaningful noise. Augmentation that risks changing the true label compounds that rather than regularizing against it. The usual intuition — "clean labels, so heavy augmentation is nearly free" — doesn't hold here. I'd run *lighter* augmentation than you would on a clean-label problem of the same size, and get your regularization instead from the frozen pretrained encoder, fold ensembling, and soft targets themselves.

## Testing them

With 4,000 noisy-label studies you can't reliably A/B individual transforms — the effect sizes are smaller than your fold-to-fold variance. Group them into three or four bundles (geometric, intensity, structural/composition, mixing) and test bundles on fixed folds. Ablate a bundle by removing it from the full set rather than adding it to nothing; removal ablations are better powered when transforms interact.