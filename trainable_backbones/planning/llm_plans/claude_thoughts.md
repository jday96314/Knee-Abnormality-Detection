I'll check the current state of MRI-specific pretrained encoders before answering, since that's the piece most likely to have moved recently.That co-plane attention paper is almost exactly your problem setup. Let me pull the details.Before the architectures — one thing that shapes everything else.

## Your 58 golden examples can't rank models

With 58 studies across 12 multi-label classes, the rarer classes will have single-digit positive counts. Bootstrap CIs on per-class AUC will be roughly ±0.10–0.15, and macro AUC won't be much tighter given the correlated errors. That set can tell you "this model is broken" or "this model is roughly sane." It cannot tell you whether architecture A at 0.89 beats architecture B at 0.92.

This also applies to your teacher: if the 0.92 macro AUC was measured on those 58, it has the same CI. Worth knowing before you treat 0.92 as the target to beat.

Practical consequence: do model selection on held-out soft labels with patient-grouped k-fold, and use the golden set only as a gate and calibration check. And seriously consider spending annotation budget expanding it — 58 → 300 buys you more decision-making capability than any architecture change on this list. Prioritize golden labels on cases where the doctor and the model ensemble disagreed; those are the informative ones.

## Reference point

The closest published analogue is CoPAS, which covered the same 12 knee abnormalities across multi-sequence MRI from 1,748 subjects and reached 0.812 macro AUC, training on 773 patients with arthroscopy-confirmed labels and using ResNet3D-18 as the shared encoder. Notably, performance dropped from 0.812 to ~0.72 on external centers, which the authors attribute largely to fewer available sequences. That drop is the single most transferable lesson for you given your 3–15 series spread.

## The architecture ladder

Your problem is hierarchical multiple-instance learning with distillation. Slices are instances within a series; series are instances within a study; only the study has a label. Almost all the design leverage is in the two aggregation stages and in the pretraining, not in which backbone you pick.

**Baseline (build this first, it's a day of work).** MRNet-style: 2D ResNet over slices, max-pool to series, logistic regression over series. Gives you a floor and catches preprocessing bugs. Expect ~0.80–0.85 against soft labels.

**Frozen foundation-model features + trained aggregation head.** Extract per-slice embeddings from a frozen MRI-pretrained encoder, then train only the aggregation stack. Candidates worth benchmarking: MRI-CORE, pretrained on 6M+ slices from 110,000 volumes across 18 body locations; Triad, trained on 131,170 3D MRI volumes with encoders from 31M to 11.8B params; or plain DINOv2/v3 as a strong general-purpose control. At 4,000 studies this decouples "learn good MRI features" (needs millions of images) from "learn to aggregate" (needs thousands), which is exactly the split your data supports.

The practical win here is underrated: cache the frozen features once — 4,000 × 5 × 30 ≈ 600k slice embeddings — and your aggregation experiments then run in seconds instead of hours. That's how you actually search the aggregation design space on a real budget.

**End-to-end 2.5D encoder + hierarchical attention.** This is my pick for the best single model:

![](claude_diagram.png)

Specifics that matter in that stack:

- **Encoder:** feed 3 adjacent slices as RGB channels. You get local through-plane context while keeping ImageNet or MRI-pretrained 2D weights usable, which matters enormously at n=4000. ConvNeXt-T, ResNet-50, or a ViT-S all work.
- **Slice aggregation:** gated attention MIL (Ilse et al.) or a 2–4 layer transformer with slice-position encoding. Attention beats max-pooling here because most abnormalities span several slices and max is noisy; but keep max as an ablation since it encodes "visible on at least one slice," which is a real prior.
- **Series aggregation:** this is where your clean sequence labels earn their keep. Each series token gets a learned embedding of (plane, fluid-sensitive, fat-suppressed) added before attention. That lets one shared encoder learn plane-specific behavior without CoPAS's separate-encoder-per-plane approach — the authors flag that design as their main source of parameter bloat and overfitting risk.
- **Heads:** shared trunk, 12 sigmoids. CoPAS found the multi-task framing specifically helped their rarest class by letting it borrow representation from correlated abnormalities.

**3D CNN branch (ResNet3D-18 or X3D, Kinetics-init).** Worth training because it's a genuinely different inductive bias and will ensemble well, not because it'll win solo. At your data scale it's more overfit-prone and can't exploit 2D pretraining as effectively.

**CoPAS-style cross-plane attention.** They rotate volumes to synthesize orthogonal views and cross-attend, specifically to compensate for anisotropic slice thickness. Clever and their ablations support it — but it assumes a fairly fixed sequence set, and you have 3–15 series of varying types. I'd defer this until the simpler stack is saturated.

## What I'd expect to win

A frozen or LoRA-tuned MRI-pretrained slice encoder, attention MIL over slices, sequence-conditioned attention over series, multi-task heads, distilled on soft labels, 5-fold ensembled. I'd put it at roughly 0.88–0.93 macro AUC against held-out soft labels, i.e. approaching the teacher rather than clearly beating it.

On beating the teacher: it's possible, but only if the teacher's errors are high-variance and roughly unbiased — the student averages them out across 4,000 examples and its inductive bias fills in. It cannot fix systematic teacher bias. If your doctor and model ensemble ever disagreed, that disagreement is recorded signal: use it as a per-example weight, and it tells you which regime you're in.

## Distillation details worth getting right

Train on the soft probabilities directly with soft-target BCE. Don't threshold to hard labels — you'd be discarding the teacher's uncertainty, which is the most useful thing it gives you. Skip focal loss here; it interacts badly with soft targets since it was designed to reweight confident-vs-hard *binary* labels. Handle imbalance with `pos_weight` instead. MixUp works unusually well with soft targets and is nearly free.

## Things that will probably outrank architecture choice

1. **More soft-labeled data.** If the teacher can label cheaply, 4k → 20k beats every architecture change on this list. Highest-ROI move by a wide margin.
2. **Per-series intensity normalization.** MRI has no HU equivalent. Percentile-clip and z-score per series. This is the most common silent performance killer.
3. **Series dropout augmentation.** Randomly drop series during training. This directly targets CoPAS's 0.812 → 0.72 external drop, which came from test sets having fewer sequences than training. Given your 3–15 spread, you need the model robust to missing sequences by construction.
4. **Laterality.** Check `ImageLaterality` before you reach for horizontal flips. MCL versus LCL is literally medial versus lateral — a careless flip destroys the distinction for two of your twelve classes. Either normalize all knees to one side or drop horizontal flips entirely.
5. **Resample using `PixelSpacing`**, not by resizing to fixed pixel dimensions, or lesion scale varies with scanner FOV.
6. **Group splits by patient**, not study, if anyone appears more than once.

If you want a concrete sequencing: baseline first, then cache frozen foundation-model features and sweep aggregation designs cheaply against held-out soft labels, then fine-tune the best two end-to-end and ensemble. The 3D branch is worth one run purely as ensemble diversity.