# Architectural Paradigms and Optimization Strategies for Multi-Label Knee MRI Abnormality Detection

## The Clinical and Computational Topography of Knee MRI Workloads

The automated interpretation of Magnetic Resonance Imaging (MRI) for musculoskeletal diagnostics represents one of the most formidable challenges in contemporary medical computer vision. The specific workload under consideration—detecting 12 distinct knee abnormalities from a heterogeneous collection of MRI series—requires an architectural pipeline capable of navigating profound dimensional variability, extreme multi-label sparsity, and weak probabilistic supervision. To architect an optimal deep learning system, it is first necessary to deconstruct the specific structural and clinical characteristics of the provided dataset, as these parameters dictate the necessary inductive biases of the neural network.

The dataset is fundamentally hierarchical and volumetrically irregular. A single patient study contains a variable number of MRI series, ranging from 3 to 15, with a median of 5. These series are not uniform acquisitions; they encompass a highly heterogeneous mix of anatomical planes—specifically the sagittal, coronal, and axial views—as well as highly specialized tissue contrast weightings, such as fluid-sensitive and fat-suppression sequences. Pathologies manifest with unique, modality-specific visual signatures across these acquisitions. For instance, the integrity of the anterior cruciate ligament (ACL) is predominantly evaluated in the sagittal plane, whereas chondral defects or bone marrow edema are often only perceptible in fluid-sensitive, fat-suppressed coronal sequences. The availability of clean labels identifying these sequence types is a substantial asset, allowing for explicit architectural conditioning rather than forcing the model to blindly infer the acquisition physics from raw pixel data.

Beneath the series level, the spatial dimensionality presents a significant bottleneck. Each series contains a variable number of 2D image slices, with a median of 30 but characterized by a severe long-tail distribution extending into several hundred slices per series. Because MRI acquisition typically yields highly anisotropic voxels—where the spacing between sequential slices along the z-axis is substantially larger and more variable than the high-resolution in-plane x and y dimensions—traditional 3D Convolutional Neural Networks (3D CNNs) often struggle to extract coherent spatial features without severe interpolation artifacts. A meniscal tear might manifest as a sub-millimeter hyperintense signal across only two or three contiguous slices out of a 150-slice sequence. Consequently, the architecture must possess the capacity to dynamically isolate highly sparse pathological signals embedded within massive volumes of healthy background tissue.

Compounding this volumetric complexity is the diagnostic objective: a 12-class multi-label classification task. In a typical clinical cohort, a patient may present with one or two concurrent abnormalities (e.g., a medial meniscal tear accompanied by joint effusion), leaving the remaining 10 to 11 diagnostic labels entirely negative. This induces an extreme positive-negative class imbalance. If subjected to standard loss formulations, the optimization landscape will be entirely dominated by gradients originating from negative classes, systematically suppressing the rare, critical signals of actual anatomical abnormalities.

Finally, the supervisory signal defining this workload is bifurcated. The primary training corpus consists of approximately 4,000 examples featuring "soft, dirty" labels derived from a clinician-model ensemble, boasting a macro-averaged ROC AUC of 0.92. These continuous probabilistic targets provide a rich manifold of diagnostic uncertainty and inter-class correlation, acting as a natural regularizer during training. Conversely, a highly constrained set of 58 "golden" examples provides pristine, human-verified binary ground truth. This dual-supervision topology necessitates a phased meta-learning strategy, leveraging the vast probabilistic knowledge of the soft labels for feature extraction while utilizing the golden set for definitive decision-boundary calibration.

## Base Feature Extraction: Bridging the Medical Domain Gap

The foundation of the proposed architecture relies on the extraction of high-fidelity spatial features from individual 2D MRI slices. Given the limited primary training corpus of roughly 4,000 studies, training a deep neural network backbone from random initialization will inevitably result in severe overfitting. Transfer learning is strictly mandatory; however, the source of the pre-trained weights and the specific architectural family of the backbone are critical determinants of downstream clinical efficacy.

### The Inadequacy of Natural Image Priors

Historically, the dominant paradigm in medical image analysis has involved initializing 2D CNNs with weights pre-trained on the ImageNet database. However, the domain gap between natural images and radiological scans is immense. Natural images are characterized by sharp, distinct edges, high color variance, and macro-level object-centric compositions. By contrast, knee MRIs are strictly grayscale, suffer from low contrast-to-noise ratios, and require the precise identification of subtle, micro-level textural disruptions, such as a slight hyperintensity within the fibrocartilage of the meniscus.

When architectures rely on ImageNet priors for medical tasks, the network attempts to map radiological textures onto geometric feature detectors optimized for natural objects. This forces the model to unlearn its foundational weights before it can begin acquiring relevant medical representations, a process that is highly inefficient and prone to settling in sub-optimal local minima when training data is constrained.

### Foundational Vision Models and the Limitations of DINOv2

Recent breakthroughs in self-supervised learning have popularized massive foundational vision models, most notably DINOv2. Pre-trained on 142 million meticulously curated natural images using self-distillation and masked patch prediction, DINOv2 exhibits unprecedented out-of-the-box feature representations and remarkable cross-task generalizability. DINOv2 has demonstrated significant utility in certain medical imaging domains, particularly those that visually resemble natural photography, such as dermatological lesion classification (dermoscopy) or retinal fundus photography.

However, empirical evaluations of DINOv2 within highly complex, multi-modal MRI environments—such as glioma grading, abdominal organ segmentation, and musculoskeletal joint analysis—reveal substantial structural limitations. When applied to MRI datasets characterized by anisotropic voxels and non-optical tissue contrast, DINOv2's performance routinely trails behind models that have been explicitly pre-trained on medical data. While advanced adaptations like MM-DINOv2 attempt to circumvent these limitations by introducing multi-modal patch embeddings and full-modality masking objectives, the fundamental domain discrepancy remains a massive computational and representational hurdle. Transformer-based architectures, which underpin DINOv2, lack the inductive bias of translational equivariance inherent to convolutions, meaning they require exponentially larger datasets to learn the basic structural boundaries of human anatomy. Given the constraint of 4,000 training studies, deploying a natural-image foundational Vision Transformer (ViT) represents a suboptimal allocation of computational resources.

### RadImageNet: Domain-Specific Medical Pre-training

To achieve optimal convergence and feature expressivity for knee MRI analysis, the 2D feature extractor must be initialized with weights from RadImageNet. RadImageNet is a highly specialized, large-scale database comprising 1.35 million radiologic images, systematically annotated by board-certified radiologists. The corpus spans computed tomography (CT), MRI, and ultrasound modalities across numerous anatomical regions, including extensive musculoskeletal datasets.

The deployment of RadImageNet pre-trained models yields statistically significant, reproducible performance gains over ImageNet models in musculoskeletal diagnostic tasks. In isolated comparative studies focusing on knee MRI, RadImageNet initialization yielded absolute AUC improvements of 4.8% for anterior cruciate ligament (ACL) tears and 4.5% for meniscal tears when compared to identical architectures initialized with ImageNet weights. Beyond raw accuracy, models initialized with RadImageNet produce more compact, well-separated feature embeddings, which accelerates convergence on downstream tasks and substantially improves model calibration—a crucial factor when optimizing against probabilistic soft labels.

### Architectural Backbone Selection: The Resurgence of Modern CNNs

While the pre-training dataset dictates the quality of the initialized weights, the structural architecture of the backbone determines the network's capacity to process high-resolution volumetric data efficiently. The seminal MRNet architecture utilized an AlexNet backbone, which is now entirely obsolete. Subsequent iterations, such as ELNet, leveraged residual connections and blur pooling to improve performance, but modern computational demands require highly optimized topological designs.

Recent outcomes from the RSNA Lumbar Spine Degenerative Classification Kaggle competitions highlight the absolute dominance of modernized CNN architectures over Vision Transformers in data-constrained 3D medical imaging tasks. The recommended base architectures for this specific workload are ConvNeXt and EfficientNetV2.

ConvNeXt bridges the gap between traditional CNNs and Vision Transformers by retrofitting a standard ResNet architecture with ViT-inspired design principles, including larger convolutional kernel sizes, inverted bottlenecks, and layer normalization. This allows ConvNeXt to achieve the expansive receptive fields typical of transformers while maintaining the computational efficiency and inductive spatial biases of convolutions. EfficientNetV2 optimizes parameter count and training speed via progressive learning algorithms and fused mobile inverted bottleneck convolutions, making it highly suited for memory-constrained environments where multiple high-resolution slices must be loaded into VRAM simultaneously for sequence aggregation.

| Architectural Backbone | Parameter Efficiency | Spatial Inductive Bias | VRAM Footprint for 2.5D Input | Suitability for RadImageNet Transfer |
| --- | --- | --- | --- | --- |
| **ResNet-50** | Moderate | High | Moderate | Moderate |
| **ViT-Base (DINOv2)** | Low | Low (Requires large data) | High | Low (Natural Image Bias) |
| **EfficientNetV2-S** | High | High | Low | High |
| **ConvNeXt-Tiny** | Very High | High | Low | Very High |

By deploying a ConvNeXt-Tiny or EfficientNetV2-Small backbone initialized with RadImageNet weights, the pipeline guarantees the extraction of domain-aligned, texturally relevant medical features while operating within a highly optimized memory footprint.

## Intra-Series Aggregation: The 2.5D and Multiple Instance Learning Paradigm

Once the 2D backbone extracts feature maps from individual slices, the pipeline faces its next major architectural hurdle: aggregating an arbitrary number of 2D slice embeddings into a single, fixed-length representation for the entire MRI series. This mechanism must be entirely agnostic to sequence length, functioning seamlessly whether a series contains 15 slices or 350 slices, while strictly preserving the signal of highly localized pathologies.

### 2.5D Spatial Context Integration

Processing single 2D slices in absolute isolation discards the crucial through-plane spatial continuity inherent to human anatomy. A meniscal tear or a complex ligamentous rupture is rarely confined to a single millimeter-thick slice; it propagates across the z-axis. To capture this contiguous structure without incurring the exponential computational cost of full 3D convolutions, the architecture should employ a 2.5D slice-fusion approach.

In a 2.5D paradigm, the network does not ingest a single slice. Instead, the sequence is reorganized into overlapping triplets of adjacent slices (e.g., slice $i-1$, slice $i$, slice $i+1$). These three consecutive grayscale slices are stacked into the RGB channels of the input tensor. The ConvNeXt backbone inherently convolves across these channels, forcing the network to learn short-range spatial dependencies and structural continuity before generating the slice-level embedding. This approach dominated recent RSNA medical imaging competitions, proving its superiority in capturing localized degenerative spine and joint conditions.

### Feature Pyramid Networks for Micro-Lesion Preservation

Knee injuries are notoriously localized. While a massive joint effusion may span the entire knee capsule, a subtle grade-I meniscal tear may only occupy a few pixels within a specific spatial quadrant. Standard CNNs aggressively downsample spatial dimensions in their deeper layers, often obliterating these micro-features before they can be evaluated.

To rectify this, the 2D backbone must be augmented with a Feature Pyramid Network (FPN) and Pyramidal Detail Pooling (PDP), a topological enhancement demonstrated effectively in the MRPyrNet architecture. An FPN extracts multi-scale feature maps from disparate stages of the CNN backbone, dynamically upsampling deep, high-level semantic features and fusing them with shallow, high-resolution spatial features via lateral connections. This ensures that the final 2D slice embedding retains both the abstract diagnostic knowledge required for classification and the low-level geometric detail necessary to detect sub-millimeter tissue disruptions.

### Gated Attention-Based Multiple Instance Learning (AB-MIL)

The aggregation of an arbitrary sequence of slices into a single series representation is classically formulated as a Multiple Instance Learning (MIL) problem. In this framework, the MRI series is defined as a "bag," and the individual 2.5D slice triplets are the "instances." The core assumption of MIL is that the bag (the series) is deemed abnormal if at least one instance (slice) exhibits pathological features.

Early knee MRI architectures, including the original MRNet, resolved this by applying Global Average Pooling (GAP) or Global Max Pooling (GMP) across the temporal/slice dimension. These operators are computationally trivial but diagnostically catastrophic. GAP acts as a mathematical low-pass filter; it averages out the highly sparse pathological signals, overwhelming them with the features of hundreds of healthy slices. Conversely, GMP acts as a high-pass filter, capturing only the single maximum activation across the entire sequence, rendering the network highly susceptible to transient noise, motion artifacts, and non-pathological hyperintensities.

The state-of-the-art resolution to this bottleneck is Gated Attention-Based Multiple Instance Learning (Gated AB-MIL). Pioneered by Ilse et al., this mechanism allows the neural network to dynamically compute a learned attention weight for every individual slice, effectively scoring the slice's diagnostic relevance to the clinical task at hand. The final representation of the series is therefore not a blind average, but a highly targeted, weighted sum of the slice embeddings.

Mathematically, for an MRI series containing $K$ slice embeddings $H = \{h_1, h_2, \dots, h_K\}$, the specific attention weight $a_k$ for slice embedding $h_k$ is computed using a dual-gated attention mechanism:

$$a_k = \frac{\exp\left( \mathbf{w}^T (\tanh(\mathbf{V} h_k^T) \odot \text{sigm}(\mathbf{U} h_k^T)) \right)}{\sum_{j=1}^{K} \exp\left( \mathbf{w}^T (\tanh(\mathbf{V} h_j^T) \odot \text{sigm}(\mathbf{U} h_j^T)) \right)}$$

In this formulation, $\mathbf{V}$ and $\mathbf{U}$ represent learnable weight matrices, $\mathbf{w}$ is a learnable vector, $\odot$ denotes element-wise multiplication, $\tanh$ provides standard non-linear feature transformation, and $\text{sigm}$ (the sigmoid function) acts as a specialized gating mechanism to regulate and stabilize the flow of information across the sequence. The final, aggregated series embedding $\mathbf{z}$ is then derived via context-pooling:

$$\mathbf{z} = \sum_{k=1}^{K} a_k h_k$$

Implementing Gated AB-MIL provides two massive, non-negotiable advantages for this specific workload:

1. **Absolute Robustness to Long-Tail Sequences:** The network can safely ignore hundreds of healthy slices by driving their attention weights $a_k$ to near-zero, preserving the absolute fidelity of the pathological signal regardless of how many extraneous slices are included in the acquisition sequence.
2. **Intrinsic Clinical Interpretability:** The generated attention weights $a_k$ provide ante-hoc, structurally faithful spatial attribution. By visualizing the attention weights across the sequence timeline, radiologists can instantly identify precisely which anatomical slices the model deemed pathological. This intrinsic localization vastly outperforms post-hoc gradient approximations like Grad-CAM, which are notoriously unstable in deep volumetric networks, as demonstrated by the HexMIL architecture in 3D medical forensics.



To further elevate feature expressivity and stabilize the attention manifold, the pipeline can be extended to employ Triple-Kernel Gated Attention. Rather than generating a single attention pathway, the slice embeddings are processed through three distinct mathematical kernel functions—such as the Laplace, Radial Basis Function (RBF), and Inverse Multiquadric (IM) kernels—prior to concatenation. This algorithmic diversification of the feature space significantly stabilizes the attention maps, preventing the network from fixating on isolated artifacts and increasing overall AUC performance by capturing highly complex, non-linear relationships within the joint anatomy.

## Multi-Series Fusion and Dynamic Metadata Modulation

Upon collapsing the variable-length slices into fixed-length, dense series embeddings, the architecture faces the challenge of multi-modal integration. A single patient study comprises 3 to 15 distinct series, spanning different anatomical planes and complex tissue contrasts. A definitive diagnosis of the 12 knee abnormalities often requires the synthesis of evidence across these disparate acquisitions.

### Sequence Routing via Embedding Conditioning

The explicit availability of clean labels describing the sequence types (e.g., "Sagittal Plane," "Fluid-Sensitive," "Fat-Suppression") provides a highly structured prior that must be injected into the architecture. Forcing a neural network to implicitly deduce the physical properties of a scan from its latent features wastes immense computational capacity and introduces unnecessary error.

Instead, each sequence type descriptor is mapped to a learnable, dense embedding vector, functioning identically to positional encodings or sequence-type embeddings in natural language processing. By adding or concatenating these sequence embeddings directly to the AB-MIL generated visual embedding $\mathbf{z}$, the network inherently understands the biophysical context of the features it is evaluating. For example, a hyperintense (bright) region is interpreted drastically differently depending on whether the network knows it is analyzing a T1-weighted anatomical sequence versus a fluid-sensitive, fat-suppressed sequence where brightness explicitly indicates edema or pathological fluid accumulation.

### Transformer-Based Global Cross-Attention

To aggregate the variable 3 to 15 sequence-conditioned series embeddings into a single, holistic patient-level representation, the recommended mechanism is a Label-Sequence Fusion Transformer.

Transformers are uniquely designed to model unordered, variable-length sets by computing dense cross-attention between all elements. In this paradigm, the individual series embeddings are treated as input "tokens" and fed into a standard multi-head Transformer Encoder. To distill this multi-series interaction into a single predictive output, a learned `[CLS]` (classification) token is appended to the input sequence, a technique universally utilized in vision foundation models.

Through self-attention, the Transformer identifies complex inter-series correlations that mirror the workflow of a human radiologist. For instance, the Transformer might learn that a subtle subarticular narrowing detected in the axial plane strongly corroborates a suspected meniscal tear observed in the sagittal plane, synthesizing these weak individual signals into a highly confident joint prediction. The final latent state of the `[CLS]` token serves as the definitive, multi-modal representation of the entire knee MRI study, containing mathematically aggregated evidence from all available planes and contrast mechanics.

### Integrating Subject and DICOM Metadata via FiLM

The dataset also provides rich auxiliary information, specifically subject demographics (e.g., gender) and DICOM metadata (e.g., scanner manufacturer, magnetic field strength, slice thickness). Patient demographics fundamentally alter Bayesian disease priors; for example, baseline cartilage thickness varies significantly across demographics, and the epidemiological prevalence of specific ligamentous tears exhibits distinct demographic skew. Similarly, DICOM metadata dictates the physical fidelity and noise floor of the image acquisition.

Simply concatenating this metadata to the final classification layer—a common fallback—is a highly sub-optimal strategy. By the final classification layer, the neural network has already formed rigid visual representations, leaving the metadata with virtually no capacity to alter the fundamental feature extraction process.

The optimal approach is to dynamically inject this metadata directly into the intermediate layers of the 2D CNN backbone using Feature-wise Linear Modulation (FiLM). FiLM modulates the activations of a neural network through highly efficient affine transformations conditioned entirely on external data.

The metadata variables are concatenated and passed through a small, auxiliary Multi-Layer Perceptron (MLP) to generate two scaling parameters, $\gamma$ and $\beta$, for each feature map channel in the network. The standard feature map activations $\mathbf{x}$ are then modulated as follows:

$$\text{FiLM}(\mathbf{x}) = \gamma \odot \mathbf{x} + \beta$$

This elegant mechanism allows the patient's gender or the scanner's field strength to globally scale, shift, amplify, or mute visual feature representations deeply within the network. If a particular scanner model is known to produce inherently noisier images, the FiLM layer can dynamically dampen high-frequency feature maps. If demographic data indicates a substantially higher prior for a specific degenerative disease, FiLM can pre-emptively amplify the feature channels responsible for detecting osteophytic activity, fundamentally altering the network's visual processing based on clinical context.

## Navigating Label Topology: Multi-Label Optimization and Asymmetric Loss

The final architectural component addresses the most statistically volatile aspect of the workload: predicting 12 distinct abnormality classes characterized by extreme sparsity and continuous, probabilistic soft labels.

### The Failure of Symmetrical Loss Functions in Sparse Topologies

Detecting 12 different abnormalities simultaneously per knee constitutes a severe multi-label classification problem. Given that any specific patient will likely exhibit only one or two concurrent pathologies, the ground truth vectors are massively dominated by negative labels.

In standard multi-label neural networks employing independent Binary Cross-Entropy (BCE) classification heads, this negative-positive imbalance dictates the entire optimization landscape. During backpropagation, the gradients generated by the multitude of easy negative examples (e.g., a pristine, clearly intact posterior cruciate ligament) mathematically overwhelm and drown out the critical, minute gradients originating from the rare positive examples (e.g., a subtle, grade-I meniscal tear). The network quickly learns that aggressively predicting "negative" for all classes yields a highly efficient reduction in global loss, leading to a model that is heavily biased toward false negatives.

While Focal Loss is frequently deployed in object detection to mitigate class imbalance by dynamically down-weighting "easy" examples, it is fundamentally sub-optimal for multi-label classification. Focal loss treats positive and negative samples symmetrically; it assumes that an easy positive sample should be down-weighted just as aggressively as an easy negative sample. In sparse multi-label medical contexts, this is a dangerous assumption. We do not want to heavily down-weight positive samples—every positive signal is rare, precious, and critical for shaping the decision boundary.

### The Superiority of Asymmetric Loss (ASL)

The state-of-the-art algorithmic solution for optimizing multi-label, long-tailed medical imaging datasets is Asymmetric Loss (ASL). ASL is explicitly engineered to address the asymmetrical nature of multi-label topologies by completely decoupling the modulation factors of positive and negative samples. This decoupling allows the network to aggressively discard gradients from overwhelming easy negatives while strictly preserving and amplifying all positive gradients.

ASL introduces two distinct focusing parameters: $\gamma_+$ for positive samples and $\gamma_-$ for negative samples. By explicitly setting $\gamma_- > \gamma_+$ (for example, $\gamma_+ = 0, \gamma_- = 2$), the network fundamentally treats the presence of disease differently than the absence of disease. Furthermore, ASL implements a mechanism known as asymmetric probability shifting. It introduces a margin $m$ that hard-thresholds extremely easy negative samples, effectively shifting their gradient contribution to absolute zero if the network confidently predicts their probability below the margin $m$.

The mathematical formulation for ASL on a single binary label prediction $p$ is derived as follows:

$$L_+ = (1 - p)^{\gamma_+} \log(p)$$

$$L_- = (p_m)^{\gamma_-} \log(1 - p_m), \quad \text{where} \quad p_m = \max(p - m, 0) \\ \text{ASL} = - y L_+ - (1 - y) L_-$$

Recent advancements have further refined this through Taylor series expansions, creating "robust asymmetric loss" functions that ensure highly stable gradients are passed to deep neural networks even when encountering highly ambiguous hard-negative samples. By applying ASL to the 12-class Transformer classification head, the architecture ensures that the pipeline dynamically focuses on confusing background features and borderline cases while reliably retaining the vital signals of rare pathological abnormalities.

## Meta-Learning Strategy: Soft Labels and the Golden Set

The primary training corpus comprises ~4,000 studies annotated with soft labels derived from a high-performing doctor-model ensemble (Macro AUC 0.92). Soft labels are continuous floating-point values $y \in [0, 1]$ rather than discrete hard integers $y \in \{0, 1\}$.

### Knowledge Distillation and the Power of "Dark Knowledge"

Training a neural network on these continuous soft labels constitutes an advanced form of Knowledge Distillation, specifically akin to Low-Rank Clinical Knowledge Distillation (LoRCKD). Soft labels contain exponentially more informational bandwidth per sample than standard hard labels because they capture the "dark knowledge" of the diagnostic process—specifically inter-class similarities and inherent diagnostic uncertainty.

For instance, a hard label simply states $y = 1$ for an ACL tear. A soft label might provide a target of 0.65. This 0.65 informs the student network that the image features are highly ambiguous, preventing the model from forming overly rigid, uncalibrated decision boundaries. Training directly on these ensemble probabilities acts identically to Online Label Smoothing, consistently improving generalization accuracy, generating remarkably well-separated feature embeddings, and preventing pathological overconfidence—a critical flaw in models trained purely on hard binary targets.

To optimize the network against these soft targets, the chosen loss function (whether a soft-target adaptation of ASL or BCE with logits) directly utilizes the continuous probability as the target distribution. The cross-entropy formulation over 12 classes becomes:

$$L_{\text{soft}} = - \sum_{i=1}^{12} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

Where $y_i$ is the ensemble's continuous probability for abnormality $i$, and $\hat{p}_i$ is the model's generated prediction.

### Strategic Utilization of the Golden Dataset

The dataset provides exactly 58 "golden" examples with perfectly clean, human-verified binary labels. While 58 samples are mathematically insignificant for primary spatial feature extraction, they serve as the ultimate arbiters for pipeline validation, threshold tuning, and final decision-boundary calibration.

The optimal utilization of these 58 golden examples involves a precise, three-phase meta-learning approach:

1. **Primary Soft-Label Distillation:** Train the entire hierarchical network—the 2.5D ConvNeXt backbone, the AB-MIL sequence aggregator, and the Transformer fusion head—from scratch (utilizing RadImageNet priors) solely on the 4,000 soft-label studies. This establishes highly robust visual representations, aligns the feature space, and calibrates the network's probabilistic outputs to match the teacher ensemble.
2. **Threshold Calibration:** Deep neural networks trained on soft labels or optimized via Asymmetric Loss output highly calibrated probabilities, but they do not inherently know the optimal binary decision threshold for clinical deployment. By default, most systems assume $p > 0.5$, which is nearly always incorrect for imbalanced medical data. The 58 golden examples must be evaluated post-training to plot precise ROC and Precision-Recall curves. From these curves, the exact probability threshold (e.g., $p > 0.38$) that maximizes the desired clinical metric (such as Macro F1-score or sensitivity) is algorithmically identified for each of the 12 abnormality classes independently.
3. **Low-Rank Adaptation (LoRA) Fine-Tuning (Optional):** In the final epochs of model development, the heavy 2D CNN backbones should be completely frozen. The lightweight Transformer fusion head and the final classification layers can then undergo a very brief, heavily regularized fine-tuning phase on the 58 golden examples using standard hard-label Asymmetric Loss. This definitive fine-tuning acts to firmly anchor the model's final multi-modal decision boundaries to strictly verified clinical truth, correcting any residual biases inherited from the dirty soft labels.



## Synthesis of the Recommended Architectural Pipeline

Based on an exhaustive analysis of the provided clinical workload, volumetric constraints, and contemporary advancements in multi-modal medical imaging, the most robust architecture to test and operationalize is a **FiLM-Modulated, Triple-Kernel Gated-Attention Hierarchical Transformer pipeline, optimized via Asymmetric Loss distillation**.

The definitive end-to-end data flow operates through the following mechanisms:

| Pipeline Stage | Architectural Mechanism | Rationale based on Workload Characteristics |
| --- | --- | --- |
| **1. Volumetric Data Structuring** | 2.5D Slice Triplets (3-channel input) | Captures crucial spatial continuity between adjacent MRI slices, replicating 3D structural integrity without the extreme VRAM costs of 3D CNNs. |
| **2. Base Feature Extraction** | ConvNeXt-Tiny pre-trained on RadImageNet | RadImageNet priors overcome the immense domain gap between natural photography and medical MRI. ConvNeXt ensures high parameter efficiency for batch-processing entire sequences. |
| **3. Micro-Lesion Preservation** | Feature Pyramid Network (FPN) | Dynamically upsamples and preserves fine-grained, high-resolution spatial features essential for identifying micro-lesions like grade-I meniscal tears before they are lost to deep pooling. |
| **4. Metadata Modulation** | Feature-wise Linear Modulation (FiLM) | Dynamically scales convolutional feature maps based on patient gender and DICOM metadata (e.g., field strength), fundamentally altering the visual processing pathway based on clinical context. |
| **5. Sequence Aggregation** | Triple-Kernel Gated Attention MIL | Dynamically identifies and weights pathological slices within sequences up to hundreds of slices long, entirely discarding healthy background data while providing intrinsic 3D spatial attribution for radiologists. |
| **6. Multi-Series Fusion** | Transformer Encoder with Sequence Embeddings | Integrates the 3 to 15 distinct series types (coronal, axial, fat-suppressed) using multi-head self-attention, utilizing clean sequence labels as dense positional/type embeddings to contextually route the visual features. |
| **7. Multi-Label Optimization** | Asymmetric Loss (ASL) on Soft Targets | Eradicates the severe positive-negative class imbalance across the 12 sparse abnormality labels; learns rich diagnostic uncertainty directly from the 4,000 dirty ensemble targets via knowledge distillation. |