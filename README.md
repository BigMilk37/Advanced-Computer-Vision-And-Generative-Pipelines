# Advanced Computer Vision & Generative Pipelines

A collection of three independent deep learning projects exploring the intersection of **generative image models**, **human visual attention**, and **adversarial robustness**. Each module tackles a distinct research problem in modern computer vision, with custom loss functions, engineered feature stacks, and rigorous metric evaluation.

---

## Repository Structure

```
├── 01-Image-Restoration-And-Artifact-Segmentation/
├── 02-Saliency-Detection-Mechanisms/
└── 03-Adversarial-Latent-Attacks-VAE/
```

---

## Projects

### 01 · Image Restoration & Artifact Segmentation

> Detecting and segmenting hallucinated artifacts introduced by generative super-resolution models (GANs / SwinIR).

Modern transformer-based upscalers hallucinate convincing high-frequency textures — but frequently introduce localized visual distortions on out-of-distribution inputs. This project benchmarks classical upscaling baselines against a transformer SR pipeline and trains a custom **U-Net segmentation model** to isolate those generative errors.

**Key design choices:**
- **10-channel input tensor** — concatenates SR image, ground-truth reference, L1 difference map, and an LDL (Local Discriminative Learning) variance channel, shifting computational weight away from the encoder and onto pre-computed feature maps
- **Jaccard + Focal loss** — selected after ablating BCE, Focal-only, and Dice combinations; the hybrid stabilizes convergence under severe pixel-class imbalance (artifacts are sparse by definition)
- **AdamW + CosineAnnealingLR** — decays from `1e-4` → `1e-6` over 50 epochs

**Results:**

| Metric | Score |
|--------|-------|
| Average IoU | 0.5629 |
| F1-Score | 0.9366 |

---

### 02 · Deep Learning Gaze Fixation Density Prediction

> Predicting continuous human visual attention maps from raw image stimuli.

Rather than discrete object detection, this project models **where a human observer is most likely to look** — generating a probabilistic fixation density surface calibrated against real eye-tracking data. All evaluation metrics are implemented natively in PyTorch (no third-party black-box libraries).

**Key design choices:**
- **Encoder-decoder spatial predictor** with a final spatial softmax layer ensuring valid probabilistic output
- **Native KL Divergence loss** regularizes the predicted distribution against ground-truth fixation histograms
- **Center bias mitigation** — isolates genuine semantic attention from the well-known tendency of viewers to fixate near frame centers

**Results:**

| Metric | Score | Std Dev |
|--------|-------|---------|
| Linear Correlation Coefficient (CC) | 0.6807 | ± 0.1231 |
| Normalized Scanpath Saliency (NSS) | 1.9116 | ± 0.5357 |
| Distribution Similarity (SIM) | 0.6037 | ± 0.0785 |

---

### 03 · Adversarial Latent Attacks on Generative Models

> Breaking automated image quality metrics while preserving human-perceived visual fidelity.

Standard adversarial perturbations add visible pixel noise. This project instead routes attacks through **VAE latent spaces** and **Deep Image Prior (DIP)** architectures, forcing distortions to manifest as organic texture deformations that fool metrics while remaining invisible to human observers.

**Modules:**
- `adversarial.py` — vectorized structural metrics (GPU-bound convolutions, no sliding window loops)
- `feature_extractors.py` — hooks into intermediate CNN activations to attack perceptual feature-space representations
- `vae_attack.py` — latent space optimization with KL divergence regularization against a Gaussian prior
- `dip_attack.py` — inverts the Deep Image Prior setup to synthesize structured multi-scale adversarial patterns
- `your_attack.py` — unified multi-objective framework combining all strategies with real-time metric tracking

**Results:**

| Metric | Score |
|--------|-------|
| Adversarial Target Performance | 0.4731 |
| Perceptual Quality Preservation | 0.7641 |
| Consolidated Optimization Score | 0.8998 |

---

## Tech Stack

| Component | Detail |
|-----------|--------|
| Language | Python 3 |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, NumPy, SciPy |
| Experiment Tracking | Custom `tqdm` loops with real-time multi-metric display |
| Notebooks | Jupyter |

---

## Skills Demonstrated

- Custom loss function design (Focal, Jaccard, KL Divergence, NSS)
- Multi-channel feature engineering for CNN encoders
- Generative model evaluation and failure mode analysis
- Adversarial robustness research (latent space & DIP-based attacks)
- End-to-end training pipelines with schedulers and metric tracking
