# Adversarial Optimization and Metric Perception Vulnerability

This project investigates the vulnerabilities of standard image quality and semantic similarity metrics (such as L1/L2 distances, continuous structural metrics, and deep feature extractors). By using deep generative constraints—specifically Variational Autoencoders (VAEs) and Deep Image Priors (DIP)—the scripts generate adversarial images that completely degrade target evaluation metric scores while preserving baseline visual quality under human observation.

Rather than applying unstructured pixel noise (such as basic gradient sign methods), this framework optimizes attacks through latent distributions and structural neural network weights to yield highly coherent adversarial distortions.

## Core Implementations and Modules

The framework is split into separate execution scripts separating metrics, feature extractors, and distinct optimization strategies:

### 1. Vectorized Metric Framework (`adversarial.py`)
* Implements fully vectorized structural evaluation metrics (such as specialized localized Structural Similarity) natively using PyTorch tensor operations.
* Avoids iterative sliding window loops by leveraging GPU-bound convolutions to process mean, variance, and cross-covariance channels simultaneously, providing highly efficient gradient paths during backpropagation.

### 2. Feature Extraction Target Engine (`feature_extractors.py`)
* Wraps deep high-capacity convolutional neural networks to register intermediate activation maps.
* Allows the adversarial optimization loops to target hidden layer representations, effectively attacking "perceptual" feature-space metrics instead of superficial pixel configurations.

### 3. Variational Autoencoder Latent Space Attacks (`vae_attack.py`)
* Optimizes adversarial examples by projecting images into a constrained latent space distribution.
* **The Optimization Objective:** Minimizes structural target metrics while simultaneously applying Kullback-Leibler (KL) divergence regularization against a prior standard Gaussian distribution. 
* **The Advantage:** Forcing the output through a trained VAE decoder acts as an architectural filter, forcing the adversarial anomalies to appear as realistic, organic texture deformations rather than artificial high-frequency static noise.

### 4. Deep Image Prior Adversarial Generation (`dip_attack.py`)
* Reverses the standard Deep Image Prior setup. Instead of optimizing an un-trained network to denoise or reconstruct a clean source, the pipeline optimizes the network weights to generate structured adversarial details.
* Exploits the inductive bias of convolutional architectures to synthesize complex, multi-scale geometric patterns that break downstream metrics while maintaining clean image semantics.

### 5. Consolidated Attack Implementation (`your_attack.py`)
* Integrates latent space constraints, feature extraction targets, and custom loss schedulers into a unified algorithmic attack framework.
* Manages multi-objective optimization weight balances, tracking metrics in real-time across iterative gradient steps.

---

## Evaluation Results

Optimized under a multi-objective configuration balancing attack payload injection against visual fidelity preservation, the unified pipeline achieved the following benchmark scores:

* **Adversarial Target Performance:** 0.4731
* **Perceptual Quality Preservation:** 0.7641
* **Consolidated Optimization Score:** 0.8998


## Technical Insights and Trade-offs

* **Gradient Clipping and Scaling:** Attacking complex fractional metrics (like localized denominator variants) introduces severe gradient explosion or saturation risks. The optimization scripts implement strict bounds on gradient magnitudes to prevent numerical decay during backpropagation.
* **The Perceptual Bottleneck:** The core challenge of advanced adversarial vision engineering is balancing the destruction of automated evaluation scores while preventing obvious visual corruption. Using VAE decoders and DIP architectures functions as a powerful regularizer, ensuring changes blend seamlessly into edge transitions and textured areas.

---

## Execution Environment
* **Core Libraries:** PyTorch, NumPy, SciPy, OpenCV
* **Hardware Profile:** Fully configured to leverage hardware accelerators via unified memory allocation setups during multi-loss backpropagation updates.
