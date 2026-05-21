# Deep Learning Gaze Fixation Density Prediction

This project models human visual attention by predicting continuous **fixation density maps** from raw visual stimuli. Unlike discrete object detection, saliency prediction mimics cognitive and biological human visual behaviors — generating a probabilistic surface map indicating where an observer is most likely to look.

The repository contains custom evaluation metric implementations, an end-to-end deep learning training pipeline, and automated performance tracking against human ground-truth eye-tracking data.

---

## Performance Evaluation

The framework evaluates the alignment between predicted probability density maps and ground-truth human fixations using three complementary metrics:

| Metric | Score | Std Dev |
|--------|-------|---------|
| Linear Correlation Coefficient (CC) | **0.6807** | ± 0.1231 |
| Normalized Scanpath Saliency (NSS) | **1.9116** | ± 0.5357 |
| Distribution Similarity (SIM) | **0.6037** | ± 0.0785 |

- **CC** — Measures linear structural alignment and spatial consistency between the predicted map and the ground-truth fixation distribution.
- **NSS** — Samples the normalized prediction map at actual human fixation coordinates. A score of 1.91 standard deviations above the mean indicates strong predictive performance at real gaze points.
- **SIM** — Computes histogram intersection between normalized spatial maps to judge distributional overlap.

---

## Architectural & Technical Implementation

The core value of this repository lies in the native implementation of specialized visual attention loss functions and evaluation benchmarks directly inside the training loop.

### 1. Custom Vectorized Saliency Benchmarks

Rather than relying on third-party black-box metrics, all benchmarks are implemented natively in PyTorch:

- **KL Divergence Loss** — Continuous loss function regularizing the network to minimize information loss between the predicted distribution *P* and the target fixation histogram *Q*.
- **NSS (native)** — Implemented by subtracting the prediction mean and dividing by its standard deviation, then using boolean ground-truth masks to efficiently index salient locations without explicit loop overhead.

### 2. Encoder-Decoder Spatial Predictor

The architecture maps raw RGB input to a single-channel spatial likelihood map:

- High-capacity deep feature extractor builds semantic, spatial representation layers
- Upscaling convolutional blocks with batch normalization and nonlinear activations reconstruct high-resolution density maps from abstract features
- Final **spatial softmax** layer ensures output complies with a valid probabilistic distribution surface

---

## Engineering Notes & Trade-offs

**Sigma / Blur Calibration** — SIM is highly sensitive to spatial density blur. Training reflects a tight optimization balance: predictions must not be over-smoothed (degrades NSS) nor over-sharpened (drops SIM).

**Center Bias Mitigation** — Human eye-tracking data inherently contains severe center bias (viewers tend to fixate near the center of a frame regardless of content). The evaluation pipeline isolates genuine semantic attention cues from predictable center-clustering artifacts.

---

## Training Setup

| Component | Detail |
|-----------|--------|
| Optimizer | Adam |
| Metrics Tracking | Custom `tqdm` loop with real-time multi-metric display across train and validation splits |
