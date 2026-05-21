# Artifact Segmentation in Image Super-Resolution

This project focuses on identifying and segmenting localized structural and texturing artifacts introduced during generative image restoration workflows (such as GANs or Transformer-based models like SwinIR). While advanced networks hallucinate realistic high-frequency patterns, they frequently introduce visual distortions on out-of-distribution test details. 

This repository implements classical upscaling baselines, benchmarks a modern transformer pipeline, and introduces a custom U-Net segmentation model to isolate these generative errors.

## Performance Metrics
Evaluated on a validation holdout under a localized artifact segmentation setting, the model achieved:
* **Average IoU:** `0.5629`
* **Total F1-Score:** `0.9366`

---

## Architecture & Engineering Strategy

Instead of feeding raw RGB inputs blindly into a backbone, the model uses an engineered feature stack optimized for artifact detection.

### 1. Multi-Source 10-Channel Tensor Configuration
The encoder accepts a structural feature stack concatenated along the channel dimension. 

**Input Tensor Channels:**
* `Img` (SR): 3 channels (RGB reconstructed images)
* `GT` (Ground Truth): 3 channels (RGB target reference images)
* `Diff` (L1 Error): 3 channels ($|\text{Img} - \text{GT}|$ absolute difference map)
* `Map` (LDL): 1 channel (Local variance maps computed via integral sliding windows)

**Total Shape:** `[Batch_Size, 10, Height, Width]`

* **Why this works:** Passing the L1 difference explicitly saves the encoder from spending capacity re-learning basic pixel-alignment errors. The Local Discriminative Learning (LDL) channel isolates structured, complex anomalies from uniform image noise.

### 2. Skip-Connection Decoder (U-Net Variant)
The network uses a symmetrical deep encoder-decoder layout with bottleneck skip connections (`torch.cat`). This structure ensures high-resolution spatial details are preserved, enabling precise mask boundary definitions even for thin or highly localized texture errors.

---

## Loss Function Experiments

Detecting artifacts is a severe **pixel-class imbalance challenge**, as artifacts only claim minor, localized regions within a high-resolution frame. Standard Binary Cross-Entropy (BCE) fails because background pixels completely dominate the gradients.

During development, I experimented with multiple loss combinations:

* **Focal Loss Only:** Focused the gradients on hard-to-classify boundary pixels by down-weighting easy background regions ($\gamma=2$). However, it suffered from sparse, fragmented segmentation maps.
* **Dice Loss Integration:** Optimized area overlap well and reduced fragmentation, but introduced training instability whenever a batch contained an absolute zero-artifact image.
* **Final Choice (Jaccard Index Loss + Focal Loss):** Peak performance was achieved by combining smooth Jaccard Loss (acting as a direct differentiable proxy for the target IoU metric) with a Focal Loss component. 

$$\text{Loss}_{\text{Total}} = \text{Loss}_{\text{Jaccard}} + \text{Loss}_{\text{Focal}}$$

This combination stabilized convergence by protecting area-level overlap metrics while enforcing strict pixel classification boundaries.

---

## Failed Ideas & Development Notes

To share a realistic view of the project's constraints and things that didn't pan out during training:

* **VRAM Limitations:** Training a heavy multi-modal or multi-stage network wasn't feasible due to video memory limits on the training setup. To get around this, computational weight was shifted to pre-computing the L1 and LDL variance maps on the data side, which dramatically reduced the parameter overhead needed in the model encoder.
* **Morphological Post-Processing:** I attempted to run morphological opening and closing filters (`cv2.MORPH_OPEN` / `cv2.MORPH_CLOSE`) on the predicted binary masks to clean up random outlier pixels. Interestingly, this did not yield a meaningful metric boost and was excluded from the final inference pipeline, suggesting the network's raw boundary predictions were already clean enough.

---

## Hyperparameters & Training Schedule
* **Optimizer:** `AdamW` (Weight decay: `1e-5`)
* **LR Scheduler:** `CosineAnnealingLR` decaying from `1e-4` down to `1e-6` across 50 epochs.
