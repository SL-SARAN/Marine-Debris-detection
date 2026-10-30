# Physics-Based Procedural Synthesis for Underwater Ghost Nets
**SIH Marine Debris Anomaly Detection Pipeline — Technical Documentation**

---

## 1. Method Overview

> [!IMPORTANT]
> **Method:** Physics-based procedural synthesis (parametric mesh + ray-cast shadow geometry + speckle noise injection). **NOT a learned diffusion model.** The noise-injection strategy is inspired by a finding in *Syn2Real* (arXiv:2410.12953) regarding residual noise improving sim-to-real generalization — the diffusion architecture itself is not used.

Ghost nets (abandoned, lost, or discarded fishing gear — ALDFG) represent one of the most hazardous marine debris categories. In side-scan sonar (SSS) imagery, ghost nets display distinct acoustic characteristics:
1. **Acoustic Highlights:** Semi-transparent, diffuse, porous multi-line reflections from synthetic polymer cords and knotted junctions rather than the solid specular returns of metallic pipes or hulls.
2. **Acoustic Shadows:** Irregular, ragged, non-geometric acoustic shadow boundaries formed by soft drape and billowed geometry over seafloor topography.
3. **Seafloor Interaction:** Conformity to seabed micro-relief, often snagging on rock mounds or settling in sand ripples.

Due to the complete absence of open-access, labeled Side-Scan Sonar ghost net datasets, this pipeline implements **physics-grounded procedural synthesis** to generate training chips with mathematically exact, noise-free YOLO ground-truth annotations.

---

## 2. Limitations & Honest Framing (Judge Q&A Defense)

> **Judge Q:** *"Why did you use synthetic data instead of real sonar images or a trained diffusion/CycleGAN model?"*

**Defense Answer:**
> *"We chose procedural synthesis over a learned generative model (GAN/diffusion) because we lack real ghost-net sonar training examples needed to train either, and lack the compute for diffusion training in hackathon time. Procedural synthesis trades some visual diversity for exact, physically-grounded, noise-free ground-truth bounding boxes.*
>
> *A GAN or diffusion model trained on inadequate or out-of-domain data suffers from severe mode collapse, phantom geometric artifacts, and hallucinated boundaries that decouple bounding boxes from true object edges. In contrast, our procedural pipeline enforces strict acoustic geometry: highlights correspond to physical strands, and acoustic shadows obey the fundamental sensor grazing equation:*
>
> $$H_{\text{target}} = \frac{L_{\text{shadow}} \cdot h_{\text{altitude}}}{R_{\text{slant}}}$$
>
> *This guarantees that our false-positive rejection filters learn physically coherent relationships rather than generative hallucinations."*

---

## 3. Optical Reference Imagery: Structural Reference Only

To parameterize the procedural mesh generator realistically, we utilize an optical reference dataset of real marine ghost net imagery located in `data/optical_ghost_net_reference/`. 

### The Acoustic Physics Mismatch & Strict Separation
> [!WARNING]
> Optical photographs are used **ONLY** as a geometric and structural reference to measure cord ratios, mesh spacing, and drape topologies. They are **DELIBERATELY FORBIDDEN** from being used as direct pixel input to any style-transfer, image-to-image translation, CycleGAN, or diffusion model.

**Reasoning:**
* **Optical light shadows** are cast by ambient surface sunlight or dive torches with conical/spherical light falloff in 3D water volume.
* **Side-scan sonar acoustic shadows** are strictly planar acoustic line-of-sight occlusions cast along cross-track range beams at shallow grazing angles ($\theta < 20^\circ$) relative to towfish altitude.
* Training an image-to-image neural model to "sonar-ify" optical photos produces visually convincing surface textures but **physically erroneous acoustic shadows**. Since our downstream false-positive rejection filter relies on acoustic shadow length to calculate physical target height, corrupting shadow physics with optical artifacts would actively break the detector's discriminative logic.

**What is extracted from optical reference imagery:**
* Mesh/knot spacing ratios relative to net footprint ($d_{\text{knot}} / L_{\text{span}}$).
* Strand thickness ratios relative to net span ($w_{\text{strand}} / L_{\text{span}}$).
* Drape and entanglement modes (loosely pooled on sediment, snagged over rock mounds, or stretched along bottom currents).
* Fragment aspect ratio distributions ($W / H$).

These properties are extracted purely as numerical parameter bounds via `src/extract_structural_params.py` and saved to `data/structural_mesh_parameters.json`.

---

## 4. Pipeline Architecture

```
┌────────────────────────────────────────────────────────┐
│ 1. Structural Parameter Bounds                         │
│    (From optical reference analysis: ratios & ranges)  │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Procedural Mesh Generation                          │
│    • Parametric Bezier spline curves + knot junctions  │
│    • Non-rigid deformers: ocean bottom current shear,  │
│      tearing, folding, and bunching                    │
│    • Transmissivity: porous multi-cord acoustic return │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Ray-Casting Acoustic Shadow Projection              │
│    • Range-dependent grazing angle projection          │
│    • Soft, ragged shadow boundaries matching drape     │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Background Compositing & Residual Noise Injection   │
│    • Authentic seabed crops from Marine_PULSE          │
│    • Coherent Rayleigh/K-distribution speckle noise    │
│      (Inspired by Syn2Real sim-to-real finding)        │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Exact YOLO Annotation Generation                    │
│    • Bounding boxes bounding highlight + shadow        │
│    • Target class: ghost_net                           │
└────────────────────────────────────────────────────────┘
```

---

## 5. Evaluation & Quality Check Framing

Quantitative fidelity checks are conducted via `src/evaluate_fid_kid.py`.

> [!NOTE]
> **Evaluation Metric Disclaimer:**
> In all scripts, logs, and defense presentations, metrics are explicitly labeled:
> *"FID/KID computed against internal synthetic and texture consistency sets — an internal consistency check on synthetic image realism, NOT a validation against real sonar ghost-net data (none available)."*

The evaluation suite reports:
* **FID (Fréchet Inception Distance):** Measures feature distribution distance in deep representation space.
* **KID (Kernel Inception Distance):** Unbiased polynomial kernel metric suitable for smaller evaluation sample sizes.
* **Acoustic Contrast Ratio (ACR):** Verifies highlight-to-background and shadow-to-background radiometric contrast matches real SSS swaths.
