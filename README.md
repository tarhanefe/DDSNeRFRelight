# Large-Scale 3D Scene Relighting using Pre‑Trained Diffusion Models

**COM507 – Optional Research Project in Communication Systems**  
**Author:** Efe Tarhan, MSc Student in Communication Systems (IVRL)  
**Supervisor:** Dr. Dongqing Wang  

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Background](#background)  
   - [Score Distillation Sampling (SDS)](#score-distillation-sampling-sds)  
   - [Delta Denoising Score (DDS)](#delta-denoising-score-dds)  
   - [DreamCatalyst (2025)](#dreamcatalyst-2025)  
3. [Problem Definition](#problem-definition)  
4. [Methodology](#methodology)  
   - [Wavelet-Based Gradient Filtering](#wavelet-based-gradient-filtering)  
5. [Results](#results)  
   - [2D Image Relighting](#2d-image-relighting)  
   - [NeRF Relighting Comparisons](#nerf-relighting-comparisons)  
   - [Multi‑scale Wavelet Filtering](#multi-scale-wavelet-filtering)  
6. [Future Directions](#future-directions)  
7. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Usage](#usage)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Contact](#contact)  

---

## Project Overview

This research presents a novel technique for high‑quality, large‑scale 3D scene relighting by combining pre‑trained 2D diffusion models with a wavelet‑based gradient filtering approach. Our method preserves high‑frequency details—such as reflections on objects—while applying target relighting edits to Neural Radiance Fields (NeRFs).

---

## Background

### Score Distillation Sampling (SDS)
- Introduced in **DreamFusion** (Poole et al., ICLR 2023).
- Uses a frozen Stable Diffusion model; gradients are backpropagated only to the NeRF’s MLP.
- Allows text‑driven 3D generation but is not optimal for fine‑grained scene editing.

### Delta Denoising Score (DDS)
- Proposed by Hertz, Aberman & Cohen‑Or (ICCV 2023).
- Utilizes two identical Stable Diffusion models guided by a **source** and a **target** prompt.
- Computes the difference between their gradient updates, improving edit quality and reducing artifacts.

### DreamCatalyst (2025)
- Kim, Lee et al. (ICLR 2025).
- Builds on DDS by adding an identity‑preservation term:
  - Emphasizes identity at high noise (early timesteps).
  - Prioritizes editability at low noise (later timesteps).

---

## Problem Definition

Existing DDS‑based methods (including DreamCatalyst) tend to introduce edits across all frequency bands, causing loss of fine, high‑frequency details (e.g., reflections).  
**Objective:** Develop a robust relighting technique that preserves high‑frequency features in 3D scenes.

---

## Methodology

### Wavelet-Based Gradient Filtering

We propose decomposing the DDS gradient into low‑ and high‑frequency components via a discrete wavelet transform (DWT). Only the low‑frequency component is backpropagated during the relighting step:

\`\`\`mermaid
flowchart LR
  A[Diffusion Model (Source Prompt)]
  B[Diffusion Model (Target Prompt)]
  C[Compute Gradient Difference (DDS)]
  D[Wavelet DWT → Low‑Freq Component]
  E[Backpropagate to NeRF MLP]
  A --> C
  B --> C
  C --> D --> E
\`\`\`

---

## Results

### 2D Image Relighting

| Method      | Example                            |
|-------------|------------------------------------|
| Original DDS    | ![DDS Output](images/dds-2d.png)       |
| DDS + Wavelet   | ![Wavelet‑Filtered](images/wavelet-2d.png) |

*Figure: Relighting “shiny balls on pavement” under snowy vs. bright‑sky conditions.*

### NeRF Relighting Comparisons

| Variant                    | Result                              |
|----------------------------|-------------------------------------|
| Original NeRF              | ![Original NeRF](images/nerf-orig.png)    |
| DreamCatalyst              | ![DreamCatalyst](images/nerf-dc.png)      |
| DreamCatalyst + Wavelet    | ![DC + Wavelet](images/nerf-dc-wave.png)  |

*Figure: Red‑sphere relighting under Daubechies 8 wavelet filtering.*

### Multi‑scale Wavelet Filtering

Comparison of 1× vs. 2× wavelet passes:

| Passes         | Result                              |
|----------------|-------------------------------------|
| 1 × Wavelet    | ![1x Wavelet](images/1x-wave.png)        |
| 2 × Wavelet    | ![2x Wavelet](images/2x-wave.png)        |

*Figure: Increasing coarseness preserves the overall shape while focusing edits on low frequencies.*

---

## Future Directions

- **Dataset expansion:** Test on additional scenes with diverse reflective materials.  
- **Regularization targets:** Incorporate depth and surface‑normal consistency.  
- **Wavelet variants:** Explore different implementations (e.g., multiwavelets, adaptive thresholding).  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- PyTorch 1.12+  
- NeRF framework of your choice (e.g., [instant-ngp](https://github.com/NVlabs/instant-ngp))  
- `diffusers` & `transformers` (Hugging Face)  
- `pywavelets` for DWT operations  

### Installation

\`\`\`bash
git clone https://github.com/yourusername/3d-relighting-wavelet.git
cd 3d-relighting-wavelet
pip install -r requirements.txt
\`\`\`

### Usage

1. **Prepare your NeRF checkpoint**  
   \`\`\`bash
   python run_nerf.py --config configs/your_scene.json
   \`\`\`
2. **Relight with wavelet filtering**  
   \`\`\`bash
   python relight.py      --nerf_ckpt runs/your_scene/ckpt.pth      --source "a photo of two reflective spheres."      --target "a photo of two reflective red spheres."      --wavelet daubechies8      --passes 2
   \`\`\`
3. **Inspect output**  
   \`\`\`bash
   open output/rendered_spheres.png
   \`\`\`

---

## Contributing

Contributions are welcome! Please:

1. Fork the repo  
2. Create a feature branch (\`git checkout -b feature/YourFeature\`)  
3. Commit your changes (\`git commit -m "Add foo feature"\`)  
4. Open a Pull Request  

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Contact

Efe Tarhan  
- Email: efe.tarhan@university.edu  
- LinkedIn: [linkedin.com/in/efe-tarhan](https://linkedin.com/in/efe-tarhan)
