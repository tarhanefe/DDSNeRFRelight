<h1 align="center">Large-Scale 3D Scene Relighting using Pre‑Trained Diffusion Models</h1>

<p align="center"><strong>COM507 – Optional Research Project in Communication Systems</strong></p>
<p align="center">Efe Tarhan, MSc Student in Communication Systems</p>
<p align="center"><strong>Supervisor:</strong> Dongqing Wang, Image and Visual Representation Lab (IVRL)</p>

---

<p align="center">
  <img src="assets/new_pipeline.png" alt="EPFL Logo" width="%100"/>
</p>

---

## 📝 Project Description

This repository implements a NeRF editing framework that enables localized relighting and texture edits using pretrained diffusion models. Built on the DDS pipeline, it integrates wavelet-based gradient filtering to preserve reflections and fine appearance details during editing. The framework keeps scene geometry fixed after NeRF training, ensuring structural consistency, and enhances edits with surface normal prediction for improved view-dependent rendering. The result is high-fidelity, semantically guided 3D scene edits with strong reflection and color consistency.


---
## 📁 Folder Structure

```
.
├── README.md
├── requirements.txt
├── assets/                          # Diagrams and logos
├── nerfstudio/3d_editing/dc_nerf/  # NeRF editing module
│   ├── data/                       # Custom datamanagers & parsers
│   ├── engine/                     # Trainer
│   ├── fields/                     # Fields used in models
│   ├── models/                     # Nerfacto/Splatfacto variants
│   └── pipelines/                  # Base + DC pipelines
├── dc/                             # Diffusion controller 
│   ├── dc.py, 
|   ├── cds.py                      # Main training logic
│   ├── dc_unet.py                  # FreeU UNet
│   └── utils/                      # Wavelet, image, FreeU utils
```

---

## ⚙️ Installation

First clone the repository 

```bash
git clone https://github.com/tarhanefe/DDSNeRFRelight.git
cd DDSNeRFRelight
```

Create a conda environment

```bash
conda create -n relight python=3.9
conda activate relight
```

Then, install the required packages.

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .
```
For 3D editing, you need to install the following packages.

```bash
cd 3d_editing
pip install -e .
pip install gsplat==0.1.6
pip install huggingface_hub==0.21.0
pip install tyro==0.6.6
pip install torchwavelets
```

In case the following command won't work. Manually register the dc method to the datamanager config in the main environment nerfstudio library. 

```bash 
ns-train -h

```

---

## 📂 Dataset

You can access the dataset used for training and evaluation via the following link:  
🔗 [Google Drive Dataset](https://drive.google.com/drive/folders/1nO8yCE3YvK-gHqXteKokSjv4ujfTLX9c?usp=sharing)

Put the data folders under the ./3d_editing folder.

---

## 🚀 CLI Usage

### 🔧 Standard Command

First train the scene using the default NeRF constructor of the nerfstudio

```bash
ns-train nerfacto --data ./gardenspheres_n nerfstudio-data --downscale-factor 8
```

Following that use the given command to edit the scene. 

```bash
ns-train dc --data ./gardenspheres_n --load-dir ./outputs/gardenspheres_n/nerfacto/2025-03-30_013255/nerfstudio_models/ \
 --pipeline.dc.src_prompt "a photo of two reflective spheres" \
 --pipeline.dc.tgt_prompt "a photo of two reflective green spheres" \
 --vis viewer \
 --max_num_iterations 200 \
 --pipeline.dc.pipeline dc \
 --pipeline.dc.wavelet_filtering True \
 --pipeline.dc.wavelet_name db4 \
 --pipeline.dc.wavelet_level 2 \
 --pipeline.dc.guidance-scale 7.5 nerfstudio-data --downscale-factor 8
```

---


## 🧾 Configuration Parameters

The following parameters in `DCConfig` control the behavior of the diffusion-guided editing pipeline:

| Parameter                  | Description |
|---------------------------|-------------|
| `num_inference_steps`     | Number of diffusion steps used during inference. |
| `pipeline`                | Type of editing pipeline ('cds' or 'dc'). |
| `min_step_ratio` / `max_step_ratio` | Defines the range of denoising steps for guided editing. |
| `src_prompt` / `tgt_prompt` | Text prompts for original and target scene descriptions. |
| `log_step`                | Interval (in iterations) for logging progress. |
| `guidance_scale`         | Strength of text guidance during editing. |
| `device`                  | Device used for computation (typically `"cuda"`). |
| `image_guidance_scale`    | Balances pixel-level and semantic guidance. |
| `psi`, `chi`, `delta`, `gamma` | Hyperparameters controlling loss terms and step size dynamics. (Only for DC) |
| `freeu_b1`, `freeu_b2`     | FreeU enhancement ratios for low-frequency channels. (Only for DC)|
| `freeu_s1`, `freeu_s2`     | FreeU suppression ratios for skip connections. (Only for DC)|
| `wavelet_filtering`       | Enable/disable wavelet-based filtering for gradients. |
| `wavelet_name`            | Name of the wavelet used (e.g., `haar`, `db2`, `sym4`). |
| `wavelet_level`           | Decomposition level of the wavelet transform. |
| `n_patches`               | Number of random image patches used for DDS/CUT loss. (Only for CDS)|
| `patch_size`              | Size of each patch (e.g., `(1, 2)` for long-and-thin patches). (Only for CDS)|
| `w_dds`, `w_cut`          | Weights for DDS and CUT loss components. (Only for CDS)|
| `scheduler_pretrained_path` | Optional path to custom DDIM scheduler. |
| `loss_multiplier`         | Scaling factor for the combined DDS/CUT loss. (Only for CDS)|

These parameters are set inside the editing pipeline and can be overridden via CLI using `--pipeline.dc.<parameter>` flags.
