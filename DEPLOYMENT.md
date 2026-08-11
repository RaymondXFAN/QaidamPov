# Deployment Guide

This document describes how to deploy the QaidamPov project on a fresh machine, including hardware/software prerequisites, installation, and verification steps.

---

## 1. Hardware Requirements

### 1.1 Recommended Configuration

| Component | Specification | Notes |
|-----------|---------------|-------|
| **GPU** | NVIDIA RTX 4090 (24 GB VRAM) | Tested on AutoDL; other Ada/Hopper GPUs work |
| **CPU** | 8+ cores / 16 threads | Data loading is the main bottleneck |
| **RAM** | 32 GB | Minimum 16 GB; data is loaded as memory-mapped NumPy arrays |
| **Disk** | **50 GB** free | 10 GB data + 30 GB checkpoints + 10 GB system |
| **Network** | 100 Mbps+ | For downloading Sentinel-2 data (optional if pre-processed) |

### 1.2 Minimum Configuration

| Component | Specification | Caveats |
|-----------|---------------|---------|
| GPU | NVIDIA RTX 3090 (24 GB) | Prithvi training requires full 16 GB VRAM |
| RAM | 16 GB | May slow down data loading |
| Disk | 20 GB | Cannot keep all checkpoints; evaluate-then-delete |

### 1.3 Operating System

- **Linux (Ubuntu 20.04+)** — recommended (this guide targets Ubuntu)
- **Windows 10/11** with WSL2 — supported, see `docs/WINDOWS.md`
- **macOS** — not supported (PyTorch CUDA dependency)

### 1.4 Cloud GPU Rental (AutoDL example)

The reference deployment uses AutoDL. Approximate costs:

| GPU | $/hour | Total cost (2 hours) |
|-----|--------|----------------------|
| RTX 4090 | $0.6 | $1.2 |
| A100 (40 GB) | $1.5 | $3.0 |
| A100 (80 GB) | $2.0 | $4.0 |

---

## 2. Software Prerequisites

### 2.1 System-Level

```bash
# Ubuntu prerequisites
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip git wget curl

# NVIDIA driver (AutoDL has pre-installed; skip if on cloud)
# Verify with: nvidia-smi
nvidia-smi  # Should show your GPU and driver version
```

### 2.2 CUDA Toolkit

This project was tested with CUDA 12.4. The PyTorch wheel will install the matching CUDA runtime.

```bash
# Check your CUDA driver version (must be >= 12.0 for PyTorch 2.x)
nvidia-smi | grep "CUDA Version"
```

### 2.3 Python Environment

We recommend **conda** for environment isolation, but **venv** or **system Python** works too.

```bash
# Option A: conda (recommended)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

conda create -n qaidam python=3.10 -y
conda activate qaidam

# Option B: venv (built-in)
python3.10 -m venv ~/qaidam-venv
source ~/qaidam-venv/bin/activate
```

---

## 3. Installation

### 3.1 Clone the Repository

```bash
git clone https://github.com/yourname/QaidamPov.git
cd QaidamPov
```

### 3.2 Install Python Dependencies

We provide a `requirements.txt` for reproducibility:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected `requirements.txt`**:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
timm>=0.9.0
matplotlib>=3.7.0
pandas>=2.0.0
```

### 3.3 Verify Installation

```bash
python -c "
import torch
import timm
import sklearn
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'timm: {timm.__version__}')
print(f'sklearn: {sklearn.__version__}')
"
```

**Expected output**:
```
PyTorch: 2.0.0+cu124
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
VRAM: 24.0 GB
timm: 0.9.0
sklearn: 1.3.0
```

If any import fails, return to Section 3.2 and re-install.

---

## 4. Data Preparation

### 4.1 Data Sources

The project uses two freely available remote-sensing products:

| Data | Source | License | Size |
|------|--------|---------|------|
| **Sentinel-2 L2A** | Copernicus Data Space or AWS open data | EU public | ~5 GB (22 tiles) |
| **ESA WorldCover 10 m v100 (2020)** | Zenodo | CC-BY 4.0 | ~2 GB |
| **ESA WorldCover 10 m v200 (2021)** | Zenodo | CC-BY 4.0 | ~2 GB |

### 4.2 Step-by-Step Acquisition

See [`docs/DATA.md`](DATA.md) for the full acquisition pipeline. Summary:

1. **Define study area**: Qaidam Basin bounding box (36°–39°N, 90°–100°E)
2. **Download Sentinel-2 tiles** (22 MGRS tiles, 2020 growing season)
3. **Download WorldCover 2020 + 2021** rasters for the same extent
4. **Run preprocessing** (in `code/01_setup.py` or standalone) to produce:
   - 6-band Sentinel-2 patches (B02/B03/B04/B08/B11/B12)
   - Binary saline/non-saline labels (WorldCover class 60 + 70 → saline)
   - 4-class change labels (0=stable non-saline, 1=stable saline, 2=expansion, 3=shrinkage)

### 4.3 Final Data Layout

After preprocessing, your directory should look like:

```
data/processed_dual/
├── train_x.npy              # (280, 6, 128, 128)  float32
├── train_y_static.npy       # (280, 128, 128)     int64
├── train_y_change.npy       # (280, 128, 128)     int64
├── val_x.npy                # (60, 6, 128, 128)
├── val_y_static.npy
├── val_y_change.npy
├── test_x.npy               # (60, 6, 128, 128)
├── test_y_static.npy
└── test_y_change.npy
```

> **Note**: The reference dataset ships as a **single train/val/test split** of 280/60/60 patches, derived from 22 Sentinel-2 tiles by sliding-window sampling with stride 64. We do not redistribute the raw data in this repository; users must follow [`docs/DATA.md`](DATA.md) to prepare it locally.

### 4.4 Download Prithvi-EO Weights

```bash
# Option A: Hugging Face (recommended)
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('/root/autodl-tmp/prithvi-100M', exist_ok=True)
# Download from IBM-NASA Prithvi-EO-1.0-100M
hf_hub_download(repo_id='ibm-nasa-geospatial/Prithvi-EO-1.0-100M',
                filename='Prithvi_100M.pt',
                local_dir='/root/autodl-tmp/prithvi-100M')
"

# Option B: Direct download (if you have a URL)
wget -O /root/autodl-tmp/prithvi-100M/Prithvi_100M.pt \
  https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_100M.pt
```

---

## 5. Project Layout

After cloning and data preparation:

```
QaidamPov/
├── code/                       # Run scripts from here
│   ├── 01_setup.py
│   ├── 02_train_dual.py
│   ├── 03_train_prithvi.py
│   ├── 04_eval_all.py
│   ├── run_all.sh
│   └── patch_03.py
├── data/
│   └── processed_dual/         # Created by you (see Section 4)
├── checkpoints/                # Created automatically by training scripts
├── results/                    # Created automatically by training scripts
└── logs/                       # Created automatically by training scripts
```

---

## 6. Running the Pipeline

### 6.1 One-Click Full Pipeline

```bash
cd code/
bash run_all.sh
```

This will execute:
1. `01_setup.py` — environment check + data symlink
2. `02_train_dual.py --mode both` — multiple seeds + λ-ablation (~30 min)
3. `03_train_prithvi.py` — Prithvi+LoRA training (~90 min)
4. `04_eval_all.py` — unified evaluation + comparison table

### 6.2 Step-by-Step (Recommended for First Run)

```bash
cd code/

# Step 1: Environment check
python 01_setup.py

# Step 2: Dual-task CNN training
# Modes: 'seed' (3 seeds, λ=1.0), 'ablation' (4 λ values), 'both', 'single'
python 02_train_dual.py --mode both

# Step 3: Prithvi+LoRA training
python 03_train_prithvi.py

# Step 4: Unified evaluation
python 04_eval_all.py
```

### 6.3 Background Execution (Recommended for Long Jobs)

```bash
cd code/

# Train dual-task in background
nohup python -u 02_train_dual.py --mode both > logs/dual_train.log 2>&1 &
DUAL_PID=$!
echo "Dual-task training PID: $DUAL_PID"

# Train Prithvi in background
nohup python -u 03_train_prithvi.py > logs/prithvi_train.log 2>&1 &
PRITHVI_PID=$!
echo "Prithvi training PID: $PRITHVI_PID"

# Monitor progress
echo "--- Dual-task training log (last 20 lines) ---"
tail -20 logs/dual_train.log
echo "--- Prithvi training log (last 20 lines) ---"
tail -20 logs/prithvi_train.log
```

**Monitor command**:
```bash
# Watch both logs in real time
watch -n 5 'echo "=== Dual-task ==="; tail -10 logs/dual_train.log; echo "=== Prithvi ==="; tail -10 logs/prithvi_train.log'
```

### 6.4 Expected Output

After all steps complete, `results/` should contain:

```
results/
├── comparison_table.md       # Final comparison table
├── eval_all_results.json     # Unified evaluation JSON
├── dual_summary.json         # Dual-task CNN aggregated results
├── prithvi_results.json      # Prithvi+LoRA aggregated results
└── prithvi_per_seed.json     # Per-seed Prithvi metrics
```

---

## 7. Verification

### 7.1 Quick Sanity Check

```bash
# Verify all expected checkpoints exist
ls -la checkpoints/
# Should contain:
#   dual_seed42_lam1.0.pth
#   dual_seed123_lam1.0.pth
#   dual_seed2024_lam1.0.pth
#   dual_seed42_lam0.3.pth
#   dual_seed42_lam0.5.pth
#   dual_seed42_lam0.7.pth
#   prithvi_lora_seed42.pth
#   prithvi_lora_seed123.pth
#   prithvi_lora_seed2024.pth

# Verify results JSONs are valid
python -c "
import json
for f in ['eval_all_results.json', 'prithvi_results.json', 'dual_summary.json']:
    with open(f'../results/{f}') as fp:
        data = json.load(fp)
    print(f'✅ {f}: {len(data)} entries')
"
```

### 7.2 Expected Comparison Table

Running `04_eval_all.py` should produce a table matching the headline numbers in the README. Verify:

```bash
cat ../results/comparison_table.md
```

Expected last line:
```
| **DualTask (mean±std)** | 0.824±0.015 | 0.803±0.020 | 0.833±0.010 | 0.690±0.017 | 0.360±0.025 |
| **Prithvi+LoRA (mean±std)** | 0.800±0.016 | 0.777±0.022 | 0.812±0.008 | 0.709±0.018 | 0.274±0.022 |
```

If your numbers deviate by more than ±0.02, see [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## 8. Troubleshooting

### 8.1 Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `CUDA out of memory` | Other GPU users | Reduce `--batch-size` from 4 to 2 |
| `timm not installed` | Missing dependency | `pip install timm` |
| `pos_embed shape mismatch` | Old Prithvi checkpoint format | Run `python patch_03.py` |
| `data/processed_dual not found` | Data not prepared | Follow `docs/DATA.md` |
| `Single-task baseline numbers off` | Different Numpy/sklearn version | Use `requirements.txt` exact versions |

### 8.2 Performance Issues

If training is slow (RTX 4090 should do ~35 sec/epoch for dual-task):

- **Enable cuDNN benchmark**: add `torch.backends.cudnn.benchmark = True` to top of training script
- **Use mixed precision**: add `torch.cuda.amp` wrapper (not included by default for reproducibility)
- **Reduce num_workers**: try `num_workers=0` if seeing I/O bottleneck

### 8.3 Reproducibility

For exact bit-exact reproducibility:

```bash
# All scripts use:
# - python 3.10
# - PyTorch 2.0.0+cu124
# - Deterministic algorithms (cudnn.deterministic=True)
# - Fixed seeds where applicable

# Verify your environment
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
"
```

---

## 9. Cost & Time Estimates

| Phase | Wall time (RTX 4090) | Approximate cost (AutoDL) |
|-------|---------------------|---------------------------|
| Environment setup | 5 min | $0.05 |
| Data download (cold) | 1–2 hours | $0.60–1.20 |
| Data preprocessing | 30 min | $0.30 |
| Dual-task CNN training | 30 min | $0.30 |
| Prithvi+LoRA training | 90 min | $0.90 |
| Evaluation | 5 min | $0.05 |
| **Total** | **~3 hours** | **~$2.20** |

---

## 10. Frequently Asked Questions

**Q: Can I run this on a CPU?**
A: No, Prithvi-EO requires GPU for fine-tuning. The dual-task CNN could theoretically run on CPU at ~100× slower speed.

**Q: Why does the README say "single-date Sentinel-2" when Prithvi is multi-temporal?**
A: Our adaptation extracts the middle frame of Prithvi's pos_embed and uses single-date Sentinel-2 inputs. Section 3.3 of the paper explains this caveat.

**Q: Can I use a different foundation model?**
A: Yes — replace `timm.create_model('vit_small_patch16_224', ...)` in `03_train_prithvi.py` with another backbone. The LoRA layer logic is model-agnostic.

**Q: How do I cite this work?**
A: See the Citation section in README.md.

**Q: Where can I get help?**
A: Open a GitHub issue or email the corresponding author.

---

## 11. License

This project is released under the MIT License. See [`LICENSE`](../LICENSE).
