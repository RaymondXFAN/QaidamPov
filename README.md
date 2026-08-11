# QaidamPov: Multi-Task Deep Learning for Salinization Monitoring in the Qaidam Basin

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

> Official implementation of **"Multi-Task Deep Learning for Salinization Monitoring in the Qaidam Basin: Integrating Sentinel-2 and WorldCover for Bi-temporal Change Detection"**.

This repository contains the code, configuration, and experimental results for a dual-task CNN framework that jointly predicts pixel-wise saline/non-saline classification and four-class bi-temporal change status, benchmarked against the NASA/IBM Prithvi-EO foundation model with LoRA fine-tuning.

---

## 📊 Headline Results

| Model | Static Acc | Static IoU | Change Acc | Change F1-macro |
|-------|-----------|-----------|-----------|----------------|
| **Dual-Task CNN (n=6 runs)** | **0.824 ± 0.015** | **0.803 ± 0.020** | 0.690 ± 0.017 | **0.360 ± 0.025** |
| Prithvi-EO + LoRA (n=3 seeds) | 0.800 ± 0.016 | 0.777 ± 0.022 | **0.709 ± 0.018** | 0.274 ± 0.022 |
| Single-Task CNN baseline (n=3) | 0.827 ± 0.008 | 0.785 ± 0.011 | — | — |

**Key findings**:
- ✅ Dual-task CNN achieves the best static IoU (0.803) and change F1-macro (0.360)
- ✅ Prithvi+LoRA is competitive on accuracy but loses 2.6 pp on IoU and 8.6 pp on change F1-macro
- ✅ The dual-task gain is λ-sensitive: λ = 0.5–0.7 is optimal (Section 4.4 of paper)
- ⚠️ Rare change classes (expansion < 0.7% of pixels) have F1 = 0 — a quantitative baseline, not a usable product

---

## 🚀 Quick Start

```bash
# 1. Clone this repository
git clone https://github.com/yourname/QaidamPov.git
cd QaidamPov

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data (see docs/DATA.md)
#    Place processed .npy files in data/processed_dual/

# 4. Train and evaluate
bash code/run_all.sh
```

**Full deployment guide**: see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
**Reproduction instructions**: see [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
**Data preparation**: see [`docs/DATA.md`](docs/DATA.md)

---

## 📁 Repository Structure

```
QaidamPov/
├── code/                          # Source code
│   ├── 01_setup.py               # Environment check + data symlink
│   ├── 02_train_dual.py          # Dual-task CNN (multi-seed + λ-ablation)
│   ├── 03_train_prithvi.py       # Prithvi-EO + LoRA training
│   ├── 04_eval_all.py            # Unified evaluation + comparison table
│   ├── run_all.sh                # One-click full pipeline
│   └── patch_03.py               # Critical patch for Prithvi multi-temporal pos_embed
│
├── results/                       # Experimental results
│   ├── comparison_table.md       # Final comparison table (paper-ready)
│   ├── eval_all_results.json     # Unified evaluation JSON
│   ├── prithvi_results.json      # Prithvi+LoRA detailed results
│   ├── PAPER_RESULTS.md          # Paper-ready result summary
│   └── log_prithvi.txt           # Prithvi training log
│
├── docs/                          # Documentation
│   ├── DEPLOYMENT.md             # Hardware/software setup + deployment
│   ├── REPRODUCE.md              # Step-by-step reproduction
│   └── DATA.md                   # Data acquisition + preprocessing
│
├── paper/                         # Manuscript
│   ├── paper.md                  # Full paper (Markdown)
│   └── figures/                  # Generated figures
│
├── assets/                        # Static assets (logos, screenshots)
│
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🔧 Hardware & Software Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| **GPU** | NVIDIA RTX 4090 (24 GB) | NVIDIA RTX 3090 (24 GB) |
| **CPU** | 8+ cores | 4 cores |
| **RAM** | 32 GB | 16 GB |
| **Disk** | 50 GB (data + checkpoints) | 20 GB |
| **Python** | 3.10 | 3.8+ |
| **PyTorch** | 2.0+ with CUDA 12.x | 1.13+ |
| **CUDA** | 12.4 | 11.8 |

**Training time on RTX 4090**:
- Dual-task CNN: ~30 minutes (6 runs × 30 epochs)
- Prithvi+LoRA: ~90 minutes (3 seeds × 50 epochs)
- Total: ~2 hours

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2026qaidam,
  title={Multi-Task Deep Learning for Salinization Monitoring in the Qaidam Basin:
         Integrating Sentinel-2 and WorldCover for Bi-temporal Change Detection},
  author={Your Name and Co-authors},
  journal={Remote Sensing of Environment},
  year={2026},
  volume={},
  pages={},
  doi={}
}
```

---

## 📜 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Data**: Sentinel-2 L2A imagery (Copernicus / AWS Open Data); ESA WorldCover 10 m v100/v200
- **Foundation model**: Prithvi-EO-1.0-100M by IBM/NASA (Hugging Face)
- **Compute**: AutoDL cloud GPU platform (RTX 4090)

---

## 📞 Contact

For questions or issues, please open a GitHub issue or contact [your email].
