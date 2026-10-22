# DEPLOY.md — how to actually run this thing

**QaidamPov · leakage-audited salinization benchmark (Sentinel-2 + ESA WorldCover)**
Maintainer: Raymond Fan · fanxiaohu@whcp.edu.cn

> **A note from Raymond.** I am a remote-sensing person, not a professional software
> engineer. Everything below is what I actually typed into a terminal, in the order I
> typed it, including the parts that failed the first three times. If something here
> looks over-explained, it is because I needed it explained to me too. If you hit a
> wall, mail me — I would rather answer than have you guess.

---

## 0. What this repository contains

| Part | What it is | Where |
|---|---|---|
| `code/rebuild/` | Rebuilds the whole dataset from public archives, no login needed | 8 files |
| `code/supp/` | 20 experiment scripts + 6 runners + self-tests | ~40 files |
| `figures/scripts/` | The 6 manuscript figures, Morandi palette, Times serif | 7 files |
| `manuscript/` | Final Word, PDF and Markdown | 3 files |
| `results/` | Filled in after you run things — not shipped (numbers are in the paper) | — |

The paper is the specification. If the code and the paper disagree, the paper is
right and the code is a bug: tell me.

---

## 1. Hardware and software

### 1.1 What I ran it on

| Item | Value |
|---|---|
| GPU | **1 × RTX 4090, 24 GB** (AutoDL instance; a 3090 also fits, just slower) |
| CPU | 16 vCPU (anything ≥ 8 works; most experiments are GPU-bound) |
| RAM | 64 GB (the pixel-level baselines hold 800 k × 9 float32, ~30 MB, so RAM is not the bottleneck) |
| Disk | ~120 GB: 60 GB imagery cache + 10 GB derived arrays + results |
| OS | Ubuntu 22.04 |
| Driver | 570.124.04, CUDA 12.8 runtime reported by `nvidia-smi` |

I never needed more than one card. Total wall-clock for the full matrix in the paper
was roughly **9 days**, of which the five-family ablation with eight seeds is about
6 days. Nothing here is a cluster job.

### 1.2 Python environment

Python **3.10**. Two ways to get there; pick one and do not mix them.

**Option A — conda (what I use)**

```bash
conda create -n qaidam python=3.10 -y
conda activate qaidam
pip install -r requirements.txt
```

**Option B — venv**

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1.3 The PyTorch line, and the one thing that will bite you

Install PyTorch **matching your CUDA runtime**, not the newest one:

```bash
# CUDA 12.x (my case: driver 570 / CUDA 12.8 -> cu124 wheels work, cu121 also works)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> **A note from Raymond.** Do not `pip install torch` from PyPI on a GPU box and
> assume it works. It will quietly give you a CPU build and your first epoch will
> take an hour. Check with
> `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
> and make sure it prints `True`.

Everything else (rasterio, pystac-client, shapely, pyproj) is needed **only** for the
dataset rebuild. If you already have the `.npy` arrays, you can skip that block
entirely and run on a CPU-only machine for the classical baselines.

---

## 2. Where the data comes from

Nothing in this project needs a login, an API key, or a data-request form. That was a
deliberate design choice so that a reviewer can rebuild the benchmark.

### 2.1 Imagery — Sentinel-2 L2A

| Item | Detail |
|---|---|
| Product | Sentinel-2 **Level-2A** surface reflectance, 6 bands: B02 B03 B04 B08 B11 B12 |
| Grid | 10 m analysis grid (the two 20 m SWIR bands are resampled) |
| Access path used | Copernicus Data Space Ecosystem, plus the AWS Earth Search STAC mirror as a fallback: `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a` |
| Asset keys on the AWS mirror | `blue, green, red, nir, swir16, swir22` (= B02, B03, B04, B08, B11, B12) |
| Tiles | **18 MGRS tiles**, UTM zones 45–47, listed in Table A1 of the paper |
| Acquisition window | 25 July – 30 September 2020 (one cloud-free scene per tile) |
| Cloud ceiling | **15 %** scene-level, lowest-cloud scene below the threshold is kept |
| Publisher credit | Contains modified Copernicus Sentinel data (2020), processed by ESA |

> **A note from Raymond.** One scene per tile, deliberately. If you start compositing
> you are no longer reproducing our numbers, you are doing a different paper. The
> reason is boring: single-date reflectance is what the papers we are arguing with
> used, and I wanted the confound to be theirs, not mine.

### 2.2 Labels — ESA WorldCover, used as a proxy

| Item | Detail |
|---|---|
| Product | ESA WorldCover 10 m |
| Versions | **v100 (2020)** for the static label, **v200 (2021)** for the change head |
| Access | Public S3, no credentials: `https://esa-worldcover.s3.eu-central-1.amazonaws.com`, path `/v100/2020/map/` and `/v200/2021/map/` |
| File naming | `ESA_WorldCover_10m_<year>_<version>_<TILE>_Map.tif`, e.g. `ESA_WorldCover_10m_2020_v100_N36E093_Map.tif`, TILE is a 3° cell (e.g. `N36E093` = 36–39 °N, 93–96 °E) |
| Class used | **class 60 only** (bare / sparse vegetation) is treated as saline |
| Licence | CC BY 4.0 |
| Credit | © ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data |

Read this carefully, because it is the whole point of the paper: **class 60 is not
salinity.** It is a bare-ground class. Over the Qaidam floor, bare ground and salt
crust are spectrally close, which is why the proxy works at all, and also why no model
can exceed IoU ≈ 0.93 on it. If you cite our numbers as "salinity mapping accuracy",
you have misread them.

### 2.3 Base model weights — Prithvi-EO-1.0

| Item | Detail |
|---|---|
| Weights | Prithvi-EO-1.0-100M, from IBM/NASA via Hugging Face |
| Licence | Apache-2.0 (check the model card before commercial use) |
| Offline note | Behind the GFW this is the only step that needs help; see §6.4 |

### 2.4 Rebuilding it yourself

```bash
cd code/rebuild
python selftest_rebuild.py            # offline, synthetic GeoTIFFs, no network. Run this FIRST.
bash run_rebuild.sh gate              # 2 real tiles, ~5–10 min. Proves the network path works.
nohup bash run_rebuild.sh all > logs/boot.log 2>&1 &      # full rebuild, 20–40 min
bash run_rebuild.sh status            # non-blocking progress. Do NOT use tail -f.
```

Key flags (defaults reproduce the paper):

| Flag | Default | Meaning |
|---|---|---|
| `--bbox` | `90,36,100,39` | sampling frame, lon/lat |
| `--date-range` | `2020-06-01/2020-09-30` | growing season |
| `--max-cloud` | `15` | scene-level ceiling, per cent |
| `--region-px` | `640` | 6.4 km window sampled inside each tile |
| `--patch / --stride` | `128 / 64` | 50 % overlap by construction |
| `--total` | `400` | 280 / 60 / 60 |
| `--saline-classes` | `60` | use `60,70` to widen the proxy |
| `--split-mode` | `patch` | `group` gives the leakage-free tile split |

Output: `data/processed_dual/{train,val,test}_{x,y_static,y_change}.npy`, with
`x` of shape `(N, 6, 128, 128)`, float32. A `dataset_card.json` next to it records
the tile list, dates and counts — keep that file, it is what Table A1 was built from.

---

## 3. Repository layout

```
QaidamPov/
├── code/
│   ├── rebuild/                 # dataset reconstruction (network)
│   │   ├── rebuild_dataset.py   # STAC search -> COG windows -> WorldCover -> .npy
│   │   ├── rb_common.py         # shared helpers (tiling, resampling, hashing)
│   │   ├── run_rebuild.sh       # gate / all / status
│   │   └── selftest_rebuild.py  # offline self-test
│   ├── supp/
│   │   ├── common.py            # paths, seeds, IoU-from-confusion-matrix, pooling
│   │   ├── exp_*.py             # one file per experiment (see §5)
│   │   ├── run_*.sh             # runners: gate / all / status, idempotent
│   │   └── selftest_*.py        # offline checks, no GPU, no data
│   └── build/
│       └── md2docx_jstars.py    # Markdown -> IEEE-style docx (tables, equations, figures)
├── figures/
│   ├── scripts/
│   │   ├── morandi_style.py     # palette + Times + save() + overlap self-check
│   │   ├── fig1_...py ... fig6_...py
│   └── png/ , pdf/              # 600 dpi raster, vector
└── manuscript/
    └── final/                   # .md source, .docx, .pdf
└── results/                     # created at runtime
```

Design rule I tried to keep: **one script = one table in the paper.** If you want to
know where a number came from, look at the table, then find the matching `exp_*.py`.
No hidden state, no notebooks, no `run_everything_v2_final.py`.

---

## 4. Quick start (the 20-minute version)

```bash
# 1. environment
conda create -n qaidam python=3.10 -y && conda activate qaidam
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. sanity, no GPU needed, no data needed
cd code/supp && python selftest_supp.py         # expect: PASS
cd ../rebuild && python selftest_rebuild.py     # expect: PASS

# 3. get the data (or copy data/processed_dual from wherever you keep it)
bash run_rebuild.sh gate
nohup bash run_rebuild.sh all > logs/boot.log 2>&1 &

# 4. one short experiment, to prove the whole chain works end to end
cd ../supp
bash run_supplementary.sh gate                  # ~5 min
bash run_supplementary.sh status

# 5. only then, the long ones
nohup bash run_supplementary.sh all > logs/boot.log 2>&1 &
```

> **A note from Raymond.** Step 4 is not optional. I once launched a 16-hour batch and
> found out afterwards that a Python exception had been swallowed and 20 jobs had
> "finished" in six seconds with no output files. The gate exists because of that day.
> If `status` says a job is done, check that the file actually exists.

---

## 5. Running the experiments

All scripts take `--data-dir`, `--out-dir` and `--seeds`; defaults point at
`data/processed_dual` and `results/supplementary`. Every runner is idempotent: it
skips a step whose output file already exists, which means **rename or delete old
outputs before a rerun**, or you will think you reran something you did not.

| Script | Produces | Paper artefact |
|---|---|---|
| `exp_j_split_report.py` | split composition, MD5 overlap audit | Table A1, Fig. 2 |
| `exp_k_label_purity.py` | band statistics, Fisher ratios, linear separability | Fig. 1a, §IV-A |
| `exp_f_classical_baselines.py` | RF / XGB, raw and index features | Table I top rows |
| `exp_c_unet_baseline.py` | U-Net single / dual | Table I |
| `exp_a_multiseed_lambda.py` | λ grid ∈ {0.3, 0.5, 0.7, 1.0} | Table VI |
| `exp_d_spatial_split.py` | the leakage-free tile split | Table II |
| `exp_p_common_test.py`, `exp_q_common_eval.py` | the common 66-patch evaluation set | §III-D, Fig. 3 |
| `exp_r_did.py` | arms A / B, difference-in-differences | Table III, Table IV |
| **`exp_u_ablation_all.py`** | **five families × three arms × 8 seeds** | **Table III, Fig. 5** |
| **`exp_v_matched_arm.py`** | **arm C′ (sample-size matched)** | **Fig. 4, §IV-C** |
| `exp_x_table1_8seed.py` | Table I recomputed on all eight seeds | Table I |
| `exp_t_leaveout.py` | placebo arm for the U-Net | Table IV |
| `exp_n_paired_stats.py`, `exp_h_paired_pixel_tests.py` | paired t, sign-flip, pixel-level tests | Table V |
| `exp_e_diagnostic.py` | difficulty-stratified diagnostic | §V-B |
| `exp_i_change_head_loss.py` | change head, three losses | Table VI note |
| `exp_m_clean_aux.py` | cleaned auxiliary-label experiment | Fig. 6 |
| `exp_b_export_raw_metrics.py` | confusion matrices, per-class metrics | Table VI |
| `exp_g_prithvi_rerun.py` | Prithvi-EO-1.0 + LoRA | Table I, Table II |
| `review_patch_tables.py` | regenerates the reviewer-facing tables | supplementary tables |

Longest single job: `exp_u_ablation_all.py` with 8 seeds × 5 families × 3 arms, about
6 days on one 4090. Run it with `nohup`, then poll:

```bash
nohup python -u exp_u_ablation_all.py --seeds 42,123,2024,7,13,99,777,2025 \
      > ../logs/expU.log 2>&1 &
tail -n 40 ../logs/expU.log        # not tail -f; you will never get the prompt back
```

---

## 6. Figures

```bash
cd figures/scripts
python fig1_design_audit.py      # writes _out/Fig1_*.{png,pdf}
python fig2_ladder_did.py
python fig3_separability_rank.py
python fig6_cprime.py
python fig4_aux_indices.py
python fig5_family_gradient.py
```

`morandi_style.py` holds the palette, the Times/Nimbus serif setup, `save()` (PNG at
600 dpi + vector PDF) and two self-checks: `overlap_report()` for text collisions and
`edge_ink_check()` for clipped labels. If you change a number, change it in the script,
not in Illustrator.

Colour semantics are fixed across all six figures: `#4E6B80` deep slate = tile-level
(honest) protocol, `#B0868A` dusty rose = patch-level (leaky) protocol, `#6FA0A0` grey
teal = deduplicated middle arm, `#C39B72` greige = ordinary data loss.

> **A note from Raymond.** Figures in this field are usually where the honesty goes to
> die — a truncated y-axis here, a cherry-picked seed there. Every figure script here
> reads the raw per-seed numbers and prints all of them, so if a bar looks good it is
> because the number is good.

---

## 7. Where the results live, and how to read them

```
results/supplementary/
├── expA_multiseed_lambda/grid_results.json      # one row per (seed, λ)
├── expB_raw_metrics/raw_metrics_all.json        # confusion matrices + per-class metrics
├── expC_unet_grouped/unet_summary.json
├── expD_spatial/data_grouped/                   # the tile-level dataset
├── expU_ablation/{family}/{arm}/                # per-run metrics
├── expU_ablation_all/TABLE_ablation.md          # the five-family table
├── expV_matched/TABLE_matched.md                # arm C′
├── expW_replicate/TABLE_replicate.md            # fresh seeds 1001–1008
└── common_test/{common_index.json, did.json}    # the 66-patch set and the DiD
```

Two habits that saved me time:

1. **Trust files, not exit codes.** A job counts as done when its JSON exists and
   parses, not when the shell says `0`.
2. **`common_index.json` is the ground truth** for which patch is seen / unseen. Do
   not infer it from IoU agreement; that heuristic is wrong and it fooled me for a
   day.

---

## 8. Reproducibility, honestly

The deep pipeline is **not bit-reproducible**. Re-running one configuration at a fixed
seed moves IoU by about **0.2 points**, and one re-execution moved it by **3.9 points**
(seed 777). `torch.backends.cudnn.deterministic = True` does not cover everything,
and one of our ablation scripts overrides it to `benchmark = True` for speed. So:

- read a single run to ±0.2 points at best;
- an eight-seed mean inherits ≈ 0.5 points from that;
- confidence intervals in the tables cover **seed-to-seed** variation only.

This is in §III-B of the paper and in the notes to Tables III–V, because a reader who
compares our numbers to theirs needs to know how much of the difference is noise.

---

## 9. Things that will break

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` on the U-Net | batch 8 + 128² on a 12 GB card | `--batch-size 4`, or `torch.cuda.empty_cache()` between arms |
| `ModuleNotFoundError: exp_u_ablation_all` | you are in the wrong directory | run from `code/supp`, or set `PROJECT_DIR` |
| Prithvi download hangs | Hugging Face blocked | `export HF_ENDPOINT=https://hf-mirror.com` **before** `import transformers`; see §6.4 |
| `libgomp: Invalid value for OMP_...` | joblib inherits a bad thread setting | `export OMP_NUM_THREADS=4` before launching |
| runner says done, no files | idempotence skipped it | delete the target directory, rerun |
| `data/processed_dual` missing | rebuild never ran | `bash run_rebuild.sh gate` |

> **A note from Raymond.** Behind the GFW, the only genuinely painful step is fetching
> the Prithvi weights. Everything else — imagery, WorldCover, all labels — comes down
> without a proxy. Set `HF_ENDPOINT` *before* the import, not after; the environment
> variable is read once, at import time, and setting it later silently does nothing.

---

## 10. Citation and contact

If you use the code or the benchmark, please cite the paper (see `README.md`) and the
two upstream products:

- Copernicus Sentinel-2 L2A, ESA.
- ESA WorldCover 10 m v100/v200, Zanaga et al., doi:10.5281/zenodo.7254221.

Bugs, questions, disagreements: **Raymond Fan — fanxiaohu@whcp.edu.cn**. I answer
e-mail faster than issues, but issues are better for the next person.

---

## 11. AI-use statement

My English is not good enough to write a manuscript in the form you have just read,
and pretending otherwise would waste a reviewer's time. So, plainly:

- The research question, the experimental design, all code, all experiments, all
  numbers, and the interpretation of those numbers are mine. No result in this
  repository was produced or selected by a language model.
- **Drafting, translation between Chinese and English, and language polishing were
  assisted by Kimi K3 and GLM.** They were used the way one uses a copy-editor: on
  text that already existed, in my own argument order, with my own claims.
- Where a model helped, I checked the sentence against the number it refers to. Any
  remaining error is mine, not the model's, and I would like to hear about it.

I state this because a reader is entitled to know which parts of a paper are
engineering and which parts are editing, and because I would rather be judged on the
experiments than on my prose.
