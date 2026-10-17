# QaidamSalineBench: a reproducible, leakage-aware salinization benchmark

> \*\*Read this before anything else. Contact Author, Raymond Xiaohu Fan, fanxiaohu@whcp.edu.cn\*\* The code's default working directory is
> `PROJECT\_DIR=/root/autodl-tmp/QaidamPov` — our own scratch layout on an AutoDL GPU box.
> Every script and runner honours the `PROJECT\_DIR` environment variable, so after cloning:
>
> ```bash
> export PROJECT\_DIR="$PWD"
> ```
>
> and every absolute path below resolves under your checkout. Nothing else is
> machine-specific.

> \*A note before you start.\* This repo is the "rebuilt" branch of the project: the original `processed\_dual/\*.npy` tensors and preprocessing scripts were lost, so everything here is regenerated from public data. That means the numbers you get will \*\*not\*\* be byte-identical to any earlier draft of the paper — they are, by construction, a new (and better-documented) dataset. If you're looking for the old 0.803, look at the paper's history, not here. We've stopped mourning it.

\---

## 1\. What this project actually does (and what it doesn't)

It answers three questions, in this order:

1. **Can we map saline land across the whole Qaidam Basin without any ground-truth labels?** Yes — using ESA WorldCover class 60 as a saline proxy over single-date Sentinel-2 imagery.
2. **Does a dual-task (multi-task) network beat a single-task one?** No. Not at this sample size. We ran the paired statistics to prove we're not just eyeballing means.
3. **Is the patch-level evaluation everyone uses actually trustworthy?** No. It leaks, badly. 100% of test patches share their source tile with the training data, and fixing that changes both the numbers and the ranking.

**What it does NOT do:** no field EC validation (we say so, repeatedly, in the paper), no multi-decadal trend analysis, no operational change-detection product. The four-class "stability" head is an *auxiliary regularizer* and a transparency baseline — not something you'd put in production.

\---

## 2\. Data: what, where, and how to get it

Everything is free and account-free. No CDSE login, no API keys, no nothing. (We spent two days discovering that CDSE would have needed an account we didn't have. Don't do that.)

### 2.1 Sentinel-2 L2A (the inputs)

* **Source:** AWS Earth Search STAC — `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`.
* **Bands used (6):** `blue, green, red, nir, swir16, swir22` → i.e. B02, B03, B04, B08, B11, B12.
* **Resolution / CRS:** 10 m (SWIR at 20 m resampled), UTM `EPSG:326xx`.
* **Format:** Cloud-Optimized GeoTIFF (COG), `uint16`. Read directly over HTTP with `rasterio` + `/vsicurl` — no bulk download required.
* **Scene selection:** lowest-cloud scene per MGRS tile inside the 2020 growing season, subject to a cloud-cover ceiling (default 15%).

### 2.2 ESA WorldCover (the labels)

* **Source:** `https://esa-worldcover.s3.eu-central-1.amazonaws.com`
* **Paths:** `/v100/2020/map/` and `/v200/2021/map/`
* **Filenames:** `ESA\_WorldCover\_10m\_2020\_v100\_{TILE}\_Map.tif` and `...\_2021\_v200\_{TILE}\_Map.tif`, where `{TILE}` is the 3°-grid tile name taken from its **lower-left** corner (e.g. `N36E093` covers 36–39°N, 93–96°E).
* **Label rule (matches the paper):**

  * saline (`1`) ⇐ class **60** (bare / sparse vegetation)
  * class **70** (snow / ice) is **excluded** from the saline label (geographically and hydro-geochemically unrelated — and the basin's perennial snow lies outside our tiles anyway)
  * everything else → non-saline (`0`)
* **Change label (2020 → 2021):** `0` stable non-saline, `1` stable saline, `2` expansion, `3` shrinkage.

> ⚠️ \*\*The v100-vs-v200 trap.\*\* 2020 is v100, 2021 is v200 — two \*different product versions\*. Their differences show up as fake "expansion". This is why the apparent expansion rate in our rebuild (≈3%) is far above the 0.7% of earlier drafts. We treat this as a product-version artefact, and say so in the paper. If you want cleaner change labels, control or correct for the version difference before diffing.

### 2.3 Study area

Qaidam Basin: **36°–39°N, 90°–100°E**. Default bbox in the pipeline is `90 36 100 39`.

\---

## 3\. Preprocessing / rebuilding the dataset

All of it lives in `rebuild/`.

```bash
cd rebuild
python selftest\_rebuild.py      # offline, synthetic GeoTIFFs, no network — run this first
bash run\_rebuild.sh gate        # 2 real tiles, \~5–10 min — proves the network path works
nohup bash run\_rebuild.sh all > logs/boot.log 2>\&1 \&   # full rebuild, \~20–40 min
bash run\_rebuild.sh status      # non-blocking progress (do NOT use tail -f)
```

**What you get** (`rebuild\_out/processed\_dual/`):

|File|Shape / dtype|Meaning|
|-|-|-|
|`train/val/test\_x.npy`|`(280/60/60, 6, 128, 128)` float32|6-band Sentinel-2 patches|
|`train/val/test\_y\_static.npy`|`(N, 128, 128)` int64|`{0 non-saline, 1 saline}`|
|`train/val/test\_y\_change.npy`|`(N, 128, 128)` int64|`{0 stable non-sal, 1 stable sal, 2 expansion, 3 shrinkage}`|
|**`patch\_tiles.npy`**|`(400,)` int64|source-tile id per patch, in `train\|val\|test` order — **this is what makes the leakage-free split possible**|
|`dataset\_card.json`|—|full provenance: bbox, dates, cloud threshold, tile→scene list, slice params, split, class shares, source URLs|

Wire it into the project root:

```bash
ln -sfn /root/autodl-tmp/QaidamPov/rebuild\_out /root/autodl-tmp/QaidamPov/data
```

**Rebuild knobs worth knowing:**

|Flag|Default|Notes|
|-|-|-|
|`--bbox W S E N`|`90 36 100 39`|study area|
|`--date-range`|`2020-06-01/2020-09-30`|growing season|
|`--max-cloud`|`15`|per-tile scene cloud ceiling|
|`--max-tiles`|`22`|tile budget (we ended up with **18** usable tiles)|
|`--region-px`|`640`|centre window per tile at 10 m (= 6.4 km)|
|`--total`|`400`|target patch count (280+60+60)|
|`--patch / --stride`|`128 / 64`|matches the paper — and yes, stride 64 = 50% overlap = the leakage source|
|`--split-mode`|`patch`|use `group` to split by tile (leakage-free)|

\---

## 4\. Code architecture — what's in the box

```
$PROJECT\_DIR/                      # the repository root (any checkout path)
├── rebuild/                       # dataset reconstruction (public sources → .npy)
│   ├── rb\_common.py               #   STAC query, WorldCover fetch, alignment, patch slicing,
│   │                              #   labelling, split, validation
│   ├── rebuild\_dataset.py         #   CLI: plan | build | all
│   ├── run\_rebuild.sh             #   one-shot runner (gate + idempotent steps + summary)
│   └── selftest\_rebuild.py        #   offline self-test (synthetic rasters, no network)
│
├── supp/                          # the experiment suite (this is the interesting part)
│   ├── common.py                  #   single source of truth: models, datasets, metrics,
│   │                              #   class weights, paired t-test (hand-rolled, no scipy)
│   ├── exp\_a\_multiseed\_lambda.py  #   EXP-A: seeds × λ grid + single-task pairing
│   ├── exp\_b\_export\_raw\_metrics.py#   EXP-B: pooled confusion matrices \& per-class metrics
│   ├── exp\_c\_unet\_baseline.py     #   EXP-C: U-Net (single \& dual head)
│   ├── exp\_d\_spatial\_split.py     #   EXP-D: leakage-free tile-level split (+ audit)
│   ├── exp\_e\_diagnostic.py        #   EXP-E: difficulty-stratified paired ΔIoU
│   ├── exp\_f\_classical\_baselines.py#  EXP-F: pixel-level Random Forest / XGBoost
│   ├── exp\_g\_prithvi\_rerun.py     #   EXP-G: Prithvi-EO-1.0-100M + LoRA (patch \& tile splits)
│   ├── exp\_h\_paired\_pixel\_tests.py#   EXP-H: paired inference on the predictions themselves,
│   │                              #     bootstrapping spatial units (tiles, or patches as a fallback)
│   ├── exp\_i\_change\_head\_loss.py  #   EXP-I: change-head loss — CE / focal / focal+balanced
│   ├── exp\_j\_split\_report.py      #   EXP-J: split census — patches per partition, shared tiles
│   ├── paired\_stats.py            #   pure-numpy inference kernel: McNemar, cluster bootstrap,
│   │                              #   pairwise cell tables (importable without torch)
│   ├── run\_supplementary.sh       #   one-shot runner (gate + idempotent steps + summary)
│   ├── run\_p1\_extra.sh            #   second runner: EXP-H / EXP-I / EXP-J (+ status, summary)
│   ├── selftest\_supp.py           #   offline self-test (synthetic data, CPU, 2 epochs)
│   └── selftest\_p1\_extra.py       #   offline self-test for the inference kernel (26/36 checks)
│
├── data/  → symlink to rebuild\_out/processed\_dual/
├── figures/                       # standalone, self-contained figure pipeline
│   ├── make\_figures.py            #   one file, no external deps beyond matplotlib/numpy;
│   │                              #   vendored "burgundy silver" theme (Times serif, >=12 pt)
│   ├── make\_fig1.py               #   study-area figure, three panels, ships its own
│   │                              #   simplified geometry in geo/ — no basemap download
│   ├── fig1\_study\_area.{pdf,png}#    (a) China (b) Qaidam bbox (c) method flow
│   ├── fig2\_architecture.{pdf,png}#   dual-task CNN block diagram
│   ├── fig3\_model\_comparison.{pdf,png}  # patch-level vs leakage-free tile-level IoU
│   └── fig4\_lambda\_ablation.{pdf,png}   # (a) IoU vs λ   (b) paired ΔIoU + 95% CI
├── results/supplementary/         # every table as JSON + Markdown (see §7)
└── DEPLOY.md                      # you are here
```

> `figures/make\_figures.py` is \*\*fully self-contained\*\*: it carries its own palette and
> rcParams, so it needs nothing from any private tooling — just `matplotlib` + `numpy`.
> It also runs a programmatic bounding-box overlap check and prints `0 overlaps` for
> each figure, then writes both a \*\*vector PDF\*\* and a \*\*300 dpi PNG\*\*.

**Design rules we held ourselves to** (so the comparison is actually fair):

* `DualTaskModel` in `supp/common.py` is **layer-for-layer identical** to the original `02\_train\_dual.py`: encoder 128→64→32 (**two** max-pools), two up-samples back to 128. (Earlier drafts of the paper described "three pooling stages" — that was a *text* bug, never a model bug. We fixed the text.)
* Normalisation uses **train-set** mean/std only; class weights, best-validation model selection, AdamW + cosine schedule all follow the original.
* **Every `std` is `ddof=1` (sample std)**, everywhere. No more mixing.
* Statistics are **paired** across identical seeds; the t-test is implemented in-house so the whole suite runs without `scipy`.
* Nothing overwrites the original scripts or `results/` — all outputs land in `results/supplementary/`.

\---

## 5\. How to run

### 5.1 Environment

```bash
# everything at once (pinned-ish; see requirements.txt for the exact list)
pip install -r requirements.txt

# or split:
# core (torch/numpy/sklearn) — everything except EXP-G and the rebuild
pip install torch torchvision numpy scikit-learn xgboost timm matplotlib

# only for the rebuild pipeline
pip install rasterio pystac-client shapely pyproj requests
```

Tested on Python 3.10–3.12, PyTorch cu124, single **RTX 4090 (24 GB)**. No multi-GPU anything — this is a small-sample benchmark, not a pretraining run.

### 5.2 Two-minute sanity check (CPU, no GPU, no network)

```bash
cd supp \&\& python selftest\_supp.py     # expect ✅ PASS
cd ../rebuild \&\& python selftest\_rebuild.py
```

If these pass, the *code* is fine. Any later failure is data or environment.

### 5.3 The gate, then the suite

```bash
cd supp
bash run\_supplementary.sh gate         # \~5 min — run this BEFORE anything long
nohup bash run\_supplementary.sh all > logs/boot.log 2>\&1 \&
bash run\_supplementary.sh status       # non-blocking; safe to re-run any time
```

`all` runs gate → EXP-A → EXP-B → EXP-C → EXP-D and is **idempotent**: finished steps are skipped. Delete a step's output folder to force a rerun.

### 5.4 Individual experiments

```bash
bash run\_supplementary.sh expA         # or expB / expC / expD
```

EXP-E / EXP-F / EXP-G are standalone (they were added later):

```bash
# EXP-E: difficulty-stratified diagnostic
python exp\_e\_diagnostic.py --data-dir ../data/processed\_dual --out-dir ../results/supplementary \\
    --seeds 42,123,2024 --epochs 30

# EXP-F: classical pixel baselines (fast, CPU-friendly)
python exp\_f\_classical\_baselines.py --data-dir ../data/processed\_dual \\
    --out-dir ../results/supplementary

# EXP-G: Prithvi-EO + LoRA — patch-level, then leakage-free
python exp\_g\_prithvi\_rerun.py --data-dir ../data/processed\_dual \\
    --out-dir ../results/supplementary --prithvi-dir /path/to/prithvi-100M \\
    --seeds 42,123,2024 --epochs 30
python exp\_g\_prithvi\_rerun.py --data-dir ../results/supplementary/expD\_spatial/data\_grouped \\
    --out-dir ../results/supplementary --tag expG\_grouped \\
    --prithvi-dir /path/to/prithvi-100M --seeds 42,123,2024 --epochs 30
```

> \*\*EXP-G needs the real Prithvi weights.\*\* Put `Prithvi\_100M.pt` in a folder and pass `--prithvi-dir`.
> If the script can't find it, it will warn and fall back to \*\*random initialisation\*\* — and then your log will say `pretrained=False`, and you must \*\*not\*\* call it a foundation model. We learned this the hard way. Check the log line `✅ Prithvi 权重匹配 N keys` / `pretrained=True` before writing a single sentence about it.

Getting the weights:

```bash
source /etc/network\_turbo
export HF\_ENDPOINT=https://hf-mirror.com
huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-1.0-100M \\
    --local-dir /root/autodl-tmp/models/prithvi-100M
# the file usually lands as Prithvi\_EO\_V1\_100M.pt — copy it to the name the script looks for:
cp /root/autodl-tmp/models/prithvi-100M/Prithvi\_EO\_V1\_100M.pt \\
   /root/autodl-tmp/models/prithvi-100M/Prithvi\_100M.pt
```

### 5.5 Runtime expectations

|Stage|Time (4090)|
|-|-|
|rebuild (full)|20–40 min|
|gate|\~5 min|
|EXP-A (15 runs)|\~75 min|
|EXP-B (reads existing ckpts)|a few minutes|
|EXP-C (6 runs)|\~40 min|
|EXP-D split + train|\~10 min|
|EXP-E (48 short runs)|15–25 min|
|EXP-F (RF + XGB, CPU+GPU)|2–5 min|
|EXP-G (3 seeds × 30 ep × 2 splits)|\~30 min|

Whole thing from a blank machine: **call it half a day**, most of it unattended.

\---

### 5.6 The paired-inference batch — EXP-H / EXP-I / EXP-J

These three (plus `expJ`) have their own runner, `supp/run\_p1\_extra.sh`, because they need
neither the λ grid nor the U-Net:

```bash
cd supp
bash run\_p1\_extra.sh gate        # \~2 min: data loads, models infer, losses train, tables written
bash run\_p1\_extra.sh expJ        # split census (seconds)
bash run\_p1\_extra.sh expH        # paired inference, patch-level split
bash run\_p1\_extra.sh expHg       # paired inference under the group-disjoint split
bash run\_p1\_extra.sh expI        # change-head loss sweep (the one step that wants a GPU)
bash run\_p1\_extra.sh summary     # one-screen digest of everything above
bash run\_p1\_extra.sh status      # non-blocking progress
```

Three things about this batch are worth knowing before you read its output.

* **The bootstrap unit is a spatial unit, not a pixel.** At \~10⁶ test pixels a pixel-level
bootstrap makes a 0.07 pp difference look astronomically significant — which is precisely
the artefact the paper is about. `expH` therefore resamples whole MGRS tiles, and prints
both intervals side by side so the contrast is visible rather than argued.
* **`expHg` needs the group-disjoint checkpoints, or it compares across two protocols.**
Pass them explicitly:

```bash
  CKPT\_DIR="$PROJECT\_DIR/results/supplementary/expA\_grouped/ckpts" \\
    bash run\_p1\_extra.sh expHg
  ```

  If the checkpoint directory and the data directory disagree about whether the split is
spatially disjoint, the runner prints a `PROTOCOL MISMATCH` banner, stamps
`protocol\_warning` into the JSON and marks the table. It will still run — it just refuses
to let the number pass as a leakage-free result.

* **`expI` labels its own output.** Below ten epochs it writes `SMOKE TEST ONLY` instead of a
verdict, because a two-epoch run has no opinion about which loss is better.

`paired\_stats.py` imports cleanly without torch, so the inference kernel is unit-testable
offline: `python selftest\_p1\_extra.py` runs 26 checks on a CPU-only box and 36 with torch
installed.

\---

## 6\. How to tune

Everything is CLI-flag or env-var. The ones that actually matter:

|Flag|Where|Default|Why you'd change it|
|-|-|-|-|
|`--seeds`|all exp|`42,123,2024`|more seeds → more power for the paired test; 3 is the honest minimum|
|`--epochs`|all exp|`30`|30 saturates on 280 patches; more just overfits|
|`--lr`|all exp|`1e-3` (5e-4 for Prithvi)|Prithvi wants the smaller one|
|`--lambda-change`|exp\_a/g, `supp`|`1.0`|the multi-task loss weight; EXP-A sweeps `{0.3,0.5,0.7,1.0}`|
|`--data-dir`|all exp|`.../data/processed\_dual`|point at `expD\_spatial/data\_grouped` for the leakage-free protocol|
|`--val-frac / --test-frac`|exp\_d|`0.15 / 0.15`|group fractions for the tile-level re-split (on 18 groups → 3/3 tiles)|
|`--max-train-pixels`|exp\_f|`800000`|classical baselines subsample train pixels; test pixels are always full|
|`--lora-r / --lora-alpha`|exp\_g|`8 / 16`|matches the original PEFT recipe|
|`--no-pretrained`|exp\_g|off|force random init (only for a deliberately labelled ablation)|

Env vars for the shell runner: `SEEDS=42,123,2024,7 LAMBDAS=1.0,0.3,0.5,0.7 bash run\_supplementary.sh expA`.

> Don't be tempted to search for the λ that "wins". It doesn't exist; EXP-A shows every λ is within one std of the single-task baseline. A "best λ" from a 3-seed sample is a λ-\*selected\* value, not an unbiased estimate. We wrote that sentence in the paper and we mean it.

\---

## 7\. How to read the results

Everything lands in `results/supplementary/<tag>/`. There are two kinds of files: `\*.json` (exact numbers) and `TABLE\_\*.md` (human-readable).

|File|What it tells you|
|-|-|
|`\_gate/grid\_results.json`|tiny pretest — data loads, model trains, metrics write out|
|`expA\_multiseed\_lambda/TABLE\_multiseed\_lambda.md`|**Table A1** mean±std per λ; **Table A2** the paired dual-vs-single test (ΔIoU, t, p, 95% CI)|
|`expB\_raw\_metrics/raw\_metrics\_all.json`|pooled confusion matrices; per-class P/R/F1; test-set class shares|
|`expC\_unet/TABLE\_unet.md`|U-Net single vs dual|
|`expD\_spatial/grouped\_split.json`|how the 18 tiles were divided (274/60/66 patches)|
|`expD\_spatial/grouped\_train\_results.json`|the **leakage-free** dual-task CNN result|
|`expE\_diagnostic/TABLE\_diagnostic.md`|paired ΔIoU on harder/rarer subsets|
|`expF\_classical/classical\_results.json`|Random Forest \& XGBoost (pixel-level)|
|`expG\_prithvi/prithvi\_results.json`|Prithvi-EO + LoRA, patch-level (`pretrained` flag inside!)|
|`expG\_grouped/prithvi\_results.json`|Prithvi-EO + LoRA, leakage-free|

### The numbers, so you know what to expect

**Patch-level split (the conventional, leaky one):**

|Model|IoU (saline)|
|-|-|
|Random Forest (pixel)|**0.926**|
|XGBoost (pixel)|0.920|
|U-Net (dual)|0.911|
|CNN (single-task)|0.902|
|Dual-task CNN|0.900|
|Prithvi-EO + LoRA|0.895|

**Leakage-free tile-level split:**

|Model|IoU (saline)|
|-|-|
|Random Forest (pixel)|**0.869**|
|Prithvi-EO + LoRA|0.856|
|CNN (single-task)|0.825|
|U-Net (dual)|0.806|
|Dual-task CNN|0.800|

Three things to take away:

1. **The multi-task head buys nothing.** Paired ΔIoU = −0.001 (p = 0.67) at λ = 1.0; no λ is significant; and under the leakage-free split every ΔIoU goes *negative*.
2. **Leakage is not a rounding error.** 100% of test patches share a tile with training data; fixing it costs 0.04–0.12 IoU and, crucially, **reorders the models** — Prithvi goes from last place to second.
3. **A six-band pixel-level Random Forest is the strongest model.** On a nearly spectrally separable proxy label, spatial deep learning buys you complexity, not accuracy. That's an uncomfortable result and we report it anyway.

**Cohen's κ (patch-level, 3 seeds — `expA\_multiseed\_lambda/`):**

|Model|κ (mean ± std)|
|-|-|
|CNN (single-task)|0.865 ± 0.001|
|Dual-task CNN (λ = 1.0)|0.860 ± 0.008|

κ and IoU tell the **same story** — the single-task CNN edges out the dual-task one by a hair, and the gap (0.005) is far inside the seed spread. Reporting two independent imbalance-aware metrics that agree is the whole point of Table 3; if they had disagreed we would have had to explain why.

\---

## 8\. Full reproduction, end to end

```bash
# 0) environment
pip install torch torchvision numpy scikit-learn xgboost timm
pip install rasterio pystac-client shapely pyproj requests

# 1) rebuild the dataset from public sources (\~20–40 min)
cd rebuild
python selftest\_rebuild.py                 # expect ✅ PASS
bash run\_rebuild.sh gate                   # expect ✅ gate passed
nohup bash run\_rebuild.sh all > logs/boot.log 2>\&1 \&
bash run\_rebuild.sh status                 # repeat until done
ln -sfn /root/autodl-tmp/QaidamPov/rebuild\_out /root/autodl-tmp/QaidamPov/data

# 2) experiment suite (\~2–2.5 h, plus EXP-E/F/G below)
cd ../supp
python selftest\_supp.py                    # expect ✅ PASS
bash run\_supplementary.sh gate
nohup bash run\_supplementary.sh all > logs/boot.log 2>\&1 \&
bash run\_supplementary.sh status

# 3) the extras
python exp\_e\_diagnostic.py   --data-dir ../data/processed\_dual --out-dir ../results/supplementary \\
     --seeds 42,123,2024 --epochs 30
python exp\_f\_classical\_baselines.py --data-dir ../data/processed\_dual --out-dir ../results/supplementary
# EXP-G needs real Prithvi weights (see §5.4), otherwise you get random-init numbers
python exp\_g\_prithvi\_rerun.py --data-dir ../data/processed\_dual --out-dir ../results/supplementary \\
     --prithvi-dir /root/autodl-tmp/models/prithvi-100M --seeds 42,123,2024 --epochs 30
python exp\_g\_prithvi\_rerun.py --data-dir ../results/supplementary/expD\_spatial/data\_grouped \\
     --out-dir ../results/supplementary --tag expG\_grouped \\
     --prithvi-dir /root/autodl-tmp/models/prithvi-100M --seeds 42,123,2024 --epochs 30

# 4) collect
bash run\_supplementary.sh summary
find ../results/supplementary -name "\*.json" -o -name "TABLE\_\*.md" | sort
```

\---

## 9\. Figures

**Yes, there is a figure script now.** It lives in `figures/make\_figures.py` and it is
deliberately self-contained — no imports from any private theme package, no hidden
data files, no "works on my machine". It carries its own palette and font settings, so
the only requirements are `matplotlib` and `numpy`.

```bash
cd figures
python make\_figures.py     # writes \*.pdf + \*.png here, prints "0 overlaps" per figure
```

What it produces:

|Figure|Content|
|-|-|
|`fig2\_architecture`|dual-task CNN block diagram — shared encoder + static head + change head, with the head detail drawn out|
|`fig3\_model\_comparison`|static saline IoU, **patch-level vs leakage-free tile-level**, side by side, with per-model Δ annotated|
|`fig4\_lambda\_ablation`|**(a)** IoU vs λ with the single-task baseline as a dashed line; **(b)** paired ΔIoU with 95% CI and the p-value printed on every point|

Three things worth knowing before you edit it:

1. **One scheme, applied everywhere: "burgundy silver"** — deep wine-red (`#7B2233`)
for the primary series, silver greys (`#B4B6BA`, `#6E7377`) for secondary, a lighter
wine tint (`#9E4A5C`, `#C08A96`) for emphasis. If you change one figure you should
change them all; a paper whose figures each wear a different palette looks like four
different papers. (Ask us how we know.)
2. **Typeface and size.** The script asks for Times New Roman and falls back through
Nimbus Roman → Liberation Serif → DejaVu Serif, so it never crashes on a box without
Times. Minimum font size is **12 pt**; on a stock Linux container without Times you'll
see `findfont: Font family 'Times New Roman' not found.` in the log and get DejaVu
instead — install `fonts-liberation` or `msttcorefonts` for the real look.
3. **The overlap check is not decoration.** `overlap\_check()` renders the figure and
compares every text bbox programmatically. Low-DPI previews hide label collisions;
this does not. All three figures currently report `0 overlaps`.

Output is both a **vector PDF** (for the manuscript) and a **300 dpi PNG** (for slides
and GitHub previews), with `bbox\_inches="tight"` so there is no dead margin.

> The study-area map (Fig. 1) is a GIS raster composite, not a `matplotlib` figure, and
> is not generated here. Everything else is.

\---

## 10\. FAQ \& the traps that bit us

**Q: `ModuleNotFoundError: No module named 'torch'` in EXP-F.**
A: You're on a build where `exp\_f` still imported the shared `common.py` (which imports torch). The current `exp\_f\_classical\_baselines.py` is deliberately self-contained (numpy + sklearn only) — make sure you have this version.

**Q: `nohup: ignoring input` then nothing, and no output files.**
A: You almost certainly forgot to pass the group file to EXP-D. Without `--groups`, `mode split` exits immediately (by design) — zero CPU, zero output, no error. Point it at `patch\_tiles.npy`.

**Q: My parallel edits to a file silently half-applied.**
A: Not our problem, but we hit it while building this: editing the same Python file with two concurrent writers loses changes silently. Do it serially.

**Q: `pretrained=False` in the EXP-G log.**
A: You forgot the weights. The comparison is then a *random-init* ViT, not Prithvi. Label it honestly or re-download.

**Q: The expansion class is \~3%, not 0.7% like the paper.**
A: WorldCover v100 vs v200. Not a bug. See §2.2.

**Q: EXP-D says `100%` leakage and I don't believe it.**
A: Print `patch\_tiles.npy`: every split contains patches from all 18 tiles. That *is* 100%. It's the whole point.

**Q: Something is `nan`.**
A: `Change F1-macro` is `nan` for single-head runs by design — there's no change head to score. Not an error.

**Q: Training is much slower than "\~30 seconds per 30-epoch run" in the old docs.**
A: That was written for a smaller run on a warmer cache. Trust your own wall clock, not old READMEs. (We've all been burned by a stale ETA.)

**Q: Can I run this on CPU?**
A: The self-tests, EXP-F, and EXP-B yes. Full training no — it'll finish eventually but you won't enjoy it.

\---

## 11\. What's deliberately missing / open

* **Ground-truth EC validation** — none. All accuracies are proxy-label agreements. We say it in the abstract, the results, and the limitations. We mean it.
* **The study-area map (Fig. 1)** — a GIS raster composite (`figures/` covers Figs. 2–4 only).
* **Multi-temporal / multi-decadal extension** — future work.
* **True leakage-free evaluation for *every* baseline under every λ** — we did it for the headline models; extending it everywhere is cheap and left as an exercise (and honestly, a good first PR).

\---

## 12\. Repository layout \& what to ship to GitHub

GitHub rejects single files over 100 MB and warns above 50 MB, so the raw tensors do
**not** belong in the git history — put them in a **GitHub Release** (2 GB/file) or on
Zenodo and keep the repo to code, small key artifacts, and figures.

**In the repo (small, safe):**

```
rebuild/        supp/           figures/
data/           → NOT committed (regenerate with rebuild/, or fetch from the Release)
results/supplementary/   \*.json + TABLE\_\*.md   (\~a few MB — commit these, they ARE the result)
DEPLOY.md       README.md       requirements.txt
```

**Not in the repo (large or not-for-release):**

|Artifact|Size|Why out|
|-|-|-|
|`rebuild\_out/processed\_dual/\*\_x.npy`|\~157 MB|raw tensors; regenerate or attach to a Release|
|`\*\_y\_static / \_y\_change.npy`|\~105 MB|same (int64; cast to uint8 and they shrink \~87%)|
|`\*.pth` checkpoints|\~tens MB each|regenerate by rerunning `supp/`|
|manuscript drafts, revision scripts|—|not part of a code release|

A ready-to-run packaging command (run from the project root on the server) is in the
companion `PACKAGE.md`. It writes a single tarball with only code + key results +
figures, and refuses to include any manuscript file.

\---

*Finally done. This started as "the dataset is gone, let's just rebuild it" and ended as "actually, nothing in the old results survives a leak-free split, and the model we were proudest of isn't the best one." That's not the paper anyone wanted to write. It is, however, the paper the data supports — and the version where you can hand someone a repo and a command and have them get the same numbers. Which is, in the end, the only kind of result worth defending.*

