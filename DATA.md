# DATA.md — inputs, provenance, and preprocessing

Raw imagery is **not** committed: the processed tensors are ~250 MB, and every byte of them
is regenerable from public, account-free sources in 20–40 minutes. This document is the
complete specification of that transformation, so the dataset can be reconstructed exactly
without shipping it.

If you only want to *use* the benchmark, run:

```bash
cd rebuild && bash run_rebuild.sh gate && bash run_rebuild.sh all
```

Everything below is what that command does.

---

## 1. Inputs

### 1.1 Sentinel-2 L2A — the predictors

| Property | Value |
|---|---|
| Source | AWS Earth Search STAC, `https://earth-search.aws.element84.com/v1` |
| Collection | `sentinel-2-l2a` |
| Bands (6) | `blue, green, red, nir, swir16, swir22` = B02, B03, B04, B08, B11, B12 |
| Native resolution | 10 m (B11/B12 resampled from 20 m) |
| CRS | UTM `EPSG:326xx` per tile |
| Format | Cloud-Optimized GeoTIFF, `uint16`, read over HTTP via `rasterio` + `/vsicurl` |
| Access | **none required** — no CDSE account, no API key, no authentication |

Scene selection, per MGRS tile: the **lowest-cloud scene within the 2020 growing season**
(`2020-06-01/2020-09-30`), subject to a cloud-cover ceiling of 15 %. One scene per tile, so
the input is **single-date** by construction — this is a deliberate constraint, not an
oversight (see §5).

### 1.2 ESA WorldCover — the labels

| Property | Value |
|---|---|
| Source | `https://esa-worldcover.s3.eu-central-1.amazonaws.com` |
| Paths | `/v100/2020/map/` and `/v200/2021/map/` |
| Filenames | `ESA_WorldCover_10m_{2020_v100|2021_v200}_{TILE}_Map.tif` |
| Tile naming | 3° grid, named from the **lower-left** corner — `N36E093` covers 36–39 °N, 93–96 °E |
| Resolution | 10 m |
| Licence | CC BY 4.0 |

### 1.3 Prithvi-EO-1.0-100M — the foundation-model weights

Open weights (Apache-2.0), downloaded once and loaded frozen with LoRA adapters
(r = 8, α = 16) on the attention query/value projections. See `DEPLOY.md` §5.4.

### 1.4 Study area

Qaidam Basin, **36°–39 °N, 90°–100 °E**; default bbox `90 36 100 39`. Tiles intersecting
the bbox are queried, and the tile budget is 22 with **18** ending up usable after cloud
filtering.

---

## 2. Labelling

Static binary label, from WorldCover:

| WorldCover class | Static label |
|---|---|
| 60 (bare / sparse vegetation) | **1 — saline** |
| 70 (snow / ice) | **excluded** from the saline label |
| everything else | 0 — non-saline |

Class 60 is a *proxy* for saline land: bare and sparsely vegetated surfaces in an arid
endorheic basin are dominated by evaporite crusts, and that is the entire justification. It
is **not** a measurement of soil salinity, and every number in this repository is therefore
reported as **proxy-label agreement**, never as accuracy against ground truth.

Four-class change label, from differencing the two WorldCover years:

| code | meaning |
|---|---|
| 0 | stable non-saline |
| 1 | stable saline |
| 2 | expansion (non-saline 2020 → saline 2021) |
| 3 | shrinkage (saline 2020 → non-saline 2021) |

> ⚠️ **The v100-vs-v200 trap — read this before interpreting the change head.**
> 2020 comes from product version **v100** and 2021 from **v200**. These are different
> product versions, not two epochs of one product, and their disagreements surface as
> spurious "expansion". That is why the apparent expansion rate in this rebuild (≈3.1 %)
> is far above the 0.7 % of earlier drafts. Treat the change label as a *product-version
> signal with a real-salinity component*, not as observed surface change. Controlling for
> the version difference before differencing is the obvious next step, and is listed as
> future work rather than quietly fixed here.

---

## 3. Preprocessing pipeline

Implemented in `rebuild/` (`rb_common.py` holds the primitives; `rebuild_dataset.py` is the
CLI with `plan | build | all`).

1. **Tile discovery.** Query the STAC API for `sentinel-2-l2a` items intersecting the bbox
   within the date range; keep the lowest-cloud item per MGRS tile under the 15 % ceiling.
2. **Window selection.** For each tile, take a centred window of `--region-px 640` pixels
   at 10 m (6.4 km on a side).
3. **Band stacking.** Read the six assets as COGs, resample the two SWIR bands to 10 m,
   stack into `(6, H, W)` `uint16`.
4. **WorldCover alignment.** Fetch the v100 and v200 tiles covering the window, crop to the
   same grid, and map to the static and change labels defined in §2. Class 70 is masked out
   rather than labelled.
5. **Patch slicing.** Cut `128 × 128` patches with **stride 64** — i.e. 50 % overlap in both
   axes. The overlap is intentional (it matches common practice) and it is *exactly* the
   source of the leakage this benchmark exists to expose. It is never silently removed.
6. **Split.**
   - `--split-mode patch` (default): random assignment of patches to 280 / 60 / 60. Because
     patches overlap by 50 %, **100 % of test patches share their source tile with the
     training data**.
   - `--split-mode group`: assignment by **source tile**, giving 274 / 60 / 66 patches whose
     tiles are disjoint between train and test. This is the leakage-free protocol. The
     per-patch tile id is persisted in `patch_tiles.npy`, which is what makes the grouping
     auditable after the fact.
7. **Float conversion.** `uint16` → `float32`; no per-patch rescaling.
8. **Provenance dump.** `dataset_card.json` records bbox, date range, cloud threshold, the
   tile→scene mapping, slice parameters, split mode, class shares and every source URL, so
   a result can always be traced back to the split that produced it.

---

## 4. What the pipeline produces

`rebuild_out/processed_dual/`:

| File | Shape / dtype | Size | Meaning |
|---|---|---|---|
| `train/val/test_x.npy` | `(N, 6, 128, 128)` float32 | ~157 MB | six-band Sentinel-2 patches |
| `train/val/test_y_static.npy` | `(N, 128, 128)` int64 | ~52 MB | `{0 non-saline, 1 saline}` |
| `train/val/test_y_change.npy` | `(N, 128, 128)` int64 | ~52 MB | `{0,1,2,3}` as in §2 |
| `patch_tiles.npy` | `(400,)` int64 | 3.3 KB | source-tile id per patch, in train\|val\|test order |
| `dataset_card.json` | — | ~4 KB | full provenance |

`N` is 280/60/60 for the patch-level split and 274/60/66 for the group split. Total ≈ 250 MB.

Wire it into the project root once:

```bash
ln -sfn "$PROJECT_DIR/rebuild_out" "$PROJECT_DIR/data"
```

---

## 5. Normalisation and the single-date constraint

- Channel normalisation uses **training-set mean and standard deviation only** — never
  computed over the full set, never per-patch. Recomputed per split, inside `common.py`.
- Inputs are **single-date** (2020 only; the 2021 image is used solely for the change
  label). Two consequences are stated in the paper rather than engineered away:
  1. Prithvi-EO is pre-trained on multi-temporal HLS stacks with a 3-D convolution patch
     embedding. Feeding it single-date input forces the temporal axis to be averaged out,
     which discards a substantial part of what Prithvi learned. Its results are therefore
     reported as **preliminary**.
  2. The four-class change head is trained on a *label* difference between two image years,
     not on an image difference. It regularises the shared encoder; it cannot detect change
     at inference time on a single date.

---

## 6. Rebuild knobs

| Flag | Default | Notes |
|---|---|---|
| `--bbox W S E N` | `90 36 100 39` | study area |
| `--date-range` | `2020-06-01/2020-09-30` | growing season |
| `--max-cloud` | `15` | per-tile scene cloud ceiling (%) |
| `--max-tiles` | `22` | tile budget (18 usable in practice) |
| `--region-px` | `640` | centred window per tile at 10 m |
| `--total` | `400` | target patch count |
| `--patch` / `--stride` | `128 / 64` | 50 % overlap = the leakage source |
| `--split-mode` | `patch` | `group` for the leakage-free split |
| `--out-dir` | `$PROJECT_DIR/rebuild_out` | output root |

Full CLI reference: `python rebuild/rebuild_dataset.py --help`.

---

## 7. If you want to skip the rebuild

The processed tensors are not committed because they are large and trivially regenerable.
If you would rather not run the STAC pipeline, two options:

1. Attach the tensor bundle from the repository's **Releases** page (same content, same
   `dataset_card.json`), or
2. Reconstruct a *compatible* dataset with your own Sentinel-2 stack — the only hard
   requirements are the six bands in the order `blue, green, red, nir, swir16, swir22`,
   `128 × 128` patches, and a `patch_tiles.npy` carrying a source-tile id per patch. Without
   that last file the leakage-free protocol silently degrades to patch-level clustering, so
   the code now prints a warning and records `cluster_kind` in every result file.

---

## 8. Citation obligations

Please cite the upstream products alongside the paper:

- **ESA WorldCover** — © ESA WorldCover project 2021, CC BY 4.0.
- **Copernicus Sentinel-2** — ESA / Copernicus open data.
- **Prithvi-EO-1.0-100M** — IBM/NASA, Apache-2.0.
