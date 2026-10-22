# GitHub metadata — copy-paste block

## Repository name

```
QaidamSalineBench
```

## One-line title (also the "About" headline)

```
A leakage-audited Sentinel-2 benchmark showing that in proxy-labelled salinity mapping the label and the split, not the architecture, set the reported accuracy
```

Shorter variant if a field is length-limited:

```
Leakage-audited salinity benchmark: the label and the split, not the architecture, set the reported accuracy
```

## One-paragraph description (repository "About" / release body)

```
A fully public-data, from-scratch benchmark for proxy-labelled soil-salinization mapping in the Qaidam Basin, China: 400 Sentinel-2 L2A patches over 18 MGRS tiles, labelled by the bare-ground class of ESA WorldCover and evaluated simultaneously under the conventional patch-level split and a leakage-free tile-level split. The repository rebuilds the dataset from open archives without credentials, reruns five model families under eight seeds, and ships every table as machine-readable JSON. Its result is deliberately uncomfortable: one spectral band with one threshold already reaches IoU 0.820 of the 0.927 that the best model achieves; patch-level evaluation inflates a U-Net by 11.6 points, of which only 1.2 come from exact duplication while 6.7 come from evaluating inside the spatial support of the training sample; and multi-task learning provides no gain that survives correction. Code, splits, per-seed metrics and figure scripts are included so the negative results can be checked as easily as the positive ones.
```

## Topics

```
remote-sensing  sentinel-2  esa-worldcover  soil-salinity  qaidam-basin
benchmark  data-leakage  spatial-cross-validation  semantic-segmentation
foundation-models  prithvi  reproducibility  negative-results  tibetan-plateau
```

## Citation stub (`CITATION.cff` body)

```yaml
cff-version: 1.2.0
message: "If you use this benchmark, please cite the journal article and the two upstream products."
authors:
  - family-names: Fan
    given-names: Raymond
    email: fanxiaohu@whcp.edu.cn
title: "QaidamSalineBench: a leakage-audited Sentinel-2 benchmark for proxy-labelled soil salinization"
version: 7.2
date-released: 2026-10-22
license: MIT
references:
  - type: dataset
    title: "ESA WorldCover 10 m 2021 v200"
    authors: "Zanaga, D. et al."
    doi: 10.5281/zenodo.7254221
```

## Suggested release notes for v7.2

```
Presentation and integrity pass; no new experiments.

Added: six numbered equations (was four); six figures embedded at first citation (was a
caption block at the end); a new Fig. 4 for the sample-size-matched arm; Appendix Table A1
with the 18-tile inventory.

Fixed: B11/B12 single-band IoU were transposed in Fig. 1 and B08 was missing entirely;
Table VI was labelled tile-level although its pixel count and static IoU identify the
patch-level split; 0.821 -> 0.820 throughout.

Changed: em dashes 75 -> 0; long parentheses 13 -> 3; body text -15 % (10,867 -> 9,241
words) by removing duplicated statements; figures redrawn in the Morandi palette at
16 pt minimum, 600 dpi plus vector.
```
