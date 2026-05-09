# Exploratory Data Analysis & Preprocessing

**Author:** Ömer Bibin 
**Course:** Introduction to AI — Term Project  
**Document version:** 1.0 (Week 3)

---

## 1. Validation Split Methodology

### 1.1 Motivation
The original `val/` split contains only 16 images, statistically insufficient for hyperparameter selection. We carve a new, stratified, patient-disjoint validation set comprising ~10 % of the original training data.

### 1.2 Algorithm
We use `sklearn.model_selection.StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` and take the first fold. This splitter simultaneously:
- preserves the NORMAL/PNEUMONIA class ratio (stratification), and
- guarantees no `patient_group` appears in both train and val (grouping).

Patient identifiers are extracted from filenames:

| Class | Pattern | Group ID |
|---|---|---|
| PNEUMONIA | `person{N}_*.jpeg` | `pneu_p{N}` |
| NORMAL    | `IM-{NNNN}-*.jpeg` (or `NORMAL2-IM-...`) | `norm_IM{NNNN}` |
| Either, regex miss | — | `singleton_{filename}` (own group) |

### 1.3 Resulting Splits
*(Paste the printed table from Cell 2f here.)*

| split | NORMAL | PNEUMONIA | TOTAL | PNEU% |
|---|---|---|---|---|
| train_split | … | … | … | … |
| val_split | … | … | … | … |
| val_original | 8 | 8 | 16 | 50.0 |
| test | 234 | 390 | 624 | 62.5 |

### 1.4 Known Limitation
NORMAL patient grouping is heuristic, not authoritative — Kermany et al. did not publish patient-to-image mapping for the NORMAL class. We document this rather than claim certainty.

---

## 2. Exploratory Data Analysis

### 2.1 Class Distribution
Insert `reports/figures/fig01_class_distribution.png`. Discuss: imbalance ratio (~2.9:1), implications for loss weighting in Week 5.

### 2.2 Sample Visualization
Insert `reports/figures/fig02_sample_grid.png`. Discuss: dimensional variability (typically 700×400 to 2700×2000 px) which justifies the 224×224 resize, and the qualitative differences a clinician would look for (consolidations, opacities) in PNEUMONIA images.

### 2.3 Pixel Intensity Analysis
Insert `reports/figures/fig03_intensity_histogram.png`. Discuss: the magnitude of the mean-intensity gap and whether brightness shortcut learning is a concern. *Threshold: a gap > 15 on the 0–255 scale warrants explicit per-image normalization (CLAHE) in Week 5.*

---

## 3. Preprocessing Pipeline

| Step | Setting | Rationale |
|---|---|---|
| Resize | 224 × 224 | Standard input for ImageNet-pretrained backbones |
| Color mode | RGB (3-channel) | Required for transfer learning; grayscale replicated |
| Normalization | `[0, 255] → [0, 1]` via `Rescaling(1./255)` | Zero-centers gradient updates, avoids saturation |
| Batch size | 32 | Memory-fits Colab T4; standard CNN convention |
| Caching | `.cache()` after rescale | Avoids repeated disk reads across epochs |
| Prefetch | `AUTOTUNE` | Overlaps GPU compute with data loading |

Augmentation (random flips, rotations, zoom) is deliberately deferred to Week 4–5, when we begin training models and need to combat overfitting.

---

## 4. Reproducibility Artifacts
- `reports/manifest.csv` — canonical (filepath, split, class, patient_group) record
- `reports/figures/fig01_class_distribution.png`
- `reports/figures/fig02_sample_grid.png`
- `reports/figures/fig03_intensity_histogram.png`

All randomness controlled via `RANDOM_SEED = 42` in numpy, random, sklearn, and tensorflow.
