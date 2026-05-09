# Data Inventory — Chest X-Ray Pneumonia Dataset

**Author:** Ömer Bibin
**Course:** Introduction to Artificial Intelligence — Term Project  
**Document version:** 1.0 (Week 2)

---

## 1. Dataset Overview

| Attribute | Value |
|---|---|
| Name | Chest X-Ray Images (Pneumonia) |
| Source | Kaggle — `paultimothymooney/chest-xray-pneumonia` |
| Original publication | Kermany et al., *Cell*, 2018 (see `02_literature_review.md` [2]) |
| Collection site | Guangzhou Women and Children's Medical Center, China |
| Patient population | Pediatric, ages 1–5 years |
| Modality | Frontal-view (anterior-posterior) chest radiograph |
| Format | JPEG, grayscale, variable resolution (typically 1000–2000 px wide) |
| Total size | ~1.2 GB, ~5,856 images |
| License | CC BY 4.0 |
| Labels | Binary: `NORMAL` vs `PNEUMONIA` (encoded as parent-folder name) |
| Label provenance | Two expert physicians; third reviewer adjudicated disagreements |

---

## 2. Directory Structure

The archive expands into a single top-level `chest_xray/` directory with three pre-defined splits:

```
chest_xray/
├── train/
│   ├── NORMAL/        (1,341 images)
│   └── PNEUMONIA/     (3,875 images)
├── val/
│   ├── NORMAL/        (8 images)
│   └── PNEUMONIA/     (8 images)
└── test/
    ├── NORMAL/        (234 images)
    └── PNEUMONIA/     (390 images)
```

> The exact counts produced by Cell 7 of `Week02_Data_Acquisition.ipynb` are saved to `reports/data_inventory.csv` and supersede the approximate numbers above.

### Filename Convention

The filenames encode useful sub-class information that the directory structure alone discards:

| Class | Pattern | Example |
|---|---|---|
| NORMAL | `IM-XXXX-XXXX.jpeg` or `NORMAL2-IM-XXXX-XXXX.jpeg` | `IM-0115-0001.jpeg` |
| PNEUMONIA (bacterial) | `person{id}_bacteria_{n}.jpeg` | `person1_bacteria_1.jpeg` |
| PNEUMONIA (viral) | `person{id}_virus_{n}.jpeg` | `person3_virus_15.jpeg` |

For this project we use the binary label only, but the bacterial/viral distinction is recoverable from the filename if a future iteration extends the task to three classes.

---

## 3. Storage Topology

| Location | Path | Lifetime | Purpose |
|---|---|---|---|
| Persistent | `/content/drive/MyDrive/PneumoniaDetection/data/raw/chest_xray/` | Across sessions | Authoritative copy; never modified after Week 2 |
| Working copy | `/content/chest_xray/` | Per Colab session | Fast I/O for training; rebuilt from Drive each session |
| Inventory record | `/content/drive/MyDrive/PneumoniaDetection/reports/data_inventory.csv` | Across sessions | Machine-readable file counts for reproducibility |

The principle is "raw data is read-only after acquisition" — any cleaning or resizing in Week 3 will produce derivatives in `data/processed/`, leaving `data/raw/` immutable.

---

## 4. Data Dictionary

| Field | Type | Source | Description |
|---|---|---|---|
| `image` | JPEG file | filesystem | Grayscale frontal CXR |
| `label` | string | parent folder | `NORMAL` or `PNEUMONIA` |
| `split` | string | parent folder | `train`, `val`, or `test` |
| `etiology` | string | filename regex | `bacteria`, `virus`, or `none` (NORMAL) |
| `patient_id` | string | filename regex | `personN` token; multiple images may share an ID |
| `image_id` | string | filename | Full filename, used as primary key |

---

## 5. Known Constraints & Biases

These are explicit, advance-declared limitations that we will revisit in the Week 12 final report's "Threats to Validity" section.

1. **Pediatric demographic bias.** All radiographs are from patients aged 1–5 years. Pediatric chest anatomy differs substantially from adult anatomy (smaller thoracic cavity, thymic shadow, softer bone density). **Any model trained on this data cannot be claimed to generalize to adults** without retraining and revalidation on adult X-rays.

2. **Single-institution sourcing.** All images originate from one hospital in Guangzhou. Imaging hardware, exposure protocols, and post-processing pipelines vary across institutions, so the model may pick up on hospital-specific artifacts that do not exist elsewhere — a well-documented failure mode in medical AI (see Zech et al., 2018).

3. **Class imbalance.** The training split is roughly 3:1 PNEUMONIA:NORMAL. Naïve training will bias the model toward the majority class. Mitigation strategies considered for Week 5+: class-weighted loss, oversampling of NORMAL, focal loss.

4. **Inadequate validation set.** The provided `val/` split contains only 16 images total, which is statistically useless for hyperparameter tuning. **Mitigation:** in Week 3 we will carve a proper validation set (~10 % stratified split) from the training data and treat the original `val/` as an additional holdout.

5. **Conflated etiologies.** Bacterial and viral pneumonias have distinct radiographic signatures, but the dataset's binary labeling collapses them. This may inflate intra-class variance and depress model performance.

6. **Residual label noise.** Although two-physician labeling with arbitration is rigorous, ~3–5 % residual disagreement is typical in radiology. We assume but cannot verify that the published labels are accurate.

7. **No patient-level split.** We have not yet verified that train/val/test splits are patient-disjoint. If the same `person{id}` appears across splits, our test metrics will be optimistically biased. **Action item for Week 3:** audit patient overlap across splits.

---

## 6. Reproducibility

To recreate this dataset on a fresh Colab runtime:

1. Run `Week02_Data_Acquisition.ipynb`, Cells 1–8 in order.
2. Provide your own Kaggle API credentials in Cell 3.
3. Verify that the resulting `data_inventory.csv` matches the counts checked into `reports/data_inventory.csv`.

If the SHA-256 hash of the downloaded zip differs from a future re-download, document the discrepancy — Kaggle datasets are occasionally updated by their maintainers.
