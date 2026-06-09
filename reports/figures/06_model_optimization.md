# Model Optimization & Hyperparameter Tuning — Week 5

**Author:** Ömer  
**Course:** Introduction to Artificial Intelligence — Term Project  
**Document version:** 1.0 (Week 5)  
**Related notebook:** `notebooks/Week05_Model_Optimization.ipynb`

---

## 1. Context: What Happened in Week 4?

The Week 4 BaselineCNN achieved peak validation metrics at **Epoch 12**
(val\_recall: 0.9918, val\_auc: 0.9831). However, the training exhibited
severe instability: at Epoch 15, val\_loss spiked to **2.76** and
val\_recall collapsed to **0.05** — meaning the model was predicting
almost every image as NORMAL, missing 95% of actual pneumonia cases.

Although `restore_best_weights=True` correctly rescued the final deployed
model to its Epoch 12 state, this instability reveals a **fragile
optimization landscape** that could fail differently under a different
random seed, data ordering, or deployment distribution shift.

---

## 2. Root Cause Diagnosis

Three compounding factors caused the violent spike:

### 2.1 Pixel-Level Memorization (No Augmentation)

Without data augmentation, the model sees **identical images every
epoch**. With 4,722 training images and batch size 32, each image is
seen roughly 148 × 12 = 1,776 times by Epoch 12. The model
progressively memorizes JPEG compression artifacts, scanner vignetting
patterns, and pixel-level textures rather than learning generalizable
radiological features (consolidations, opacities, costophrenic angles).

### 2.2 Explosive Cross-Entropy on Confident Wrong Predictions

Binary cross-entropy loss is non-linear in the confidence regime:

| Sigmoid output | True label | Loss value |
|---|---|---|
| 0.90 | 1.0 | −log(0.90) ≈ 0.11 |
| 0.99 | 1.0 | −log(0.01) ≈ 4.60 |
| 0.999 | 1.0 | −log(0.001) ≈ 6.91 |

A memorizing model becomes maximally confident. When it encounters
even a small batch of misclassified examples, the loss contribution
per sample is near-infinite — producing the observed spike.

### 2.3 Class-Weighted Loss Amplification

The all-NORMAL prediction shift triggers 3,509 simultaneous PNEUMONIA
misclassifications (each weighted by 0.674), producing a synchronized
loss explosion. The `restore_best_weights` callback recovered the
best weights, but the underlying fragility remained unaddressed.

---

## 3. Proposed Solutions

### 3.1 Data Augmentation as a Dynamic Regularizer

Data augmentation is not merely a dataset-expansion trick; it is a
form of **learned invariance regularization**. By randomly transforming
images each epoch, it:

- Creates a practically infinite variety of training examples from the
  same 4,722 source images
- Prevents the model from anchoring to specific pixel patterns
- Smooths the loss landscape by introducing controlled gradient variance
- Enforces the prior belief that diagnosis should be invariant to minor
  geometric and photometric perturbations

### 3.2 L2 Weight Regularization

L2 regularization adds a penalty term `λ × ‖W‖²` to the total loss,
which penalizes large weight magnitudes. Large weights are the direct
mechanism behind overconfident sigmoid outputs. By constraining the
weight norm, L2 regularization prevents the model from entering
confidence regimes where cross-entropy explodes.

### 3.3 Cosine Decay Learning Rate Schedule

Unlike `ReduceLROnPlateau` (which reduces LR in discrete, reactive
steps), **CosineDecay** decreases the learning rate smoothly and
predictably from `initial_lr` to `alpha × initial_lr` following a
cosine curve. This:

- Eliminates the abrupt LR change events that can trigger parameter
  drift and landscape transitions
- Allows larger updates early (exploration) and smaller updates later
  (fine-tuning convergence)
- Is the standard schedule used in state-of-the-art CV training
  pipelines (He et al., ResNets; Huang et al., DenseNets)

---

## 4. Augmentation Pipeline: Medical Validity Rationale

Chest X-ray augmentation must respect clinical constraints. An
augmented image must remain radiologically plausible — i.e., it must
resemble an X-ray that could actually be acquired in a clinical setting.

| Transform | Setting | Status | Clinical Justification |
|---|---|---|---|
| RandomFlip("horizontal") | — | ✅ Included | AP chest findings are not strictly lateralized; horizontal flips produce valid-looking radiographs |
| RandomFlip("vertical") | — | ❌ **Excluded** | Would invert the diaphragm/heart shadow — this acquisition geometry does not exist in clinical practice |
| RandomRotation | factor=0.05 (±18°) | ✅ Included | Simulates real patient-positioning variability in AP acquisition |
| RandomZoom | height/width=0.10 (±10%) | ✅ Included | Simulates detector-distance and body-habitus variation (pediatric patients aged 1–5 have significant size variance) |
| RandomTranslation | height/width=0.05 (±5%) | ✅ Included | Simulates centering variability between acquisitions |
| RandomBrightness | — | ❌ **Excluded** | Week 3 pixel-intensity gap was only 3.2 (well below the 10-unit safety threshold); large brightness changes risk erasing the subtle opacity differences that define pneumonia |
| RandomContrast | — | ❌ **Excluded** | Consolidation patterns have specific contrast characteristics; arbitrary contrast changes could corrupt the diagnosis signal |

Visual validation of all augmentations is shown in
`reports/figures/fig05_augmentation_validity.png`. Each augmented
image was manually inspected to confirm that lung fields remain
recognizable, no upside-down X-rays appear, and heart shadows
do not shift to the right side.

---

## 5. Experimental Design

Three variants are trained against the Week 4 baseline to isolate
the contribution of each regularization strategy:

| Model | Augmentation | L2 Reg | Dropout | LR Schedule | Hypothesis |
|---|---|---|---|---|---|
| Baseline (Week 4) | None | None | 0.50 | Fixed 1e-4 + ReduceLROnPlateau | Reference |
| **Model A** | ✅ Yes | None | 0.50 | Fixed 1e-4 + ReduceLROnPlateau | Augmentation alone resolves instability |
| **Model B** | ✅ Yes | 1e-4 | 0.50 | Fixed 1e-4 + ReduceLROnPlateau | Augmentation + L2 jointly prevent overconfidence |
| **Model C** | ✅ Yes | None | 0.60 | CosineDecay (1e-4 → 1e-6) | Smooth LR + higher dropout achieves the most stable trajectory |

All other settings are held constant across variants (architecture,
batch size 32, class weights, EarlyStopping patience=5 on val\_auc,
seed=42) to ensure a fair controlled comparison.

---

## 6. Results

> **Note:** Fill in this table with the Cell 4 output after running
> `Week05_Model_Optimization.ipynb`. Green cells indicate the best
> value in each column; `max_val_loss` measures spike severity
> (the primary stability diagnostic).

| Model | Best Epoch | val\_loss | max\_val\_loss | val\_accuracy | val\_precision | val\_recall | val\_auc | val\_pr\_auc |
|---|---|---|---|---|---|---|---|---|
| Baseline (Week 4) | 12 | — | **2.76** | — | — | 0.9918 | 0.9831 | — |
| Model A | | | | | | | | |
| Model B | | | | | | | | |
| Model C | | | | | | | | |

**Key success criteria (from Week 1):**

| Metric | Target | Best model achieved | Status |
|---|---|---|---|
| val\_accuracy | ≥ 0.88 | | |
| val\_recall | ≥ 0.90 | | |
| val\_auc | ≥ 0.90 | | |

---

## 7. Learning Curve Analysis

Overlay plots of val\_loss and val\_recall across all three models
are shown in `reports/figures/fig06_week5_overlay_curves.png`.

> **Note:** Complete this section after reviewing the curves.
> Address the following:
> - Did the max\_val\_loss drop significantly vs. the 2.76 baseline spike?
> - Which model showed the smoothest val\_recall trajectory?
> - Was there any model that still exhibited instability after augmentation?
> - Did Model C's cosine decay produce a noticeably smoother late-epoch
>   behavior compared to A and B?

---

## 8. Selected Model for Week 6

Based on the comparative analysis above, **Model \_\_\_** is selected
as the foundation for Week 6's full CNN evaluation, because:

> *(Complete after reviewing results. Justify in 2–3 sentences using
> the data from Section 6: which model best balances val\_recall ≥ 0.90
> with the lowest max\_val\_loss stability metric?)*

---

## 9. Threats to Validity

1. **Short training run.** EarlyStopping with patience=5 means some models
   may have stopped before finding their true optimum. Week 6 will retrain
   the selected model with a longer patience budget.
2. **Augmentation hyperparameters not tuned.** The rotation factor (0.05)
   and zoom factor (0.10) were chosen based on clinical reasoning, not
   grid search. Stronger augmentation might further improve generalization
   but risks introducing non-clinically-plausible training samples.
3. **val\_split size.** The validation set (494 images) is small enough
   that single-epoch metric fluctuations carry high variance. The
   Week 7 evaluation on the held-out test set (624 images) will be the
   authoritative performance estimate.
4. **Medical equipment shortcut (carried forward from Week 3).** The
   presence of monitoring cables and catheters in PNEUMONIA images
   remains an unresolved confound. Data augmentation does not remove
   this bias; it only prevents pixel-level memorization.

---

## 10. Artifacts

| Artifact | Path | Description |
|---|---|---|
| Model A weights | `models/model_a.keras` | Best val\_auc checkpoint |
| Model B weights | `models/model_b.keras` | Best val\_auc checkpoint |
| Model C weights | `models/model_c.keras` | Best val\_auc checkpoint |
| Training histories | `reports/history/model_{a,b,c}_history.json` | Epoch-by-epoch metrics |
| Augmentation check | `reports/figures/fig05_augmentation_validity.png` | Medical validity visual |
| Overlay curves | `reports/figures/fig06_week5_overlay_curves.png` | val\_loss and val\_recall comparison |
| Comparison table | `reports/week05_comparison.csv` | Machine-readable results table |
| Notebook | `notebooks/Week05_Model_Optimization.ipynb` | Fully runnable source |

---

## 11. References

- [1] P. Rajpurkar et al., "CheXNet," arXiv:1711.05225, 2017.
- [2] D. S. Kermany et al., "Identifying Medical Diagnoses... by Image-Based
  Deep Learning," *Cell*, vol. 172, no. 5, 2018.
- [3] R. B. Zech et al., "Variable generalization performance of a deep
  learning model to detect pneumonia in chest radiographs," *PLOS Medicine*,
  vol. 15, no. 11, 2018. *(Shortcut-learning / equipment-bias reference)*
- [4] S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE TKDE*,
  vol. 22, no. 10, 2010.
- [5] I. Loshchilov and F. Hutter, "SGDR: Stochastic Gradient Descent
  with Warm Restarts," *ICLR*, 2017. *(Cosine decay theoretical basis)*
