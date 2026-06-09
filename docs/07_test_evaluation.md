# Clinical Evaluation on Hold-Out Test Set

**Author:** Ömer Bibin
**Course:** Introduction to AI — Term Project  
**Document version:** 1.0 (Week 7)

---

## 1. Evaluation Methodology
In Week 7, we evaluated our two primary models against the canonical `test_ds` (624 images), which served as a completely unseen hold-out set. This step is critical to measure the true real-world generalization capability of the architectures developed in Week 5 (Baseline Model B) and Week 6 (Deep CNN).

## 2. Clinical Metrics Breakdown

| Metric | Baseline (Model B) | Deep CNN (5-Block) |
|---|---|---|
| **Sensitivity (Recall)** | **0.9897** | 0.9769 |
| **Specificity** | 0.1068 | **0.6453** |
| **Precision** | 0.6487 | **0.8211** |
| **F1-Score** | 0.7837 | **0.8923** |
| **ROC-AUC** | 0.891 | **0.945** |
| **PR-AUC** | 0.935 | **0.961** |

## 3. Confusion Matrix Analysis (The Clinical Dilemma)
The Confusion Matrices revealed a stark contrast in model behavior:
- **Model B (The "Paranoid" Classifier):** Due to aggressive class weights, Model B caught almost all pneumonia cases (386/390). However, it suffered from a catastrophic false positive rate, diagnosing 209 out of 234 healthy children as having pneumonia (Specificity: 10.68%). Clinically, this is unviable as it would lead to massive overtreatment.
- **Deep CNN (The Balanced Expert):** The 5-block architecture demonstrated vastly superior feature extraction capabilities. While missing 5 more pneumonia cases than Model B, it correctly identified 151 healthy patients (Specificity: 64.53%). Its ROC-AUC (0.945) and PR-AUC (0.961) confirm it is a mathematically superior and much more stable classifier.

## 4. Conclusion & Justification for Week 8
The Deep CNN proved that a deeper from-scratch architecture learns better features, but the False Negatives (9 missed patients) indicate we have hit the **Information Ceiling** of our 4,722-image training set. 

To break this ceiling—maximizing Sensitivity without destroying Specificity—we must leverage prior knowledge. These empirical findings scientifically justify our transition to **Transfer Learning (ResNet/DenseNet)** in Week 8, where we will utilize ImageNet-pretrained weights to extract rich, low-level radiological features.
