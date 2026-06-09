# Baseline Model — Custom CNN from Scratch

**Author:** Ömer Bibin
**Course:** Introduction to AI — Term Project  
**Document version:** 1.0 (Week 4)

---

## 1. Purpose of the Baseline
A baseline establishes the floor against which all future modeling decisions are evaluated. Without it, no later improvement can be claimed scientifically.

## 2. Class Imbalance Handling
Computed via Sklearn. Training set ratio is 2.89:1 (PNEUMONIA:NORMAL).
- NORMAL (0) Weight: 1.9464
- PNEUMONIA (1) Weight: 0.6728

## 3. Architecture & Training
- 3 Conv→BN→ReLU→MaxPool blocks (32 → 64 → 128 channels, ~110K parameters).
- Trained using Adam (lr=1e-4) with EarlyStopping on val_auc.

## 4. Results (Peak at Epoch 12)
- val_loss: 0.2145
- val_accuracy: 0.8907
- val_recall: 0.9918 (Headline clinical metric)
- val_auc: 0.9831

## 5. Optimization Instability Note
At Epoch 15, severe high variance occurred with val_loss spiking to 2.76 and val_recall collapsing to 0.05, demonstrating the need for regularization in Week 5.
