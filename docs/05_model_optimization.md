# Model Optimization and Hyperparameter Tuning

**Author:** Ömer Bibin
**Course:** Introduction to AI — Term Project  
**Document version:** 1.0 (Week 5)

---

## 1. Diagnosis of Baseline Instability
The validation collapse at Epoch 15 in the baseline model indicated severe overfitting to training distribution.To resolve this, we introduced Data Augmentation (Random Rotation, Zoom, and Horizontal Flips) along with regularization mechanisms. Vertical flips were blocked to preserve medical/anatomical correctness.

## 2. Experimental Setup (3 Variations)
- **Model A:** Baseline + Data Augmentation.
- **Model B:** Baseline + Data Augmentation + L2 Kernel Regularization.
- **Model C:** Baseline + Data Augmentation + Higher Dropout (0.6) + Adjusted LR Schedule.

## 3. Comparative Evaluation
Model B effectively stabilized the training loop, mitigating the validation loss spikes from 2.76 down to 1.55 while successfully restoring the clinical headline metric (**val_recall: 0.9918**). Model B is selected as the optimal architecture moving forward.
