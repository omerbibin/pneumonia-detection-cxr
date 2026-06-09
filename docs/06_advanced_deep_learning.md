# Advanced Deep Learning — Network Depth & Architecture Expansion

**Author:** Ömer Bibin
**Course:** Introduction to AI — Term Project  
**Document version:** 1.0 (Week 6)

---

## 1. Architectural Hypothesis
In Week 6, we tested whether expanding the model's capacity by deepening the network topology from a 3-block CNN (110K parameters) to a 5-block CNN (1.64M parameters, sequential channels: 32 -> 64 -> 128 -> 256 -> 512) would lead to better high-level feature abstraction and superior clinical performance. We integrated an advanced Cosine Decay learning rate schedule to stabilize convergence.

## 2. Theoretical Finding: The Information Ceiling
The experimental results provided a crucial scientific conclusion: **increasing network depth from scratch on this dataset does not improve clinical recall.** While the Deep CNN achieved a higher overall ROC-AUC (0.9912 vs 0.9618), it suffered from severe optimization instability. At Epoch 7, the model experienced a massive validation collapse, with `val_loss` spiking to 3.37 and `val_recall` dropping to ~0.05. This behavior is a textbook signature of an over-parameterized network finding brittle decision boundaries on a limited sample size (4,722 training images).

## 3. Empirical Comparison Matrix

| Metric | Model B (Week 5) | Deep CNN (Week 6) | Verdict |
|---|---|---|---|
| **val_recall (Clinical Headline)** | **0.9918** | 0.9536 | **Model B (+3.8 pp)** |
| val_auc | 0.9618 | **0.9912** | Deep CNN (+2.9 pp) |
| val_pr_auc | 0.9859 | **0.9990** | Deep CNN (+1.3 pp) |
| max_val_loss (Stability) | **1.5598** | 3.3687 | **Model B (More Stable)** |
| val_accuracy | 0.8057 | **0.9555** | Deep CNN |
| val_precision | 0.7961 | **0.9859** | Deep CNN |

## 4. Strategic Conclusion
Although `restore_best_weights=True` rescued the Deep CNN at the Epoch 5 checkpoint, it misses 4.6% of pneumonia cases compared to Model B's 0.8%. In a clinical screening scenario, missing sick children is unacceptable.

Therefore, **Model B remains our designated Champion Baseline Model**. The Week 6 Deep CNN serves as an ablation study proving that data information capacity, not network size, is the limiting factor for from-scratch architectures. This directly justifies the transition to Transfer Learning in Week 8.

## 5. Artifacts Committed
- `models/deep_cnn.keras`
- `reports/history/deep_cnn_history.json`
- `reports/week06_comparison.csv`
