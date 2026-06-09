# Week 9: Explainable AI (XAI) with Grad-CAM

**Author:** Ömer Bibin
**Objective:** Medical Transparency & Feature Localization

---

## 1. The "Black Box" Problem
In clinical settings, a high-accuracy model is insufficient if it cannot justify its decisions. Our DenseNet121 model achieved high AUC scores, but we must verify that it is identifying **pathological features** (e.g., consolidations, infiltrates) rather than exploiting spatial artifacts or background text in the X-rays.

## 2. Grad-CAM Methodology
We implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize the internal attention mechanism of the model:
1. **Gradient Extraction:** We calculated the gradients of the target class (Pneumonia) with respect to the final convolutional layer activations.
2. **Global Average Pooling:** We performed weighted pooling of these gradients to obtain the importance weights ($\alpha_k^c$) for each feature map.
3. **Heatmap Generation:** We performed a weighted combination of forward activation maps to highlight regions that contributed most to the prediction.

$$L_{Grad-CAM}^c = ReLU(\\sum_k \\alpha_k^c A^k)$$

## 3. Clinical Validation (Visual Results)
Below are representative Grad-CAM heatmaps showing the model's focus during diagnosis:

| Original X-Ray | Grad-CAM Overlay (Attention Map) | Diagnostic Outcome |
| :--- | :--- | :--- |
| ![TP](assets/tp_case.png) | ![TP_Heat](assets/tp_gradcam.png) | True Positive (Correct) |
| ![TN](assets/tn_case.png) | ![TN_Heat](assets/tn_gradcam.png) | True Negative (Correct) |

*Interpretation:* Red regions indicate areas where the model found high density of pneumonia-related features. Blue regions indicate low relevance.

## 4. Conclusion
The Grad-CAM analysis provides visual confirmation that our model correctly focuses on the lung parenchyma, validating the model’s clinical utility for pediatrician support.
