# Week 8: Transfer Learning & High-Performance MLOps Optimization

**Author:** Ömer Bibin
**Course:** Introduction to AI — Term Project  
**Document Version:** 1.0 (Week 8)

---

## 1. Executive Summary & Strategy Shift
In Week 7, our rigorous hold-out validation demonstrated that while deeper custom CNN architectures expand feature abstraction capabilities, training from scratch on a limited sample size (4,722 images) hits a definitive information ceiling. The custom models either suffered from hyper-sensitivity (overly paranoid with poor specificity) or experienced optimization instability.

To systematically break this ceiling, Week 8 introduced the **Transfer Learning paradigm**. Instead of forcing the network to learn low-level visual primitives (edges, textures, shapes) from scratch, we leveraged `DenseNet121` pre-trained on the massive ImageNet dataset (1.2M+ natural images). DenseNet’s feature reuse strategy via dense connectivity blocks makes it the definitive literature benchmark for identifying subtle pulmonary infiltrates and consolidations in pediatric chest radiographs (CXR).

---

## 2. MLOps Breakthrough: Eliminating GPU Starvation
During initial implementation, the training sequence encountered a critical performance bottleneck, stalling at ~15 minutes per epoch (5–6 seconds per step). A comprehensive MLOps diagnostic revealed a classic architectural trap: **GPU Starvation via Synchronous CPU Blocking.**

### The Bottleneck Mechanics
1. **Hardware Misconfiguration:** The local environment had temporarily dropped the T4 GPU connection, falling back to sequential CPU execution.
2. **Synchronous Data Graph:** The data augmentation layers (`RandomFlip`, `RandomRotation`) were initially embedded directly inside the `tf.keras.Sequential` model topology. This forced the execution graph to pause synchronously after every batch; the GPU remained completely idle while the CPU processed spatial matrix transformations in a single thread.

### The Asynchronous Solution
We decoupled the preprocessing/augmentation pipeline entirely from the main model graph and implemented **Asynchronous Preprocessing Pipelining** leveraging the `tf.data` API:

```python
# Moving augmentation out of the model graph to background CPU workers
train_ds_optimized = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y), 
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(buffer_size=tf.data.AUTOTUNE)
