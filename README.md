# Pediatric Pneumonia Detection System (CXR)

> AI-powered clinical decision support system for automated pneumonia classification in pediatric chest radiographs.

![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Deployment](https://img.shields.io/badge/Deployment-Gradio-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Author
**Ömer Bibin** — Introduction to AI, Term Project

---

## 🚀 Project Overview
This project addresses the critical need for rapid and accurate pediatric pneumonia detection in chest radiographs. Moving beyond basic classification, this pipeline integrates **Explainable AI (XAI)** and **high-performance MLOps** to ensure clinical interpretability and real-time diagnostic support.

---

## 🛠️ Technical Pipeline (10-Week Evolution)

| Week | Phase | Key Achievement |
| :--- | :--- | :--- |
| 1-3 | **Setup & EDA** | High-speed data pipeline (Zip/RAM caching) |
| 4-6 | **Modeling** | Custom CNN vs. Deep CNN (Ablation Study) |
| 7-8 | **Optimization** | **DenseNet121 Transfer Learning** & GPU Starvation fix |
| 9 | **XAI** | **Grad-CAM** heatmaps for clinical transparency |
| 10 | **Deployment** | **Gradio** web interface for real-time inference |

---

## 🧠 Core Engineering Achievements

### 1. Eliminating GPU Starvation (MLOps)
We identified a bottleneck where the T4 GPU was idling due to synchronous augmentation. By refactoring the augmentation layer into an **Asynchronous `tf.data` pipeline**, we collapsed training time from **15 minutes to <30 seconds per epoch**.

### 2. Clinical Interpretability (XAI)
We implemented **Grad-CAM** to visualize model attention. This validates that the model focuses on lung infiltrates rather than image noise, making it a reliable support tool for radiologists.

### 3. Production-Ready Interface
Deployed via **Gradio**, the system allows clinicians to upload raw X-rays and instantly receive:
- Binary Diagnosis (Normal vs. Pneumonia)
- Confidence Scores
- Visual Heatmap (Grad-CAM)

---

## 📂 Repository Structure
```text
├── assets/             # Grad-CAM results & training curves
├── docs/               # Weekly technical reports (01_scope to 10_deployment)
├── models/             # Optimized final model (densenet_finetuned.keras)
├── notebooks/          # Master Notebook (Complete pipeline)
└── reports/            # Training history & clinical evaluation tables
