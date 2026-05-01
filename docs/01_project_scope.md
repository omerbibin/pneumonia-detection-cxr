# 01 — Project Scope

> **pneumonia-detection-cxr** | Introduction to AI — Term Project | Week 1/12

---

## 1. Problem Statement

Pneumonia is one of the leading causes of death worldwide, responsible for over 2.5 million deaths annually. It disproportionately affects children under five and elderly populations, particularly in low- and middle-income countries where access to trained radiologists is severely limited.

Chest X-ray (CXR) imaging is the most widely used and cost-effective diagnostic tool for pneumonia. However, accurate interpretation of CXRs requires significant clinical expertise. Radiologist shortages, high workloads, and inter-reader variability make manual diagnosis slow and error-prone. An automated, AI-based screening system could drastically reduce diagnostic delays and support overworked clinicians — especially in resource-constrained environments.

---

## 2. Motivation & Stakeholders

This project is motivated by the real-world gap between the availability of CXR equipment and the shortage of trained personnel to interpret results. Deep learning models have demonstrated radiologist-level performance on medical imaging tasks, making this a high-impact application of AI.

**Primary stakeholders:**
- **Radiologists** — the system acts as a second opinion, reducing cognitive load and improving throughput.
- **Hospitals & Clinics** — faster triage decisions, reduced bottlenecks in emergency settings.
- **Patients in low-resource settings** — access to accurate screening where specialist care is unavailable or unaffordable.

---

## 3. Project Objectives

**Primary objective:**
- Build a **binary image classifier** (Normal vs. Pneumonia) trained on publicly available chest X-ray data, achieving competitive accuracy (target: ≥ 90% on test set).

**Secondary objectives:**
- Deploy the trained model as an **interactive Gradio demo** that accepts an uploaded X-ray image and returns a prediction with confidence score.
- Document a **reproducible ML pipeline** covering data preprocessing, model training, evaluation, and deployment — suitable for replication or extension by others.

---

## 4. Scope & Out-of-Scope

### In Scope
- Binary classification: **Normal** vs. **Pneumonia** using the Kaggle Chest X-Ray Pneumonia dataset.
- Model development using TensorFlow/Keras (CNN / transfer learning).
- Performance evaluation: accuracy, precision, recall, F1, AUC-ROC.
- Simple web-based demo via Gradio (hosted on Hugging Face Spaces or Colab).
- Weekly documentation pushed to this GitHub repository.

### Out of Scope
- **Multi-disease classification** (e.g., COVID-19, tuberculosis, lung cancer) — possible future extension.
- Integration with hospital PACS or DICOM systems.
- Clinical validation or regulatory approval (this is an academic project).
- Real-time video or CT scan analysis.

---

*Last updated: Week 1 | Will be revised as project scope evolves.*
