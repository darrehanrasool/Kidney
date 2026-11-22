# Kidney Disease Benchmark Datasets & Data Sources

This directory contains external dataset links, data availability statements, and clinical data source references for the Kidney-AI research project.

---

## 📄 Attached Document

- [`dataset_links.docx`](./dataset_links.docx) — Original document containing the dataset links and data availability statement.

---

## 🌐 Dataset Links & Data Availability Statements

### 1. Chronic Kidney Disease (CKD) Dataset — Kaggle & UCI ML Repository

> **Data Availability Statement:**  
> The dataset used in this study is publicly available and can be accessed from:
> - **Kaggle**: [https://www.kaggle.com/datasets/mansoordaku/ckdisease](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
> - **UCI Machine Learning Repository**: [https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease) (New portal: [UCI Dataset 336](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease))

#### Dataset Overview
- **Cohort Size**: 400 patient records collected over 2 months in India (Apollo Hospitals).
- **Target Variable**: `class` (250 CKD positive cases, 150 non-CKD / healthy control cases).
- **Features**: 24 clinical, laboratory, and symptomatic attributes (11 numeric / continuous, 13 categorical / nominal).
- **Key Biomarkers**: Blood Urea Nitrogen (`bu`), Serum Creatinine (`sc`), Hemoglobin (`hemo`), Packed Cell Volume (`pcv`), Albumin (`al`), Specific Gravity (`sg`), Blood Glucose Random (`bgr`), Red Blood Cell Count (`rc`), White Blood Cell Count (`wc`), Hypertension (`htn`), Diabetes Mellitus (`dm`), Pedal Edema (`pe`).
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

### 2. Multi-Modal Medical Imaging Datasets (Referenced in Literature)

#### NIH DeepLesion Dataset
- **URL**: [https://nihcc.app.box.com/v/DeepLesion](https://nihcc.app.box.com/v/DeepLesion)
- **Description**: Large-scale dataset of CT scan images containing diverse lesions (including renal masses, kidney stones, and calcifications). Used for benchmarking deep learning convolutional neural network (CNN) architectures and lesion segmentation.

#### Kidney and Kidney Tumor Segmentation (KiTS) Challenge
- **Challenge URL**: [https://kits-challenge.org/](https://kits-challenge.org/)
- **Description**: High-resolution 3D contrast-enhanced abdominal CT scans for segmentation of kidney parenchyma and renal tumors.

---

### 3. Open-Source Benchmark Software & Toolkits

- **`df-analyze` AutoML Toolkit**: [https://github.com/stfxecutables/df-analyze/](https://github.com/stfxecutables/df-analyze/)
  - Automated machine learning framework for tabular medical datasets with automated feature inference, missingness imputation, feature selection (filter, wrapper, embedded LightGBM), and cross-validation benchmarking.
