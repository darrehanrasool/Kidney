# Research Papers & Academic Literature

This directory contains key research manuscripts and benchmark studies collected for the Kidney-AI framework.

---

## 📚 Paper Catalog

| # | Paper Title | Authors | Publication / Year | Key Methods | File Link |
|---|-------------|---------|---------------------|-------------|-----------|
| 1 | **Detection and diagnosis of chronic kidney disease using deep learning-based heterogeneous modified artificial neural network** | Fuzhe Ma, Tao Sun, Lingyun Liu, Hongyu Jing | *Future Generation Computer Systems*, 111 (2020) 17–26 | Heterogeneous Modified ANN (HMANN), SVM, MLP-BP, Level Set Segmentation, Gabor filter | [`Ma_et_al_2020_HMANN_CKD_Diagnosis.pdf`](./Ma_et_al_2020_HMANN_CKD_Diagnosis.pdf) |
| 2 | **Detection and classification of kidney disease using convolutional neural networks** | Shaikh Abdul Hannan, Pushparaj Pal | *Journal of Neurology & Neurorehabilitation Research*, Vol. 8, Issue 2 (2023) 136 | Multi-layer 2D CNN, Anisotropic Diffusion Filtering, Ultrasound & CT Classification | [`Hannan_Pal_2023_CNN_Kidney_Disease.pdf`](./Hannan_Pal_2023_CNN_Kidney_Disease.pdf) |
| 3 | **Diagnosing and Characterizing Chronic Kidney Disease with Machine Learning: The Value of Clinical Patient Characteristics as Evidenced from an Open Dataset** | Juan Figueroa, Patrick Etim, Adithyan Karanathu Shibu, Derek Berger, Jacob Levman | *Electronics*, 13, 4326 (2024) | LightGBM, AutoML (`df-analyze`), Embedded & Filter Feature Selection, K-Means Clustering, PCA | [`Figueroa_et_al_2024_Diagnosing_Characterizing_CKD_ML.pdf`](./Figueroa_et_al_2024_Diagnosing_Characterizing_CKD_ML.pdf) |
| 4 | **Intelligent Diagnostic Prediction and Classification Models for Detection of Kidney Disease** | Ramesh Chandra Poonia, Mukesh Kumar Gupta, Ibrahim Abunadi, Amani Abdulrahman Albraikan, Fahd N. Al-Wesabi, Manar Ahmed Hamza, Tulasi B | *Healthcare*, 10, 371 (2022) | Logistic Regression, SVM, KNN, Random Forest, Chi-Square Selection, RFE Selection | [`Poonia_et_al_2022_Intelligent_Diagnostic_Models_Kidney_Disease.pdf`](./Poonia_et_al_2022_Intelligent_Diagnostic_Models_Kidney_Disease.pdf) |

---

## 🔬 Summary of Included Studies

### 1. Ma et al. (2020) — FGCS
- **Focus**: Early detection, segmentation, and diagnosis of chronic renal failure on the Internet of Medical Things (IoMT) platform.
- **Methodology**: Heterogeneous Modified Artificial Neural Network (HMANN) combining Support Vector Machines (SVM) with Multilayer Perceptron Backpropagation (MLP-BP) and Gabor wavelet filtering.
- **Results**: Reached 97.5% classification accuracy and 0.98 dice similarity on kidney image segmentation.

### 2. Hannan & Pal (2023) — JNNR
- **Focus**: Detection and classification of kidney disease (kidney stones, cystic disease, tumors) using CNNs.
- **Methodology**: 6-layer CNN with anisotropic diffusion preprocessing, max pooling, and ReLU/SoftMax activations on B-mode ultrasound and CT scan slices.
- **Results**: 87.4%–99% classification accuracy across normal vs. pathological categories.

### 3. Figueroa et al. (2024) — MDPI Electronics
- **Focus**: Systematic evaluation of clinical patient symptomatic variables (hypertension, diabetes, appetite, pedal edema, anemia, coronary artery disease) on CKD diagnostic performance.
- **Methodology**: Open-source AutoML (`df-analyze`), evaluating LightGBM, Random Forest, KNN, MLP, and SVM with embedded and prediction-based feature selection; unsupervised K-Means (K=2 and K=6) with PCA projection.
- **Results**: LightGBM achieved 99.4% to 100% predictive accuracy across holdout and 5-fold cross-validation when clinical symptomatic variables were included.

### 4. Poonia et al. (2022) — MDPI Healthcare
- **Focus**: Feature-based machine learning diagnostic prediction models comparing feature selection techniques (RFE vs. Chi-Square).
- **Methodology**: Logistic Regression, Naive Bayes, SVM, KNN, Random Forest, and ANN evaluated with 10-fold cross validation.
- **Results**: Logistic Regression with optimal Chi-Square feature selection ($5 < K < 14$) achieved 98.75% accuracy with top features including WBC count, blood glucose random, blood urea, serum creatinine, and hemoglobin.
