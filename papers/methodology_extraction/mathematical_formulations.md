# Mathematical Formulations: Chronic Kidney Disease Machine Learning

This document provides rigorous mathematical formulations for the clinical indices, feature selection criteria, machine learning architectures, and evaluation metrics implemented in KidneyCare-AI.

---

## 1. Clinical Nephrology Biomarkers & Estimations

### 1.1 Estimated Glomerular Filtration Rate (eGFR) — 4-Variable MDRD Equation
The Modification of Diet in Renal Disease (MDRD) equation provides a non-invasive numerical surrogate for glomerular clearance rate:

$$\text{eGFR}_{\text{MDRD}} = 175 \times (\text{S}_{\text{cr}})^{-1.154} \times (\text{Age})^{-0.203} \times [0.742 \text{ if female}] \times [1.212 \text{ if African American}]$$

Where:
- $\text{S}_{\text{cr}}$: Serum Creatinine measured in $\text{mg/dL}$.
- $\text{Age}$: Patient age in completed years.
- Normal renal clearance: $\ge 90 \text{ mL/min/1.73 m}^2$.
- Kidney failure threshold: $< 15 \text{ mL/min/1.73 m}^2$.

### 1.2 Blood Urea Nitrogen to Creatinine Ratio
Distinguishes prerenal azotemia from intrinsic parenchymal renal pathology:

$$\text{BUN/Cr Ratio} = \frac{\text{Blood Urea (mg/dL)}}{\max(\text{Serum Creatinine (mg/dL)}, 0.1)}$$

- $\text{Ratio} > 20$: Highly indicative of prerenal hypoperfusion (e.g. dehydration, heart failure).
- $10 \le \text{Ratio} \le 15$: Intrinsic renal damage (acute tubular necrosis, chronic glomerulonephritis).

---

## 2. Statistical Feature Selection

### 2.1 Chi-Square ($\chi^2$) Contingency Test
Used in Poonia et al. (2022) to evaluate the independence between categorical/discretized biomarker $X$ and the CKD target $Y$:

$$\chi^2 = \sum_{i=1}^{R} \sum_{j=1}^{C} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where:
- $O_{ij}$: Observed patient frequency in category $i$ with diagnosis $j$.
- $E_{ij} = \frac{\sum_{k=1}^C O_{ik} \sum_{k=1}^R O_{kj}}{N}$: Expected frequency under the null hypothesis of independence.

---

## 3. Machine Learning Formulations

### 3.1 Soft-Margin Support Vector Machine (Dual Formulation)
The soft-margin SVM optimizes the decision hyperplane separating patients with CKD from healthy controls:

$$\min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) - \sum_{i=1}^{n} \alpha_i$$

Subject to:
$$0 \le \alpha_i \le C \quad \forall i, \quad \sum_{i=1}^{n} \alpha_i y_i = 0$$

Using the Radial Basis Function (RBF) Gaussian kernel:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right), \quad \gamma = \frac{1}{2\sigma^2}$$

### 3.2 Random Forest Gini Impurity
For a candidate node splitting on clinical biomarker $A$ with threshold $t$:

$$I_G(p) = 1 - \sum_{k=0}^{1} p_k^2 = 1 - \left(p_{\text{CKD}}^2 + p_{\text{NotCKD}}^2\right)$$

The split criterion maximizes information gain:
$$\Delta I_G = I_G(\text{parent}) - \left(\frac{N_L}{N} I_G(\text{left}) + \frac{N_R}{N} I_G(\text{right})\right)$$

### 3.3 Deep Tabular Multilayer Perceptron
A multi-stage feedforward transformation mapping preprocessed patient feature vector $\mathbf{x} \in \mathbb{R}^d$:

$$\mathbf{h}^{(1)} = \text{ReLU}\left(\text{BatchNorm}\left(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}\right)\right)$$
$$\mathbf{h}^{(2)} = \text{ReLU}\left(\text{BatchNorm}\left(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)}\right)\right)$$
$$\mathbf{z} = \mathbf{W}^{(3)}\mathbf{h}^{(2)} + \mathbf{b}^{(3)}$$
$$\hat{p}(\text{CKD} \mid \mathbf{x}) = \frac{\exp(z_1)}{\exp(z_0) + \exp(z_1)}$$

Cross-Entropy loss with label smoothing ($\epsilon = 0.05$):
$$\mathcal{L}(\mathbf{y}, \hat{\mathbf{p}}) = -\sum_{k=0}^{1} \left((1 - \epsilon) y_k + \frac{\epsilon}{2}\right) \log \hat{p}_k$$

---

## 4. Calibration & Decision Metrics

### 4.1 Expected Calibration Error (ECE)
Partitions predicted probabilities into $M$ bins $B_1, \dots, B_M$:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(y_i = \hat{y}_i), \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

### 4.2 Brier Score
Strictly proper scoring rule assessing mean squared probability calibration:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$

### 4.3 Youden's $J$ Statistic (Optimal Thresholding)
Selects threshold $\tau^*$ maximizing diagnostic sensitivity and specificity on validation data:

$$J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1$$
$$\tau^* = \arg\max_{\tau \in [0, 1]} J(\tau)$$
