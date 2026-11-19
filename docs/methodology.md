# Research Methodology & Mathematical Formulations

## 1. Estimated Glomerular Filtration Rate (eGFR)
The primary diagnostic metric for Chronic Kidney Disease staging is estimated via the 2021 CKD-EPI creatinine equation (refit without race variable):

$$\text{eGFR} = 142 \times \min\left(\frac{S_{\text{cr}}}{\kappa}, 1\right)^\alpha \times \max\left(\frac{S_{\text{cr}}}{\kappa}, 1\right)^{-1.200} \times 0.9938^{\text{Age}} \times [1.012 \text{ if Female}]$$

where:
- $S_{\text{cr}}$ is serum creatinine in mg/dL.
- $\kappa = 0.7$ for females, $0.9$ for males.
- $\alpha = -0.241$ for females, $-0.302$ for males.

## 2. Focal Loss for Severe Clinical Class Imbalance
In medical screening, imbalanced disease prevalence and borderline classes can degrade gradient updates:

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $\gamma = 2.0$ downweights easily classified non-disease samples, focusing backpropagation on borderline renal pathological cases.

## 3. Post-Hoc Probability Calibration
Predicted logits $z$ are calibrated via Platt scaling:

$$P(Y=1 \mid z) = \frac{1}{1 + \exp(-(A z + B))}$$

where $A$ and $B$ are optimized by minimizing log-loss on the validation split.
