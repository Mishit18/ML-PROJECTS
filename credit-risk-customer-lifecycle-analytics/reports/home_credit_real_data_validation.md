# Home Credit Real-Data Validation

- Applications: **307,511**
- Holdout applications: **46,127**
- Engineered model features: **60**
- Isotonic LightGBM ROC-AUC: **0.7826**
- KS: **0.4209**
- ECE (10 bins): **0.0024**
- 80% policy approval rate: **82.23%**
- Approved realized default rate: **4.66%**

## Claim Boundary

Application, bureau, previous-credit, and installment records are real competition data. Expected loss uses an explicit 45% LGD assumption and is therefore modeled, not observed profit. The source provides no application timestamp, so evaluation uses a stratified three-way split, not out-of-time validation.
