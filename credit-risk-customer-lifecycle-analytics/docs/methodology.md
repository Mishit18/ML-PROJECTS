# Methodology

1. Load the official Home Credit competition archive without redistributing source data.
2. Aggregate bureau credits, previous applications, and installment-payment behavior by applicant.
3. Engineer 60 application, affordability, external-score, bureau, and repayment features.
4. Create stratified train, validation, and untouched test partitions.
5. Benchmark logistic regression against LightGBM; fit isotonic calibration only on validation scores.
6. Evaluate ROC-AUC, PR-AUC, KS, Brier score, and 10-bin ECE on 46,127 test applications.
7. Build the approval frontier in DuckDB and label 45% LGD expected-loss values as modeled.
8. Export fairness and PSI stability diagnostics; exclude gender from model features.

The source has no application timestamp, so this project does not claim out-of-time validation.
