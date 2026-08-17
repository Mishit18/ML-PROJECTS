# Model Card

## Purpose

Research prototype for probability-of-default ranking and approval-policy analysis.

## Data

Official Home Credit competition data: 307,511 applications, 1,716,428 bureau records, 1,670,214 previous applications, and 13,605,401 installment-payment records.

## Model

LightGBM challenger with logistic-regression baseline and validation-only isotonic calibration. Gender is excluded from model features.

## Test Results

- ROC-AUC: 0.7830 raw LightGBM
- KS: 0.4221 raw LightGBM
- Isotonic ECE: 0.0024
- Test applications: 46,127

## Limitations

No application timestamp is available, so validation is stratified rather than out-of-time. Expected loss uses a modeled 45% LGD. The artifact is not approved for production underwriting.
