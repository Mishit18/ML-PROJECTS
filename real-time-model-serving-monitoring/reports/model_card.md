# Model Card - Breast Cancer Risk Classifier

## Intended Use

Demonstration model for real-time serving, prediction logging, latency monitoring, and feature-drift checks. It is not a clinical product.

## Model

- Algorithm: RandomForestClassifier
- Model version: rf-breast-cancer-20260807204104
- Feature count: 30
- Training rows: 455
- Test rows: 114

## Test Metrics

| Metric | Value |
|---|---:|
| accuracy | 0.9474 |
| roc_auc | 0.9934 |
| f1 | 0.9583 |
| precision | 0.9583 |
| recall | 0.9583 |

## Monitoring

- Prediction requests are logged to `reports/prediction_log.jsonl`.
- Latency is tracked in memory and exposed through `/monitor/latency`.
- Feature drift is measured with population stability index through `/monitor/drift`.

## Limitations

- Public built-in dataset; no private production data.
- Drift detection is monitoring only and does not automatically retrain the model.
- This project demonstrates deployment discipline rather than state-of-the-art modeling.
