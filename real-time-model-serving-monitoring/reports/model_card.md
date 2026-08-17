# Model Card - Home Credit Default Risk

## Intended Use

Calibrated probability-of-default inference for a reproducible ML serving and monitoring demonstration. It is not a lending decision system.

## Data and Model

- Source: Home Credit Default Risk public competition data
- Applications: 307,511
- Behavioral records aggregated: 16,992,043
- Training rows: 215,257
- Validation rows: 46,127
- Test rows: 46,127
- Numeric features: 51
- Algorithm: LightGBMClassifier with isotonic calibration

## Held-out Test Metrics

| Metric | Value |
|---|---:|
| roc_auc | 0.7775 |
| pr_auc | 0.2562 |
| ks | 0.4120 |
| brier_score | 0.0667 |
| ece_10bin | 0.0025 |

## Monitoring and Governance

- Request validation and structured prediction logging.
- p50/p95 latency telemetry and PSI feature-drift monitoring.
- Shadow-model agreement and retraining-candidate decision endpoints.
- Protected gender attribute excluded from model features.

## Limitations

- Public competition data may not represent a current lending population.
- The API demonstrates engineering controls; it does not make autonomous credit decisions.
- Expected loss and approval policies require institution-specific validation before use.
