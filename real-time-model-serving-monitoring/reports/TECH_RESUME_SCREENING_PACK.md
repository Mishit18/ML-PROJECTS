# Tech Resume Screening Pack

## Resume Positioning

- Built FastAPI real-time ML serving layer with `/predict`, `/batch_predict`, `/health`, drift monitoring, latency telemetry, request logging, and Docker packaging.
- Versioned a reproducible RandomForest model artifact with metadata, baseline feature statistics, model card, and governance readiness report.
- Implemented PSI-based feature drift checks, p50/p95 latency reporting, data-quality validation, and shadow-model agreement diagnostics.
- Verified 11/11 pytest tests covering API behavior, model loading, prediction validation, drift scoring, latency tracking, and governance utilities.

## Verified Evidence

- Test suite: 11/11 passed in 19.28s.
- Model metrics: 94.74% accuracy, 99.34% ROC-AUC, 95.83% F1, 95.83% precision, 95.83% recall.
- Latency report: p50 9.10 ms, p95 11.06 ms, max 11.20 ms over the governance sample.
- Drift report: severe PSI status caught deliberately shifted monitoring data; retraining action was `collect_more_traffic`.
- Shadow agreement: 0.0% disagreement rate over the sample comparison.

## Interview Defense

The point is not the breast-cancer dataset. The point is the serving and monitoring system around a model: schema validation, artifact versioning, reproducible training, prediction logging, latency metrics, feature drift, governance reports, and tests. This is the right project to defend for ML platform, backend, MLOps, and AI engineering roles.

## Honest Scope

This is not a clinical product and does not use private production data. Drift detection is monitoring-only; it does not automatically retrain or promote a model. The resume should frame this as a production-style MLOps pipeline, not as a state-of-the-art medical model.

## Resume-Safe Bullets

- Built FastAPI ML serving pipeline with real-time/batch inference, request validation, latency telemetry, drift monitoring, Docker packaging, and governance reports.
- Versioned RandomForest model artifacts with metadata, baseline statistics, and model card; achieved 99.34% ROC-AUC and 95.83% F1 on held-out test data.
- Added PSI drift checks, p50/p95 latency reporting, shadow-model agreement diagnostics, and 11 passing pytest tests for serving and monitoring paths.
