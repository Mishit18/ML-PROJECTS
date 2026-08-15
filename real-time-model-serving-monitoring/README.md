# Real-Time Model Serving and Monitoring Pipeline

Production-style ML serving project focused on the parts interviewers ask for most: reliable inference APIs, model artifacts, request validation, latency tracking, prediction logging, and basic drift monitoring.

This project is intentionally lightweight and reproducible on a laptop. The model is a scikit-learn breast-cancer risk classifier trained from a built-in dataset so the serving, monitoring, and testing workflow can be verified without private data or GPU access.

## Tech Resume Screening Summary

Verified locally:
- Pytest suite passes 11/11 tests covering API endpoints, model loading, monitoring, drift checks, and governance utilities.
- Versioned RandomForest model reaches 94.74% accuracy, 99.34% ROC-AUC, and 95.83% F1 on the held-out test split.
- Governance readiness report records p50 latency 9.10 ms, p95 latency 11.06 ms, severe PSI drift detection, and safe shadow-model agreement.
- Project demonstrates model serving, MLOps monitoring, request validation, model-card governance, Docker packaging, and production-style API design.

## Why This Project Exists

Most ML portfolio projects stop at model accuracy. This project demonstrates the next layer:

- Real-time and batch inference through FastAPI
- Versioned model artifact and metadata
- Input validation with Pydantic
- Prediction logging for auditability
- p50 / p95 latency reporting
- Feature-drift monitoring using population stability index
- Data-quality checks for missing, invalid, and out-of-range values
- Dockerized reproducibility
- Automated tests for serving and monitoring paths

## Repository Structure

```text
real-time-model-serving-monitoring/
├── artifacts/                 # Generated model and metadata
├── examples/                  # Example API payloads
├── reports/                   # Generated model card and monitoring outputs
├── scripts/
│   ├── benchmark_api.py       # Latency benchmark against running API
│   └── train_model.py         # Reproducible training + artifact generation
├── src/ml_monitoring/
│   ├── app.py                 # FastAPI app
│   ├── model.py               # Artifact loading + prediction service
│   ├── monitoring.py          # Drift, latency, and logging utilities
│   └── schemas.py             # Request/response schemas
├── tests/                     # Unit and API tests
├── Dockerfile
└── requirements.txt
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/train_model.py
pytest -q
uvicorn src.ml_monitoring.app:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d @examples/sample_request.json
```

Batch request:

```bash
curl -X POST http://localhost:8000/batch_predict ^
  -H "Content-Type: application/json" ^
  -d @examples/batch_request.json
```

Latency benchmark:

```bash
python scripts/benchmark_api.py --url http://localhost:8000/predict --requests 100
```

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model load status, version, and feature count |
| `POST /predict` | Single real-time prediction |
| `POST /batch_predict` | Batch inference |
| `POST /monitor/drift` | Feature drift report against training baseline |
| `GET /monitor/latency` | p50, p95, average latency and request count |

## Generated Metrics

Run `python scripts/train_model.py` to regenerate artifacts and metrics. Outputs:

- `artifacts/model.joblib`
- `artifacts/model_metadata.json`
- `artifacts/baseline_stats.json`
- `reports/model_card.md`

The model card records test accuracy, ROC-AUC, F1, precision, recall, feature count, dataset size, and limitations.

## Monitoring Design

### Feature Drift

The drift endpoint computes population stability index (PSI) for each feature against the training baseline. The default thresholds are:

- `PSI < 0.10`: stable
- `0.10 <= PSI < 0.25`: moderate drift
- `PSI >= 0.25`: severe drift

### Data Quality

Requests are validated for:

- Missing features
- Non-numeric values
- NaN / infinity
- Batch-size bounds
- Unexpected feature count

### Latency

Every prediction path records latency in milliseconds. The `/monitor/latency` endpoint reports request count, average, p50, p95, and max latency.

## Docker

```bash
docker build -t ml-serving-monitoring .
docker run -p 8000:8000 ml-serving-monitoring
```

## Resume Bullets

- Built a FastAPI-based real-time ML serving pipeline with `/predict`, `/batch_predict`, `/health`, latency telemetry, request logging, and feature-drift monitoring for production-style model governance.
- Versioned a scikit-learn classifier with reproducible training artifacts, model metadata, baseline feature statistics, and an auto-generated model card covering accuracy, ROC-AUC, F1, precision, recall, and limitations.
- Implemented monitoring utilities for p50/p95 inference latency, PSI-based feature drift, and data-quality validation, with automated tests covering API, model loading, and monitoring logic.

## Limitations

- Uses a public built-in dataset for reproducibility, not private production data.
- Drift detection is statistical monitoring, not automated retraining.
- The model is intentionally lightweight; the project evaluates deployment and monitoring discipline rather than claiming state-of-the-art model performance.
