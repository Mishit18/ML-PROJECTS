# Home Credit ML Serving and Monitoring

Production-style FastAPI service for a calibrated LightGBM probability-of-default model trained on the public Home Credit Default Risk dataset. The project connects real-data feature engineering, held-out model evaluation, artifact versioning, request validation, latency telemetry, PSI drift detection, shadow evaluation, and retraining governance.

## Verified Evidence

| Item | Result |
|---|---:|
| Applications | 307,511 |
| Aggregated bureau, prior-application, and installment records | 16,992,043 |
| Numeric deployment features | 51 |
| Held-out test applications | 46,127 |
| Calibrated ROC-AUC | 0.7775 |
| KS statistic | 0.4120 |
| Expected calibration error, 10 bins | 0.0025 |
| Automated tests | 13 passing |

The split is stratified into training, calibration, and test partitions. Isotonic calibration is fitted only on the validation partition. `CODE_GENDER` is retained for offline fairness analysis in the source project but excluded from deployment features.

## Capabilities

- `POST /predict` and `POST /batch_predict` for calibrated default-risk inference.
- `GET /health` for model version and feature-contract checks.
- `POST /monitor/drift` for per-feature PSI against 2,000 sampled training rows.
- `GET /monitor/latency` for average, p50, p95, and maximum inference latency.
- `POST /monitor/shadow_agreement` for champion/challenger probability comparison.
- `POST /monitor/retraining_decision` for evidence-based monitoring actions.
- Structured prediction logs, Pydantic validation, Docker packaging, and pytest coverage.

## Reproduce

The official Kaggle archive is intentionally not committed. Accept the Home Credit competition rules and download `home-credit-default-risk.zip`, then run:

```bash
pip install -r requirements.txt
python scripts/train_model.py --archive /path/to/home-credit-default-risk.zip
python -m pytest -q
uvicorn src.ml_monitoring.app:app --host 0.0.0.0 --port 8000
```

In another shell:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json

python scripts/benchmark_api.py --requests 500 --concurrency 16 --warmup 25
```

The benchmark writes `reports/load_benchmark.json`. Results are local-machine measurements, not a cloud SLO.

Latest local run on a single Uvicorn process (`500` requests, concurrency `16`): **500/500 successful**, **198.04 requests/second**, **79.61 ms p50**, **105.20 ms p95**, and **0% errors**.

## Artifacts

- `artifacts/model.joblib`: LightGBM deployment model.
- `artifacts/calibrator.joblib`: validation-only isotonic calibrator.
- `artifacts/model_metadata.json`: version, feature contract, split sizes, and test metrics.
- `artifacts/baseline_stats.json`: drift baseline and training medians.
- `reports/model_card.md`: intended use, evaluation, governance, and limitations.
- `examples/sample_request.json`: request matching the 51-feature contract.

## Claim Boundary

This is a reproducible public-data engineering demonstration, not a live lending system. It does not make autonomous credit decisions, establish a production SLO, or claim institutional policy validation.
