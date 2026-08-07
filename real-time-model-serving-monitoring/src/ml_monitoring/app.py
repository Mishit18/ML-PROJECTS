from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .model import ModelService
from .monitoring import LatencyTracker, compute_drift_report, validate_feature_vector
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftRequest,
    DriftResponse,
    HealthResponse,
    LatencyReport,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(
    title="Real-Time Model Serving and Monitoring Pipeline",
    version="1.0.0",
    description="FastAPI inference service with logging, latency telemetry, and PSI drift checks.",
)

model_service = ModelService()
latency_tracker = LatencyTracker()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.model is not None,
        model_version=model_service.model_version,
        feature_count=len(model_service.feature_names),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = model_service.predict(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    latency_tracker.record(result.latency_ms)
    return PredictionResponse(**result.__dict__)


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = [predict(item) for item in request.items]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.post("/monitor/drift", response_model=DriftResponse)
def monitor_drift(request: DriftRequest) -> DriftResponse:
    try:
        for row in request.rows:
            validate_feature_vector(row, expected_count=len(model_service.feature_names))
        report = compute_drift_report(
            model_service.baseline_rows,
            request.rows,
            model_service.feature_names,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DriftResponse(**report)


@app.get("/monitor/latency", response_model=LatencyReport)
def monitor_latency() -> LatencyReport:
    return LatencyReport(**latency_tracker.report())
