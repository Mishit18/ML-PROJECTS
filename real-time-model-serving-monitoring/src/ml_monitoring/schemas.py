from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)

    @field_validator("features")
    @classmethod
    def validate_numeric_features(cls, values: list[float]) -> list[float]:
        for value in values:
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("features must not contain NaN or infinity")
        return values


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class DriftRequest(BaseModel):
    rows: list[list[float]] = Field(..., min_length=1, max_length=5000)


class DriftFeatureReport(BaseModel):
    feature: str
    psi: float
    status: str


class DriftResponse(BaseModel):
    rows_checked: int
    drifted_features: int
    max_psi: float
    status: str
    features: list[DriftFeatureReport]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    feature_count: int


class LatencyReport(BaseModel):
    request_count: int
    avg_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None


class ErrorResponse(BaseModel):
    detail: str | dict[str, Any]
