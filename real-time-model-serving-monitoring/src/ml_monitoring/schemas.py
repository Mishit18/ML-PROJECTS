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
    probability_default: float
    probability_non_default: float
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


class RetrainingDecisionResponse(BaseModel):
    action: str
    reasons: list[str]
    severe_features: list[str]
    moderate_features: list[str]
    drifted_feature_share: float
    request_count: int


class ShadowAgreementRequest(BaseModel):
    champion_probabilities: list[float] = Field(..., min_length=1, max_length=5000)
    challenger_probabilities: list[float] = Field(..., min_length=1, max_length=5000)
    tolerance: float = Field(0.10, gt=0, le=1)


class ShadowAgreementResponse(BaseModel):
    rows_compared: int
    mean_abs_probability_delta: float
    max_abs_probability_delta: float
    disagreement_rate: float
    verdict: str


class ErrorResponse(BaseModel):
    detail: str | dict[str, Any]
