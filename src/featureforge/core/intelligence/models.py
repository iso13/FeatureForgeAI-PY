"""Typed records for the Step Intelligence Layer (TS: the interfaces in each module)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FailureLayer(StrEnum):
    UI = "UI"
    API_CONTRACT = "API_CONTRACT"
    BUSINESS_LOGIC = "BUSINESS_LOGIC"
    DATA_LAYER = "DATA_LAYER"
    FULL_STACK = "FULL_STACK"
    UNKNOWN = "UNKNOWN"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class NetworkRequest(BaseModel):
    url: str
    method: str
    status: int | None = None
    request_body: Any = None
    response_body: Any = None
    duration_ms: float | None = None


class LayerResult(BaseModel):
    layer: FailureLayer
    passed: bool
    actual: str | None = None
    expected: str | None = None
    evidence: str | None = None


class FailureClassification(BaseModel):
    gherkin_step: str
    scenario_name: str
    feature_name: str
    failure_layers: list[FailureLayer]
    root_cause: FailureLayer
    confidence: float = Field(ge=0.0, le=1.0)
    error_message: str
    stack_trace: str | None = None
    layer_results: list[LayerResult] = Field(default_factory=list)
    method: str = "heuristic"  # "heuristic" | "llm" | "heuristic-fallback"
    reasoning: str | None = None
    timestamp: str = Field(default_factory=utc_now)


class LayerValidationReport(BaseModel):
    ui: LayerResult
    api: LayerResult
    business_logic: LayerResult
    captured_requests: list[NetworkRequest] = Field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        return self.ui.passed and self.api.passed and self.business_logic.passed


class LayerSummary(BaseModel):
    passed: bool
    evidence: str | None = None


class DeveloperBrief(BaseModel):
    id: str
    timestamp: str
    feature: str
    scenario: str
    gherkin_step: str
    root_cause: FailureLayer
    confidence: float
    failure_layers: list[FailureLayer]
    summary: str
    error_message: str
    layer_analysis: dict[str, LayerSummary]
    suggested_fix: str
    business_impact: str
    compliance_risk: str | None = None
    classification_method: str = "heuristic"
    stack_trace: str | None = None
