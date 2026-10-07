"""Failure classifier. TS equivalent: ``src/core/intelligence/failureClassifier.ts``.

Same decision rules as the TS engine:
  * exactly one layer signalled  → that layer, 0.85 confidence
  * exactly two layers signalled → FULL_STACK, 0.60
  * none, or more than two       → ask the LLM; without one, fall back (0.30)

Signals are word-bounded regexes instead of substring checks. In the TS version
``'500'`` matched ``Timeout 5000ms exceeded`` and turned every UI timeout into an
API failure, and one business-logic signal (``'tob\\u0435'``) contained a Cyrillic
letter, so it could never match.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from featureforge.core.intelligence.models import (
    FailureClassification,
    FailureLayer,
    LayerResult,
    LayerValidationReport,
)
from featureforge.core.llm import LLMClient, NullLLM


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


SIGNALS: dict[FailureLayer, re.Pattern[str]] = {
    FailureLayer.UI: _rx(
        r"\blocator\b",
        r"\bselector\b",
        r"\belement\b",
        r"\bvisible\b",
        r"\bclick\b",
        r"\bpage\.",
        r"\bplaywright\b",
        r"timeout waiting",
        r"waiting for (?:locator|selector)",
        r"strict mode violation",
        r"element not found",
        r"to_?be_?visible",
        r"to_?have_?text",
        r"get_?attribute",
    ),
    FailureLayer.API_CONTRACT: _rx(
        r"status code",
        r"response\.(?:body|json)",
        r"\b(?:axios|fetch|requests|httpx|urllib)\b",
        r"\b(?:400|401|403|404|409|422|429|500|502|503|504)\b",
        r"\bnetwork\b",
        r"ECONNREFUSED",
        r"connection (?:refused|reset)",
        r"request failed",
        r"unexpected token",
        r"json ?(?:parse|decode)",
        r"content-type",
    ),
    FailureLayer.BUSINESS_LOGIC: _rx(
        r"expected value",
        r"assertion failed",
        r"toBe\(",
        r"toEqual\(",
        r"toMatch",
        r"business rule",
        r"\bcalculation\b",
        r"incorrect result",
        r"wrong amount",
        r"invalid state",
    ),
    FailureLayer.DATA_LAYER: _rx(
        r"\bdatabase\b",
        r"\bsql\b",
        r"\bquery\b",
        r"\b(?:postgres|psycopg|sqlalchemy|drizzle)\b",
        r"undefined column",
        r"relation .* does not exist",
        r"foreign key",
        r"\bconstraint\b",
        r"\bmigration\b",
        r"\bschema\b",
    ),
}

# A bare AssertionError/assert only says "a check failed". It counts as a
# business-logic signal when nothing more specific (UI, API, data) is present —
# Playwright's expect() also raises AssertionError for locator checks.
GENERIC_ASSERTION = _rx(r"\bAssertionError\b", r"\bassert\b")

_EVIDENCE: dict[FailureLayer, str] = {
    FailureLayer.UI: "UI-related error signals detected in the error/stack trace",
    FailureLayer.API_CONTRACT: "API/network error signals detected in the error/stack trace",
    FailureLayer.BUSINESS_LOGIC: "Assertion/business-logic failure signals detected",
    FailureLayer.DATA_LAYER: "Database/schema error signals detected in the error/stack trace",
}

LLM_PROMPT = """You are a software quality engineer analyzing a BDD test failure.

Gherkin Step: "{step}"
Scenario: "{scenario}"
Feature: "{feature}"
Error Message: "{error}"
Stack Trace: "{stack}"

Classify this failure into exactly ONE of these categories:
- UI: Selector broken, element not found, UI rendering issue
- API_CONTRACT: API response shape changed, wrong status code, network error
- BUSINESS_LOGIC: Correct UI and API but wrong business outcome/calculation
- DATA_LAYER: Database schema change, migration issue, data integrity problem
- FULL_STACK: Multiple layers broken simultaneously
- UNKNOWN: Cannot determine from available information

Respond with JSON only:
{{"rootCause": "CATEGORY", "confidence": 0.0-1.0, "reasoning": "one sentence"}}"""


def detect_layers(text: str) -> list[FailureLayer]:
    """Layers whose signals appear in ``text``, in a stable order."""
    layers = [layer for layer, pattern in SIGNALS.items() if pattern.search(text)]
    if not layers and GENERIC_ASSERTION.search(text):
        layers.append(FailureLayer.BUSINESS_LOGIC)
    return layers


def _dedupe(layers: Sequence[FailureLayer]) -> list[FailureLayer]:
    return list(dict.fromkeys(layers))


def classify_failure(
    step_text: str,
    scenario_name: str,
    feature_name: str,
    error_message: str,
    stack_trace: str | None = None,
    llm: LLMClient | None = None,
) -> FailureClassification:
    client = llm or NullLLM()
    haystack = f"{error_message}\n{stack_trace or ''}"
    layers = detect_layers(haystack)
    results = [
        LayerResult(layer=layer, passed=False, evidence=_EVIDENCE[layer]) for layer in layers
    ]

    method = "heuristic"
    reasoning: str | None = None

    if len(layers) == 1:
        root, confidence = layers[0], 0.85
    elif len(layers) == 2:
        root, confidence = FailureLayer.FULL_STACK, 0.6
        layers.append(FailureLayer.FULL_STACK)
    else:
        prompt = LLM_PROMPT.format(
            step=step_text,
            scenario=scenario_name,
            feature=feature_name,
            error=error_message[:2000],
            stack=(stack_trace or "not available")[:4000],
        )
        answer = client.complete_json(prompt, max_tokens=200, temperature=0.1)
        parsed_root = _parse_layer(answer.get("rootCause") if answer else None)
        if answer is not None and parsed_root is not None:
            method = "llm"
            root = parsed_root
            confidence = _clamp(answer.get("confidence"), default=0.5)
            reasoning = str(answer.get("reasoning") or "") or None
            if root not in layers and root is not FailureLayer.UNKNOWN:
                layers.append(root)
                results.append(LayerResult(layer=root, passed=False, evidence=reasoning))
        else:
            method = "heuristic-fallback"
            root = layers[0] if layers else FailureLayer.UNKNOWN
            confidence = 0.3

    return FailureClassification(
        gherkin_step=step_text,
        scenario_name=scenario_name,
        feature_name=feature_name,
        failure_layers=_dedupe(layers),
        root_cause=root,
        confidence=confidence,
        error_message=error_message,
        stack_trace=stack_trace,
        layer_results=results,
        method=method,
        reasoning=reasoning,
    )


def _parse_layer(value: object) -> FailureLayer | None:
    try:
        return FailureLayer(str(value).strip().upper())
    except ValueError:
        return None


def _clamp(value: object, default: float) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)


def reconcile_with_layers(
    classification: FailureClassification, report: LayerValidationReport
) -> FailureClassification:
    """Use the layer evidence to move from symptom to cause.

    A UI or assertion failure in a scenario where an API call failed is usually
    the API's fault (the page rendered an error because the backend did). The TS
    engine reported the symptom layer.
    """
    symptom = classification.root_cause
    symptoms = (FailureLayer.UI, FailureLayer.BUSINESS_LOGIC, FailureLayer.FULL_STACK)
    if report.api.passed or symptom not in symptoms:
        return classification
    if FailureLayer.API_CONTRACT in classification.failure_layers:
        return classification  # already attributed to the API
    layers = _dedupe([*classification.failure_layers, FailureLayer.API_CONTRACT])
    return classification.model_copy(
        update={
            "root_cause": FailureLayer.API_CONTRACT,
            "failure_layers": layers,
            "confidence": 0.75,
            "method": f"{classification.method}+layers",
            "reasoning": (
                f"{symptom.value} symptom, but the API layer failed: {report.api.evidence}"
            ),
            "layer_results": [*classification.layer_results, report.api],
        }
    )
