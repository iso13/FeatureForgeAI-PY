"""Layer validators. TS equivalent: ``src/core/intelligence/layerValidator.ts``.

The browser is only touched in :func:`snapshot_page`; every validator works on
plain data, so they are deterministic and unit-testable.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from featureforge.core.intelligence.models import (
    FailureLayer,
    LayerResult,
    LayerValidationReport,
    NetworkRequest,
)
from featureforge.core.telemetry import span

ERROR_SELECTORS = (
    '[role="alert"]',
    ".error",
    ".error-message",
    '[data-testid*="error"]',
    '[class*="error"]',
)

ASSERTION_MARKERS = ("expect", "assert", "received", "expected", "to_equal", "toequal")


class PageSnapshot(BaseModel):
    url: str
    title: str
    error_text: str = ""


def snapshot_page(page: Any) -> PageSnapshot | None:
    """Read URL, title and the first visible error banner. Never raises."""
    if page is None:
        return None
    try:
        url, title = str(page.url), str(page.title())
    except Exception:
        return None
    error_text = ""
    for selector in ERROR_SELECTORS:
        try:
            locator = page.locator(selector).first
            if locator.is_visible():
                error_text = str(locator.inner_text()).strip()
                break
        except Exception:
            continue
    return PageSnapshot(url=url, title=title, error_text=error_text)


def validate_ui_layer(snapshot: PageSnapshot | None, step_text: str) -> LayerResult:
    with span("validateUILayer", **{"step.text": step_text}) as current:
        if snapshot is None:
            return LayerResult(
                layer=FailureLayer.UI,
                passed=True,
                evidence="No browser page available (non-UI step)",
            )
        current.set_attribute("ui.url", snapshot.url)
        current.set_attribute("ui.title", snapshot.title)
        current.set_attribute("ui.errorText", snapshot.error_text)
        if snapshot.error_text:
            return LayerResult(
                layer=FailureLayer.UI,
                passed=False,
                actual=snapshot.error_text,
                expected=f'No error indicators for step: "{step_text}"',
                evidence=f'Error element found: "{snapshot.error_text}"',
            )
        return LayerResult(
            layer=FailureLayer.UI,
            passed=True,
            actual=f"Page: {snapshot.title} at {snapshot.url}",
            expected=f'No error indicators for step: "{step_text}"',
            evidence=f"Page rendered cleanly at {snapshot.url}",
        )


def _is_api_call(request: NetworkRequest) -> bool:
    return (
        any(marker in request.url for marker in ("/api/", "/graphql", "/rest/"))
        or request.method.upper() != "GET"
        or (request.status is not None and request.status >= 400)
    )


def validate_api_layer(requests: list[NetworkRequest], step_text: str) -> LayerResult:
    with span("validateAPILayer", **{"step.text": step_text}) as current:
        api_calls = [r for r in requests if _is_api_call(r)]
        failed = [r for r in api_calls if r.status is not None and r.status >= 400]
        current.set_attribute("api.requestCount", len(api_calls))
        current.set_attribute("api.failedCount", len(failed))
        if failed:
            details = ", ".join(f"{r.method} {r.url} → {r.status}" for r in failed)
            return LayerResult(
                layer=FailureLayer.API_CONTRACT,
                passed=False,
                actual=details,
                expected="All API requests return 2xx status codes",
                evidence=f"{len(failed)} failed API request(s): {details}",
            )
        if not api_calls:
            return LayerResult(
                layer=FailureLayer.API_CONTRACT,
                passed=True,
                evidence="No API requests captured in this scenario",
            )
        return LayerResult(
            layer=FailureLayer.API_CONTRACT,
            passed=True,
            evidence=f"{len(api_calls)} API request(s) all returned successful status codes",
        )


def validate_business_logic_layer(
    requests: list[NetworkRequest], step_text: str, error_message: str
) -> LayerResult:
    """API succeeded but an assertion failed → the business rule is what broke.

    For steps with no browser traffic (model checks, pure-Python services) an
    assertion failure is itself business-logic evidence; the TS engine reported
    those as passing, which hid the most common failure type in AI packs.
    """
    with span("validateBusinessLogicLayer", **{"step.text": step_text}) as current:
        lowered = error_message.lower()
        assertion_failed = any(marker in lowered for marker in ASSERTION_MARKERS)
        successful = [
            r
            for r in requests
            if _is_api_call(r) and r.status is not None and 200 <= r.status < 300
        ]
        current.set_attribute("logic.responseCount", len(successful))
        if assertion_failed and successful:
            snapshot = json.dumps(
                [{"url": r.url, "body": r.response_body} for r in successful],
                default=str,
                indent=2,
            )[:500]
            return LayerResult(
                layer=FailureLayer.BUSINESS_LOGIC,
                passed=False,
                actual=f"API returned success but assertion failed: {error_message}",
                expected=f'Step "{step_text}" business rule to be satisfied',
                evidence=(
                    "API responses were successful but data did not match the expected "
                    f"business outcome.\nResponse snapshot: {snapshot}"
                ),
            )
        if assertion_failed and not any(_is_api_call(r) for r in requests):
            return LayerResult(
                layer=FailureLayer.BUSINESS_LOGIC,
                passed=False,
                actual=error_message.splitlines()[0] if error_message else None,
                expected=f'Step "{step_text}" business rule to be satisfied',
                evidence="Assertion failed with no API traffic: the rule under test broke",
            )
        return LayerResult(
            layer=FailureLayer.BUSINESS_LOGIC,
            passed=True,
            evidence="No business logic violations detected",
        )


def validate_all_layers(
    step_text: str,
    error_message: str,
    snapshot: PageSnapshot | None = None,
    requests: list[NetworkRequest] | None = None,
) -> LayerValidationReport:
    captured = requests or []
    return LayerValidationReport(
        ui=validate_ui_layer(snapshot, step_text),
        api=validate_api_layer(captured, step_text),
        business_logic=validate_business_logic_layer(captured, step_text, error_message),
        captured_requests=captured,
    )
