"""Developer Brief. TS equivalent: ``src/core/intelligence/developerBrief.ts``.

Generation, persistence and rendering are separate functions here (the TS
version wrote files and printed from inside ``generateDeveloperBrief``). The
HTML renderer escapes every value: error messages and page text end up in the
report, and the TS template interpolated them raw.
"""

from __future__ import annotations

import html
import json
import secrets
import time
from pathlib import Path

from featureforge.core.intelligence.models import (
    DeveloperBrief,
    FailureClassification,
    FailureLayer,
    LayerResult,
    LayerSummary,
    LayerValidationReport,
)
from featureforge.core.llm import LLMClient, NullLLM

SUMMARY_LIMIT = 50

FIX_PROMPT = """You are a senior software engineer analyzing a BDD test failure \
in a production system.

Gherkin Step: "{step}"
Scenario: "{scenario}"
Feature: "{feature}"
Root Cause Layer: {root}
Confidence: {confidence}%
Error: "{error}"

Layer Analysis:
- UI Layer: {ui}
- API Layer: {api}
- Business Logic: {logic}

Provide a JSON response with:
{{"fix": "specific, actionable fix (2-3 sentences)",
  "impact": "business impact if unfixed (1-2 sentences)",
  "compliance": "compliance risk (HIPAA/SOC2/FDA/insurance) — omit if not applicable"}}"""

_SUMMARIES: dict[FailureLayer, str] = {
    FailureLayer.UI: (
        'UI element failure detected ({c}% confidence). The step "{s}" failed because a UI '
        "element was not found or not in the expected state. This is likely a selector "
        "change or layout update."
    ),
    FailureLayer.API_CONTRACT: (
        'API contract violation detected ({c}% confidence). The step "{s}" failed because an '
        "API endpoint returned an unexpected response. This may indicate a backend change "
        "that broke the contract."
    ),
    FailureLayer.BUSINESS_LOGIC: (
        'Business logic failure detected ({c}% confidence). The step "{s}" failed because the '
        "system returned a value that does not match the business rule. The UI and API "
        "layers appear healthy — the issue is in the calculation or rule engine."
    ),
    FailureLayer.DATA_LAYER: (
        'Data layer failure detected ({c}% confidence). The step "{s}" failed due to a database '
        "or schema issue. Check for recent migrations or schema changes."
    ),
    FailureLayer.FULL_STACK: (
        'Multi-layer failure detected across: {layers} ({c}% confidence). The step "{s}" has '
        "failures at multiple layers — this may indicate a cascading failure from a single "
        "root change."
    ),
}


def _layer_line(result: LayerResult) -> str:
    return f"{'PASSED' if result.passed else 'FAILED'} — {result.evidence or 'no evidence'}"


def summarize(classification: FailureClassification) -> str:
    template = _SUMMARIES.get(classification.root_cause)
    if template is None:
        return (
            f'Unclassified failure for step "{classification.gherkin_step}". Manual '
            f"investigation required. Error: {classification.error_message}"
        )
    return template.format(
        c=round(classification.confidence * 100),
        s=classification.gherkin_step,
        layers=" + ".join(layer.value for layer in classification.failure_layers),
    )


def _new_id() -> str:
    return f"brief-{int(time.time() * 1000)}-{secrets.token_hex(3)}"


def generate_developer_brief(
    classification: FailureClassification,
    report: LayerValidationReport,
    llm: LLMClient | None = None,
) -> DeveloperBrief:
    client = llm or NullLLM()
    answer = client.complete_json(
        FIX_PROMPT.format(
            step=classification.gherkin_step,
            scenario=classification.scenario_name,
            feature=classification.feature_name,
            root=classification.root_cause.value,
            confidence=round(classification.confidence * 100),
            error=classification.error_message[:2000],
            ui=_layer_line(report.ui),
            api=_layer_line(report.api),
            logic=_layer_line(report.business_logic),
        ),
        max_tokens=400,
        temperature=0.2,
    )
    fix = str((answer or {}).get("fix") or "").strip() or (
        f"Investigate the {classification.root_cause.value} layer for the failing step: "
        f'"{classification.gherkin_step}". Review the error message and stack trace for '
        "specific file and line references."
    )
    impact = str((answer or {}).get("impact") or "").strip() or (
        "Unable to determine business impact automatically — manual review required."
    )
    compliance = str((answer or {}).get("compliance") or "").strip() or None

    return DeveloperBrief(
        id=_new_id(),
        timestamp=classification.timestamp,
        feature=classification.feature_name,
        scenario=classification.scenario_name,
        gherkin_step=classification.gherkin_step,
        root_cause=classification.root_cause,
        confidence=classification.confidence,
        failure_layers=classification.failure_layers,
        summary=summarize(classification),
        error_message=classification.error_message,
        layer_analysis={
            "ui": LayerSummary(passed=report.ui.passed, evidence=report.ui.evidence),
            "api": LayerSummary(passed=report.api.passed, evidence=report.api.evidence),
            "business_logic": LayerSummary(
                passed=report.business_logic.passed, evidence=report.business_logic.evidence
            ),
        },
        suggested_fix=fix,
        business_impact=impact,
        compliance_risk=compliance,
        classification_method=classification.method,
        stack_trace=classification.stack_trace,
    )


def write_brief(brief: DeveloperBrief, directory: Path) -> Path:
    """Write ``<id>.json`` and append to the rolling ``failure-summary.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{brief.id}.json"
    path.write_text(brief.model_dump_json(indent=2), encoding="utf-8")

    summary_path = directory / "failure-summary.json"
    try:
        history = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            history = []
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(brief.model_dump(mode="json"))
    summary_path.write_text(json.dumps(history[-SUMMARY_LIMIT:], indent=2), encoding="utf-8")
    return path


_LAYER_LABELS = {"ui": "UI Layer", "api": "API Layer", "business_logic": "Business Logic"}


def render_console(brief: DeveloperBrief) -> str:
    divider = "═" * 70

    def mark(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    lines = [
        divider,
        "FEATUREFORGE AI — DEVELOPER BRIEF",
        divider,
        f"Feature:    {brief.feature}",
        f"Scenario:   {brief.scenario}",
        f"Step:       {brief.gherkin_step}",
        f"Root cause: {brief.root_cause.value} ({round(brief.confidence * 100)}% confidence, "
        f"{brief.classification_method})",
        divider,
        "LAYER ANALYSIS",
    ]
    for key, label in _LAYER_LABELS.items():
        result = brief.layer_analysis[key]
        lines.append(f"  [{mark(result.passed)}] {label:<15} {result.evidence or 'no data'}")
    lines += [divider, "SUMMARY", f"  {brief.summary}", divider, "SUGGESTED FIX"]
    lines += [f"  {brief.suggested_fix}", divider, "BUSINESS IMPACT", f"  {brief.business_impact}"]
    if brief.compliance_risk:
        lines += [divider, "COMPLIANCE RISK", f"  {brief.compliance_risk}"]
    lines.append(divider)
    return "\n".join(lines)


def render_html(brief: DeveloperBrief) -> str:
    e = html.escape

    def row(label: str, result: LayerSummary) -> str:
        colour, background = ("#276749", "#f0fff4") if result.passed else ("#c53030", "#fff5f5")
        status = "PASS" if result.passed else "FAIL"
        return (
            f'<tr style="background:{background}">'
            f'<td style="padding:8px;font-weight:bold;color:{colour}">{status} {e(label)}</td>'
            f'<td style="padding:8px;color:#4a5568">{e(result.evidence or "No evidence")}</td></tr>'
        )

    rows = "".join(row(_LAYER_LABELS[k], brief.layer_analysis[k]) for k in _LAYER_LABELS)
    compliance = (
        f'<h4>Compliance Risk</h4><p style="color:#c53030">{e(brief.compliance_risk)}</p>'
        if brief.compliance_risk
        else ""
    )
    layers = e(", ".join(layer.value for layer in brief.failure_layers))
    return (
        '<div style="font-family:monospace;border:2px solid #e53e3e;border-radius:8px;'
        'padding:16px;margin:8px 0;background:#fff">'
        '<h3 style="color:#c53030;margin:0 0 12px">FeatureForge AI — Developer Brief</h3>'
        f"<p><b>Root cause:</b> {e(brief.root_cause.value)} "
        f"({round(brief.confidence * 100)}% confidence, {e(brief.classification_method)})<br>"
        f"<b>Failed step:</b> {e(brief.gherkin_step)}<br>"
        f"<b>Layers affected:</b> {layers}</p>"
        f'<pre style="white-space:pre-wrap;background:#f7fafc;padding:8px">'
        f"{e(brief.error_message)}</pre>"
        f'<table style="width:100%;border-collapse:collapse">{rows}</table>'
        f"<h4>Summary</h4><p>{e(brief.summary)}</p>"
        f'<h4>Suggested Fix</h4><p style="color:#276749">{e(brief.suggested_fix)}</p>'
        f'<h4>Business Impact</h4><p style="color:#c05621">{e(brief.business_impact)}</p>'
        f"{compliance}"
        f'<p style="color:#a0aec0;font-size:11px">Brief ID: {e(brief.id)} | '
        f"{e(brief.timestamp)}</p></div>"
    )
