import json

from featureforge.core.intelligence import (
    NetworkRequest,
    PageSnapshot,
    classify_failure,
    generate_developer_brief,
    render_console,
    render_html,
    validate_all_layers,
    write_brief,
)
from featureforge.core.intelligence.brief import SUMMARY_LIMIT
from featureforge.core.intelligence.models import FailureLayer


def ok(url="/api/policy", method="GET", status=200, body=None):
    return NetworkRequest(url=url, method=method, status=status, response_body=body)


def test_ui_layer_flags_visible_error_banner():
    snap = PageSnapshot(url="https://x/app", title="App", error_text="Session expired")
    report = validate_all_layers("Then I see my policy", "Timeout", snapshot=snap)
    assert not report.ui.passed
    assert "Session expired" in report.ui.evidence


def test_no_page_means_ui_layer_not_blamed():
    report = validate_all_layers("Then x", "AssertionError")
    assert report.ui.passed
    assert "non-UI" in report.ui.evidence


def test_failed_api_requests_are_reported():
    report = validate_all_layers(
        "When I save", "boom", requests=[ok(method="POST", status=422), ok()]
    )
    assert not report.api.passed
    assert "POST /api/policy → 422" in report.api.evidence


def test_static_asset_gets_are_not_api_calls():
    report = validate_all_layers("When I load", "boom", requests=[ok(url="/app.js")])
    assert report.api.passed
    assert "No API requests" in report.api.evidence


def test_assertion_with_successful_api_is_business_logic():
    report = validate_all_layers(
        "Then the premium is 100", "AssertionError: expected 100", requests=[ok(body={"p": 90})]
    )
    assert not report.business_logic.passed
    assert '"p": 90' in report.business_logic.evidence


def test_assertion_without_traffic_is_business_logic():
    report = validate_all_layers("Then the model is fair", "AssertionError: delta 0.05")
    assert not report.business_logic.passed
    assert not report.overall_passed


def make_brief(error="AssertionError: delta too large"):
    classification = classify_failure("Then it is fair", "S", "F", error)
    report = validate_all_layers("Then it is fair", error)
    return generate_developer_brief(classification, report)


def test_brief_without_llm_has_fallback_guidance():
    brief = make_brief()
    assert brief.root_cause is FailureLayer.BUSINESS_LOGIC
    assert "BUSINESS_LOGIC layer" in brief.suggested_fix
    assert brief.compliance_risk is None
    assert "Business logic failure detected (85% confidence)" in brief.summary


def test_brief_uses_llm_fix_when_available():
    class LLM:
        name = "stub"

        def complete_json(self, prompt, **_):
            return {
                "fix": "Recalibrate the model.",
                "impact": "Unfair decisions.",
                "compliance": "",
            }

    classification = classify_failure("Then fair", "S", "F", "AssertionError")
    brief = generate_developer_brief(classification, validate_all_layers("x", "assert"), LLM())
    assert brief.suggested_fix == "Recalibrate the model."
    assert brief.business_impact == "Unfair decisions."
    assert brief.compliance_risk is None


def test_write_brief_keeps_rolling_summary(tmp_path):
    for _ in range(SUMMARY_LIMIT + 3):
        path = write_brief(make_brief(), tmp_path)
    assert json.loads(path.read_text())["root_cause"] == "BUSINESS_LOGIC"
    history = json.loads((tmp_path / "failure-summary.json").read_text())
    assert len(history) == SUMMARY_LIMIT


def test_html_is_escaped():
    brief = make_brief(error="AssertionError: <script>alert(1)</script>")
    rendered = render_html(brief)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_console_render_lists_every_layer():
    text = render_console(make_brief())
    for label in ("UI Layer", "API Layer", "Business Logic"):
        assert label in text
