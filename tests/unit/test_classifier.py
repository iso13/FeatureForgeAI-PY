from featureforge.core.intelligence.classifier import classify_failure, detect_layers
from featureforge.core.intelligence.models import FailureLayer


class StubLLM:
    name = "stub"

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def complete_json(self, prompt, *, max_tokens=400, temperature=0.1):
        self.prompts.append(prompt)
        return self.answer


def classify(error, llm=None, stack=None):
    return classify_failure("Then it works", "Scenario", "Feature", error, stack, llm)


def test_single_ui_signal_is_high_confidence():
    result = classify("TimeoutError: waiting for locator('#submit') to be visible")
    assert result.root_cause is FailureLayer.UI
    assert result.confidence == 0.85
    assert result.method == "heuristic"


def test_timeout_milliseconds_are_not_http_status_codes():
    # The TS substring check matched '500' inside '5000ms' and blamed the API.
    assert FailureLayer.API_CONTRACT not in detect_layers("Timeout 5000ms exceeded on locator")


def test_http_status_detected_with_word_boundaries():
    assert detect_layers("GET /api/policy returned status 503") == [FailureLayer.API_CONTRACT]


def test_python_assertion_is_business_logic():
    result = classify("AssertionError: expected delta <= 0.02, got 0.05\nassert False")
    assert result.root_cause is FailureLayer.BUSINESS_LOGIC


def test_two_layers_become_full_stack():
    result = classify("locator click failed after POST /api/save returned 500")
    assert result.root_cause is FailureLayer.FULL_STACK
    assert result.confidence == 0.6
    assert FailureLayer.FULL_STACK in result.failure_layers


def test_ambiguous_failure_uses_llm():
    llm = StubLLM({"rootCause": "data_layer", "confidence": 0.9, "reasoning": "Stale row"})
    result = classify("Something odd happened", llm=llm)
    assert result.method == "llm"
    assert result.root_cause is FailureLayer.DATA_LAYER
    assert result.confidence == 0.9
    assert result.reasoning == "Stale row"
    assert "Something odd happened" in llm.prompts[0]


def test_llm_garbage_falls_back():
    result = classify("Something odd happened", llm=StubLLM({"rootCause": "MAGIC"}))
    assert result.method == "heuristic-fallback"
    assert result.root_cause is FailureLayer.UNKNOWN
    assert result.confidence == 0.3


def test_no_llm_configured_falls_back_without_network():
    result = classify("Something odd happened")
    assert result.method == "heuristic-fallback"


def test_llm_confidence_is_clamped():
    result = classify("Something odd", llm=StubLLM({"rootCause": "UI", "confidence": 7}))
    assert result.confidence == 1.0


def test_playwright_expect_assertion_is_ui_not_business():
    result = classify("AssertionError: Locator expected to be visible\nwaiting for get_by_text")
    assert result.root_cause is FailureLayer.UI


def test_reconcile_moves_ui_symptom_to_api_cause():
    from featureforge.core.intelligence import (
        NetworkRequest,
        reconcile_with_layers,
        validate_all_layers,
    )

    classification = classify("TimeoutError: waiting for locator('#summary')")
    failing = [NetworkRequest(url="/api/policy", method="POST", status=500)]
    report = validate_all_layers("Then summary", classification.error_message, requests=failing)
    reconciled = reconcile_with_layers(classification, report)
    assert reconciled.root_cause is FailureLayer.API_CONTRACT
    assert reconciled.method == "heuristic+layers"
    assert set(reconciled.failure_layers) >= {FailureLayer.UI, FailureLayer.API_CONTRACT}
    healthy = validate_all_layers("Then summary", "x", requests=[])
    assert reconcile_with_layers(classification, healthy) is classification
