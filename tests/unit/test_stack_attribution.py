import json
from importlib.metadata import entry_points

import pytest

from featureforge.core.intelligence.stack_attribution import (
    StackAttribution,
    attribute_failure,
    detect_stack_layers,
    layer_of_component,
)
from featureforge.core.stack import (
    CompilerError,
    DeviceError,
    NumericalMismatch,
    PerformanceRegression,
    StackLayer,
)

_BDD_INSTALLED = any(ep.name == "pytest-bdd" for ep in entry_points(group="pytest11"))
_EXTRA = () if _BDD_INSTALLED else ("-p", "pytest_bdd.plugin")


def _raised(exc):
    try:
        raise exc
    except Exception as caught:
        return caught


def test_explicit_stack_errors_win():
    err = _raised(CompilerError("lowering of op 'softmax' failed"))
    result = attribute_failure(err, f"CompilerError: {err}")
    assert (result.layer, result.method, result.confidence) == (
        StackLayer.COMPILER,
        "explicit",
        0.95,
    )
    assert result.evidence == ["CompilerError: lowering of op 'softmax' failed"]


def test_explicit_hardware_beats_a_software_change():
    err = _raised(DeviceError("card 2 disagrees"))
    result = attribute_failure(err, str(err), changes={"compiler": ("0.9.1", "0.9.2")})
    assert result.layer is StackLayer.HARDWARE


def test_numerical_failure_with_one_changed_component_points_at_it():
    err = _raised(NumericalMismatch("mismatch_fraction 0.31 > 0.001"))
    changes = {"compiler": ("0.9.1", "0.9.2"), "firmware": ("1.4.2", "1.4.2")}
    result = attribute_failure(err, str(err), changes=changes)
    assert (result.layer, result.method) == (StackLayer.COMPILER, "change")
    assert "compiler (0.9.1 → 0.9.2)" in result.evidence[0]


def test_numerical_failure_with_several_changes_says_bisect():
    err = _raised(NumericalMismatch("cosine 0.98 < 0.9999"))
    changes = {"compiler": ("1", "2"), "runtime": ("3", "4")}
    result = attribute_failure(err, str(err), changes=changes)
    assert result.layer is StackLayer.MODEL_NUMERICS
    assert set(result.candidates) == {StackLayer.COMPILER, StackLayer.RUNTIME}
    assert "bisect" in result.evidence[-1]


def test_performance_failure_without_evidence_stays_performance():
    err = _raised(PerformanceRegression("ttft_p50_ms +18% over budget"))
    result = attribute_failure(err, str(err))
    assert (result.layer, result.method) == (StackLayer.PERFORMANCE, "unresolved")


@pytest.mark.parametrize(
    ("message", "layer"),
    [
        ("RuntimeError: MLIR lowering failed for tile size 128", StackLayer.COMPILER),
        ("TimeoutError: firmware mailbox did not respond in 5s", StackLayer.FIRMWARE),
        ("RuntimeError: DMA transfer stalled on queue 3", StackLayer.RUNTIME),
        ("HTTPError: HTTP 503 from /v1/completions", StackLayer.SERVING),
        ("OSError: uncorrectable ECC error on card 1", StackLayer.HARDWARE),
    ],
)
def test_signals_in_the_error_name_the_layer(message, layer):
    result = attribute_failure(_raised(RuntimeError(message)), message)
    assert (result.layer, result.method) == (layer, "signals")


def test_conflicting_signals_are_unresolved_with_candidates():
    message = "compiler emitted a kernel that hit an ECC error"
    result = attribute_failure(_raised(RuntimeError(message)), message)
    assert result.layer is StackLayer.UNKNOWN
    assert set(result.candidates) == {StackLayer.COMPILER, StackLayer.HARDWARE}


def test_python_errors_in_step_code_are_test_bugs():
    namespace = {}
    code = compile("def step():\n    return undefined_name\n", "steps/board_steps.py", "exec")
    exec(code, namespace)
    try:
        namespace["step"]()
    except NameError as err:
        result = attribute_failure(err, str(err))
    assert (result.layer, result.method) == (StackLayer.TEST, "test-code")


def test_pass_on_rerun_is_flaky_and_keeps_the_first_failure():
    first = StackAttribution(layer=StackLayer.RUNTIME, confidence=0.75, method="signals")
    result = attribute_failure(None, passed_on_rerun=True, first_failure=first)
    assert result.layer is StackLayer.FLAKY
    assert "first attempt looked like RUNTIME" in result.evidence


def test_component_names_map_to_layers():
    assert layer_of_component("compiler") is StackLayer.COMPILER
    assert layer_of_component("firmware_cp") is StackLayer.FIRMWARE
    assert layer_of_component("board") is StackLayer.HARDWARE
    assert layer_of_component("model_weights") is None


def test_ui_timeouts_do_not_look_like_serving_errors():
    assert detect_stack_layers("Timeout 5030ms exceeded waiting for locator") == []


# ─── end to end ─────────────────────────────────────────────────────────────

FLAKY_PROJECT = """
[tool.pytest.ini_options]
addopts = "-p featureforge.plugin"

[tool.featureforge]
packs = []
reports_dir = "reports"
"""

FLAKY_FEATURE = """Feature: Flaky link
  Scenario: Passes on the second attempt
    Given a link that drops once
"""

FLAKY_TESTS = """
from pathlib import Path
from pytest_bdd import given, scenarios

scenarios("flaky.feature")


@given("a link that drops once")
def drops_once():
    marker = Path("attempted")
    if not marker.exists():
        marker.write_text("1")
        raise TimeoutError("PCIe link training timed out")
"""


def test_rerun_turns_a_one_off_failure_into_flaky(pytester):
    pytester.makefile(".toml", pyproject=FLAKY_PROJECT)
    pytester.makefile(".feature", flaky=FLAKY_FEATURE)
    pytester.makepyfile(test_flaky=FLAKY_TESTS)
    result = pytester.runpytest_subprocess(*_EXTRA, "--ff-reruns", "2")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*Flaky (passed on rerun): Passes on the second attempt*"])
    metrics = json.loads((pytester.path / "reports/run-metrics.json").read_text())
    (scenario,) = metrics["scenarios"]
    assert scenario["flaky"] and scenario["attempts"] == 2
    assert scenario["stack_layer"] == "FLAKY"
    assert "first attempt looked like HARDWARE" in scenario["stack_evidence"]
    assert metrics["summary"]["flaky"] == 1


def test_without_reruns_the_same_failure_is_hardware(pytester):
    pytester.makefile(".toml", pyproject=FLAKY_PROJECT)
    pytester.makefile(".feature", flaky=FLAKY_FEATURE)
    pytester.makepyfile(test_flaky=FLAKY_TESTS)
    result = pytester.runpytest_subprocess(*_EXTRA)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Stack layer: HARDWARE (75%, signals)*"])


CARD_PROJECT = """
[tool.pytest.ini_options]
addopts = "-p featureforge.plugin"

[tool.featureforge]
packs = ["inference"]
reports_dir = "reports"
device_backend = "featureforge.packs.inference.backends:simulated_card"

[tool.featureforge.tiers]
available = ["unit", "sim", "single_card", "multi_card"]
"""

COLLECT = "from pytest_bdd import scenarios\nfrom featureforge.collect import feature_paths\n"
COLLECT += "scenarios(*feature_paths())\n"


def test_a_bad_card_is_attributed_to_hardware(pytester, monkeypatch):
    monkeypatch.setenv("FF_SIM_FAULTY_CARD", "2")
    pytester.makefile(".toml", pyproject=CARD_PROJECT)
    pytester.makepyfile(test_all=COLLECT)
    result = pytester.runpytest_subprocess(*_EXTRA, "--ff-tier", "multi_card")
    result.assert_outcomes(failed=1, deselected=8)
    result.stdout.fnmatch_lines(
        ["*Stack layer: HARDWARE (95%, explicit)*", "*Card(s) 2 disagree with the other 3*"]
    )
    metrics = json.loads((pytester.path / "reports/run-metrics.json").read_text())
    assert metrics["summary"]["failures_by_stack_layer"] == {"HARDWARE": 1}
    assert metrics["summary"]["failures_by_layer"] == {}


def test_a_typo_in_step_code_is_not_blamed_on_a_changed_component():
    namespace = {}
    exec(compile("def step():\n    return nope\n", "steps/x_steps.py", "exec"), namespace)
    try:
        namespace["step"]()
    except NameError as err:
        result = attribute_failure(err, str(err), changes={"compiler": ("1", "2")})
    assert result.layer is StackLayer.TEST
