"""Regression tests for defects found in review of the quality features."""

import json
import math
from importlib.metadata import entry_points

import pydantic
import pytest

from featureforge.cli import main
from featureforge.core.gate import GatePolicy
from featureforge.packs.inference.backends import Generation, ReferenceBackend
from featureforge.packs.inference.compare import compare_logits
from featureforge.packs.inference.steps import first_divergence, odd_cards_out
from featureforge.packs.inference.tolerance import TolerancePolicy, ToleranceRule

_BDD_INSTALLED = any(ep.name == "pytest-bdd" for ep in entry_points(group="pytest11"))
_EXTRA = () if _BDD_INSTALLED else ("-p", "pytest_bdd.plugin")
_XDIST_INSTALLED = any(ep.name == "xdist.plugin" for ep in entry_points(group="pytest11"))
_XDIST = () if _XDIST_INSTALLED else ("-p", "xdist.plugin")

PYPROJECT = """
[tool.pytest.ini_options]
addopts = "-p featureforge.plugin"

[tool.featureforge]
packs = []
reports_dir = "reports"
history_dir = "history"
"""


def _history(path):
    return sorted((path / "history").glob("run-*.json"))


# ─── parallel runs ──────────────────────────────────────────────────────────


def test_parallel_workers_report_every_result_to_the_controller(pytester, monkeypatch):
    pytest.importorskip("xdist")
    pytester.makefile(".toml", pyproject=PYPROJECT)
    tests = "\n".join(f"def test_ok_{i}():\n    pass\n" for i in range(6))
    tests += "\ndef test_broken():\n    assert False\n"
    pytester.makepyfile(test_many=tests)
    result = pytester.runpytest_subprocess(*_EXTRA, *_XDIST, "-n", "2")
    result.assert_outcomes(passed=6, failed=1)
    metrics = json.loads((pytester.path / "reports/run-metrics.json").read_text())
    assert len(metrics["scenarios"]) == 7
    assert len(_history(pytester.path)) == 1
    monkeypatch.chdir(pytester.path)
    assert main(["gate"]) == 1


# ─── teardown ───────────────────────────────────────────────────────────────

TEARDOWN = """
import pytest


@pytest.fixture
def leaky():
    yield
    raise RuntimeError("device handle not released")


def test_uses_leaky(leaky):
    pass
"""


@pytest.mark.parametrize("reruns", ["0", "1"])
def test_teardown_errors_are_errors_not_passes(pytester, reruns):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makepyfile(test_teardown=TEARDOWN)
    pytester.runpytest_subprocess(*_EXTRA, "--ff-reruns", reruns)
    metrics = json.loads((pytester.path / "reports/run-metrics.json").read_text())
    (result,) = metrics["scenarios"]
    assert result["outcome"] == "error" and not result["flaky"]


SESSION = """
import pytest
from pathlib import Path

COUNT = Path("setups")


@pytest.fixture(scope="session")
def device():
    COUNT.write_text(str(int(COUNT.read_text() if COUNT.exists() else "0") + 1))
    yield


def test_flaky_last(device):
    marker = Path("tried")
    if not marker.exists():
        marker.write_text("1")
        assert False, "link dropped"
"""


def test_reruns_keep_session_fixtures_up(pytester):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makepyfile(test_session=SESSION)
    result = pytester.runpytest_subprocess(*_EXTRA, "--ff-reruns", "1")
    result.assert_outcomes(passed=1)
    assert (pytester.path / "setups").read_text() == "1"


# ─── numerics ───────────────────────────────────────────────────────────────


def test_nan_logits_fail_the_tolerance_check():
    reference = ReferenceBackend().logits("p", 4)
    candidate = [list(row) for row in reference]
    candidate[1][3] = math.nan
    candidate[2][7] = math.inf
    rule = ToleranceRule(
        rtol=1,
        atol=1,
        max_mismatch_fraction=0.5,
        min_cosine=0.0,
        min_top1_agreement=0.0,
        min_topk_overlap=0.0,
        max_perplexity_delta_pct=100,
    )
    result = compare_logits(reference, candidate, rule)
    assert not result.passed
    assert result.stats.non_finite == 2
    assert result.violated("non_finite") and result.violated("perplexity_delta_pct")


def test_misspelled_limits_are_rejected(tmp_path):
    with pytest.raises(pydantic.ValidationError):
        GatePolicy.model_validate({"max_flakey": 0})
    policy = tmp_path / "t.toml"
    policy.write_text('[policy]\nversion = "x"\n[rules."*/int8"]\natoll = 0.1\n')
    with pytest.raises(pydantic.ValidationError):
        TolerancePolicy.load(policy).rule_for("m", "int8")


def test_different_lengths_count_as_divergence():
    runs = [Generation([1, 2, 3], 1, 2), Generation([1, 2], 1, 2)]
    assert first_divergence(runs) == (1, 2)


def test_a_tie_between_cards_blames_no_single_card():
    runs = {0: Generation([1], 1, 1), 1: Generation([2], 1, 1)}
    assert odd_cards_out(runs) == [0, 1]
    runs[2] = Generation([1], 1, 1)
    assert odd_cards_out(runs) == [1]


# ─── history scope & paths ──────────────────────────────────────────────────


def test_a_narrow_green_run_is_not_the_baseline_for_the_full_suite(pytester, monkeypatch):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makepyfile(test_a="def test_a():\n    pass\n")
    pytester.makepyfile(test_b="import os\n\ndef test_b():\n    assert not os.getenv('BREAK')\n")
    pytester.runpytest_subprocess(*_EXTRA).assert_outcomes(passed=2)  # full, green
    pytester.runpytest_subprocess(*_EXTRA, "test_a.py").assert_outcomes(passed=1)  # narrow
    monkeypatch.setenv("BREAK", "1")
    pytester.runpytest_subprocess(*_EXTRA).assert_outcomes(passed=1, failed=1)
    monkeypatch.chdir(pytester.path)
    assert main(["gate", "--policy", "commit"]) == 1
    gate = json.loads((pytester.path / "reports/gate-commit.json").read_text())
    new = next(c for c in gate["checks"] if c["name"] == "new failures vs last good")
    assert new["measured"] == "1" and "test_b" in new["detail"]


def test_cli_finds_history_from_a_subfolder(pytester, capsys, monkeypatch):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makepyfile(test_a="def test_a():\n    pass\n")
    pytester.runpytest_subprocess(*_EXTRA).assert_outcomes(passed=1)
    sub = pytester.path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    assert main(["history"]) == 0
    assert "100.0%" in capsys.readouterr().out
