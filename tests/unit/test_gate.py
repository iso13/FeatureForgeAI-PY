import json
from datetime import date

import pytest

from featureforge.cli import main
from featureforge.core.escapes import EscapeLog
from featureforge.core.gate import GatePolicy, Quarantine, evaluate
from featureforge.core.history import RunHistory
from featureforge.core.manifest import BuildManifest
from featureforge.core.metrics import Measurement, RunMetrics, ScenarioResult
from featureforge.core.settings import Settings
from featureforge.core.tiers import Tier

TODAY = date(2026, 10, 14)


def _result(name, outcome="passed", tier="sim", **extra):
    return ScenarioResult(
        nodeid=f"t::{name}", feature="F", scenario=name, outcome=outcome, tier=tier, **extra
    )


def _metrics(*results, components=None, started="2026-10-14T12:00:00"):
    return RunMetrics(
        started_at=started,
        manifest=BuildManifest(components=components or {}, git_commit="abc123"),
        scenarios=list(results),
    )


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


def test_all_green_passes():
    metrics = _metrics(_result("a", tier="unit"), _result("b"))
    result = evaluate("commit", GatePolicy(required_tiers=[Tier.UNIT, Tier.SIM]), metrics)
    assert result.passed and result.decision == "PASS"


def test_a_failure_blocks_and_is_named():
    metrics = _metrics(_result("a"), _result("cards agree", outcome="failed"))
    result = evaluate("commit", GatePolicy(), metrics)
    check = _check(result, "pass rate")
    assert not result.passed and check.measured == "50.0% of 2"
    assert "cards agree" in check.detail


def test_required_tier_that_was_skipped_blocks():
    metrics = _metrics(_result("a"), _result("rack", outcome="skipped", tier="rack"))
    result = evaluate("release", GatePolicy(required_tiers=[Tier.SIM, Tier.RACK]), metrics)
    rack = _check(result, "tier rack")
    assert not rack.passed and rack.measured == "0 ran, 1 skipped"
    assert _check(result, "tier sim").passed


def test_quarantine_waives_until_it_expires():
    failing = _result("Every card in the node", outcome="failed", tier="multi_card")
    metrics = _metrics(_result("a"), failing)
    waiver = Quarantine(match="Every card", ticket="HW-231", expires=date(2026, 11, 1))
    result = evaluate("nightly", GatePolicy(quarantine=[waiver]), metrics, today=TODAY)
    assert result.passed
    assert _check(result, "pass rate").measured == "100.0% of 1, 1 waived"

    late = evaluate("nightly", GatePolicy(quarantine=[waiver]), metrics, today=date(2026, 11, 2))
    assert not late.passed
    assert "HW-231 expired 2026-11-01" in _check(late, "quarantine").detail


def test_new_failures_are_measured_against_last_known_good(tmp_path):
    history = RunHistory(tmp_path)
    history.archive(_metrics(_result("a"), _result("b"), started="2026-10-13T12:00:00"))
    now = _metrics(_result("a", outcome="failed"), _result("b"))
    history.archive(now)
    result = evaluate("commit", GatePolicy(min_pass_rate=0.0), now, history)
    new = _check(result, "new failures vs last good")
    assert not new.passed and new.detail == "a"


def test_flaky_and_perf_regressions_count():
    slow = Measurement(
        name="ttft_p50_ms",
        value=19.0,
        baseline=16.4,
        delta_pct=15.9,
        budget_pct=10,
        regressed=True,
    )
    metrics = _metrics(_result("a", flaky=True, attempts=2), _result("b", measurements=[slow]))
    result = evaluate("release", GatePolicy(max_flaky=0, max_perf_regressions=0), metrics)
    assert not _check(result, "flaky (passed on rerun)").passed
    perf = _check(result, "perf regressions")
    assert not perf.passed and "ttft_p50_ms +15.9% in b" in perf.detail


def test_release_needs_every_component_versioned():
    policy = GatePolicy(required_components=["firmware", "compiler"])
    result = evaluate("release", policy, _metrics(_result("a"), components={"compiler": "0.9"}))
    check = _check(result, "build manifest")
    assert not check.passed and check.detail == "missing: firmware"


ESCAPES = """
[[escape]]
id = "ESC-1"
title = "drift"
found = 2026-09-12
layer = "COMPILER"
should_have_caught = "sim"
regression_test = "features/Drift.feature"

[[escape]]
id = "ESC-2"
title = "slow p99"
found = 2026-10-06
layer = "RUNTIME"
regression_test = ""
"""


def test_escapes_without_a_test_block_release(tmp_path):
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality/escapes.toml").write_text(ESCAPES)
    (tmp_path / "features").mkdir()
    (tmp_path / "features/Drift.feature").write_text("Feature: Drift\n")
    log = EscapeLog.load(tmp_path / "quality/escapes.toml", root=tmp_path)
    assert [e.id for e in log.without_test] == ["ESC-2"]
    assert log.by_layer() == {"COMPILER": 1, "RUNTIME": 1}
    assert log.by_tier() == {"sim": 1, "unassigned": 1}
    assert "with a regression test: 1 (50%)" in log.render()
    result = evaluate(
        "release", GatePolicy(max_escapes_without_test=0), _metrics(_result("a")), escapes=log
    )
    assert "ESC-2 (slow p99)" in _check(result, "escapes without a test").detail


def test_gate_policies_merge_defaults_with_pyproject(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.featureforge.gates.release]\nmax_flaky = 3\n"
        "[tool.featureforge.gates.smoke]\nrequired_tiers = ['sim']\n"
    )
    settings = Settings.load(pyproject, environ={})
    release = settings.gate_policy("release")
    assert release.max_flaky == 3 and release.max_escapes_without_test == 0  # default kept
    assert settings.gate_policy("smoke").required_tiers == [Tier.SIM]
    with pytest.raises(KeyError, match="Known: commit, nightly, release, smoke"):
        settings.gate_policy("weekly")


def test_gate_command_exit_codes_and_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.featureforge]\nreports_dir = "reports"\nhistory_dir = "history"\n'
        "[tool.featureforge.gates.commit]\nrequired_tiers = ['sim']\n"
    )
    assert main(["gate"]) == 2  # no run yet
    _metrics(_result("a")).write(tmp_path / "reports/run-metrics.json")
    assert main(["gate"]) == 0
    _metrics(_result("a", outcome="failed")).write(tmp_path / "reports/run-metrics.json")
    assert main(["gate"]) == 1
    report = json.loads((tmp_path / "reports/gate-commit.json").read_text())
    assert report["decision"] == "BLOCK"
    assert "Release gate 'commit': BLOCK" in capsys.readouterr().out
    assert main(["gate", "--policy", "weekly"]) == 2
