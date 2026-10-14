import json
from importlib.metadata import entry_points

import pytest

from featureforge.cli import main
from featureforge.core.history import RunHistory
from featureforge.core.manifest import BuildManifest
from featureforge.core.metrics import Measurement, RunMetrics, ScenarioResult
from featureforge.core.perf import Baseline, LatencySummary, PinnedBaselines, delta_pct, judge

_BDD_INSTALLED = any(ep.name == "pytest-bdd" for ep in entry_points(group="pytest11"))
_EXTRA = () if _BDD_INSTALLED else ("-p", "pytest_bdd.plugin")

# ─── manifest ───────────────────────────────────────────────────────────────


def test_manifest_layers_static_file_and_env(tmp_path):
    build = tmp_path / "manifest.json"
    build.write_text(json.dumps({"components": {"compiler": "0.9.2", "runtime": "3.1"}}))
    manifest = BuildManifest.collect(
        static={"components": {"firmware": "1.4.2", "compiler": "0.9.0"}},
        manifest_file=build,
        environ={"FF_COMPONENT_RUNTIME": "3.2", "FF_HARDWARE": "board-07"},
        cwd=tmp_path,
    )
    assert manifest.components == {"firmware": "1.4.2", "compiler": "0.9.2", "runtime": "3.2"}
    assert manifest.hardware == "board-07"


def test_changes_since_lists_only_what_differs():
    good = BuildManifest(components={"compiler": "0.9.1", "firmware": "1.4.2"}, hardware="b1")
    now = BuildManifest(components={"compiler": "0.9.2", "firmware": "1.4.2"}, hardware="b2")
    assert now.changes_since(good) == {"compiler": ("0.9.1", "0.9.2"), "board": ("b1", "b2")}


def test_fingerprint_ignores_the_board():
    a = BuildManifest(components={"compiler": "1"}, hardware="b1")
    assert a.fingerprint == a.model_copy(update={"hardware": "b2"}).fingerprint
    assert a.fingerprint != a.model_copy(update={"components": {"compiler": "2"}}).fingerprint


# ─── perf ───────────────────────────────────────────────────────────────────


def test_latency_summary_uses_nearest_rank():
    summary = LatencySummary.of([float(v) for v in range(1, 101)])
    assert (summary.p50, summary.p90, summary.p99) == (50.0, 90.0, 99.0)
    with pytest.raises(ValueError):
        LatencySummary.of([])


def test_delta_is_positive_when_worse_in_both_directions():
    assert delta_pct(110, 100, higher_is_better=False) == pytest.approx(10)
    assert delta_pct(90, 100, higher_is_better=True) == pytest.approx(10)
    assert delta_pct(90, 100, higher_is_better=False) == pytest.approx(-10)


def test_judge_uses_the_worse_baseline():
    m = Measurement(name="ttft_p50_ms", value=20.0, unit="ms")
    judged = judge(m, [Baseline(17.0, "pinned"), Baseline(19.5, "last-known-good")], 10)
    assert judged.baseline_source == "pinned" and judged.regressed
    assert judged.delta_pct == pytest.approx(17.647, abs=1e-3)


def test_pinned_baseline_stops_the_ratchet():
    """3% creep per run never trips a 10% budget against last-known-good alone."""
    pinned, value = 100.0, 100.0
    for _ in range(5):
        last_good = value
        value *= 1.03
        alone = judge(Measurement(name="t", value=value), [Baseline(last_good, "lkg")], 10)
        assert not alone.regressed
    both = judge(
        Measurement(name="t", value=value),
        [Baseline(pinned, "pinned"), Baseline(last_good, "lkg")],
        10,
    )
    assert both.regressed and both.baseline_source == "pinned"


def test_pinned_baselines_file(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('["m/int8"]\nttft_p50_ms = 16.4\n')
    pinned = PinnedBaselines.load(path)
    assert pinned.get("m/int8", "ttft_p50_ms") == 16.4
    assert pinned.get("m/bf16", "ttft_p50_ms") is None
    assert PinnedBaselines.load(tmp_path / "missing.toml").get("m/int8", "x") is None


# ─── history ────────────────────────────────────────────────────────────────


def _run(started, outcomes, compiler="1", measurements=None, flaky=()):
    return RunMetrics(
        started_at=started,
        manifest=BuildManifest(components={"compiler": compiler}, git_commit="abc"),
        scenarios=[
            ScenarioResult(
                nodeid=nodeid,
                feature="F",
                scenario=nodeid,
                outcome=outcome,
                flaky=nodeid in flaky,
                measurements=(measurements or {}).get(nodeid, []),
            )
            for nodeid, outcome in outcomes.items()
        ],
    )


def test_last_known_good_skips_failing_runs(tmp_path):
    history = RunHistory(tmp_path)
    history.archive(_run("2026-10-01T00:00:00", {"a": "passed"}, compiler="1"))
    history.archive(_run("2026-10-02T00:00:00", {"a": "failed"}, compiler="2"))
    good = history.last_known_good()
    assert good is not None and good.manifest.components["compiler"] == "1"
    assert history.newly_failing(_run("2026-10-03T00:00:00", {"a": "failed"})) == ["a"]


def test_history_keeps_only_the_newest_runs(tmp_path):
    history = RunHistory(tmp_path, keep=2)
    for day in (1, 2, 3):
        history.archive(_run(f"2026-10-0{day}T00:00:00", {"a": "passed"}))
    assert [r.started_at[:10] for r in history.runs()] == ["2026-10-02", "2026-10-03"]


def test_baseline_measurement_comes_from_a_green_run(tmp_path):
    history = RunHistory(tmp_path)
    good = [Measurement(name="ttft", value=10.0, key="m/int8")]
    bad = [Measurement(name="ttft", value=99.0, key="m/int8")]
    history.archive(_run("2026-10-01T00:00:00", {"a": "passed"}, measurements={"a": good}))
    history.archive(
        _run("2026-10-02T00:00:00", {"a": "passed", "b": "failed"}, measurements={"a": bad})
    )
    found = history.baseline_measurement("m/int8", "ttft")
    assert found is not None and found[0].value == 10.0
    assert history.baseline_measurement("m/bf16", "ttft") is None


def test_flips_on_the_same_build_are_flaky_but_across_builds_are_not(tmp_path):
    history = RunHistory(tmp_path)
    history.archive(_run("2026-10-01T00:00:00", {"a": "passed", "b": "passed"}, compiler="1"))
    history.archive(_run("2026-10-02T00:00:00", {"a": "failed", "b": "passed"}, compiler="1"))
    history.archive(_run("2026-10-03T00:00:00", {"a": "passed", "b": "failed"}, compiler="2"))
    history.archive(_run("2026-10-04T00:00:00", {"c": "passed"}, flaky={"c"}))
    flaky = {t.nodeid: t for t in history.flaky_tests()}
    assert set(flaky) == {"a", "c"}  # b flipped only when the compiler changed
    assert flaky["a"].flips == 1 and flaky["c"].passed_on_rerun == 1


def test_a_flip_between_boards_is_not_flaky(tmp_path):
    history = RunHistory(tmp_path)
    for day, board, outcome in ((1, "b1", "passed"), (2, "b2", "failed"), (3, "b1", "passed")):
        run = _run(f"2026-10-0{day}T00:00:00", {"a": outcome})
        run.manifest.hardware = board
        history.archive(run)
    assert history.flaky_tests() == []


def test_history_command_reports_runs_changes_and_flakes(tmp_path, capsys):
    history = RunHistory(tmp_path)
    history.archive(_run("2026-10-01T00:00:00", {"a": "passed"}, compiler="0.9.1"))
    history.archive(_run("2026-10-02T00:00:00", {"a": "failed"}, compiler="0.9.2"))
    history.archive(_run("2026-10-03T00:00:00", {"a": "passed"}, compiler="0.9.2"))
    history.archive(_run("2026-10-04T00:00:00", {"a": "failed"}, compiler="0.9.2"))
    assert main(["history", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Changed since last known-good (2026-10-03T00:00:00)" not in out  # no change vs 10-03
    assert "Flaky in the last 20 runs:" in out and "2 flip(s) on the same build" in out
    assert main(["history", "--dir", str(tmp_path / "none")]) == 2


# ─── end to end ─────────────────────────────────────────────────────────────

PYPROJECT = """
[tool.pytest.ini_options]
addopts = "-p featureforge.plugin"

[tool.featureforge]
packs = []
reports_dir = "reports"
history_dir = "history"

[tool.featureforge.manifest.components]
firmware = "1.4.2"
runtime = "3.1.0"
"""

HEALTHY = """Feature: int8 accuracy
  Scenario: int8 within policy
    Given the accelerator runs "toy-llm" in "int8"
    When I compare logits on the "smoke" prompt set
    Then every limit in the tolerance policy should hold
"""

DRIFTED = HEALTHY.replace('in "int8"\n', 'in "int8"\n    And a 2% scale drift is injected\n')

COLLECT = "from pytest_bdd import scenarios\nfrom featureforge.collect import feature_paths\n"
COLLECT += "scenarios(*feature_paths())\n"
COLLECT += "pytest_plugins = ['featureforge.packs.inference.steps']\n"


def test_numerical_break_after_a_compiler_bump_is_blamed_on_the_compiler(pytester, monkeypatch):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makepyfile(test_all=COLLECT)
    (pytester.path / "features").mkdir()
    feature = pytester.path / "features" / "Accuracy.feature"

    feature.write_text(HEALTHY)
    monkeypatch.setenv("FF_COMPONENT_COMPILER", "0.9.1")
    pytester.runpytest_subprocess(*_EXTRA).assert_outcomes(passed=1)

    feature.write_text(DRIFTED)
    monkeypatch.setenv("FF_COMPONENT_COMPILER", "0.9.2")
    result = pytester.runpytest_subprocess(*_EXTRA)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*Stack layer: COMPILER (80%, change)*",
            "*only compiler (0.9.1 → 0.9.2) changed since the last known-good run*",
            "*Changed since last known-good: compiler 0.9.1 → 0.9.2*",
        ]
    )
    runs = sorted((pytester.path / "history").glob("run-*.json"))
    assert len(runs) == 2
    latest = json.loads(runs[-1].read_text())
    assert latest["manifest"]["components"]["firmware"] == "1.4.2"
    assert latest["scenarios"][0]["stack_layer"] == "COMPILER"


SLOW = """Feature: int8 latency
  Scenario: int8 latency within budget
    Given the accelerator runs "toy-llm" in "int8"
    And a 15% slowdown is injected
    When I send 32 requests of 64 tokens from the "smoke" prompt set
    Then "ttft_p50_ms" should not regress more than 10% against the baseline
"""


def test_latency_regression_fails_and_is_recorded(pytester):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makepyfile(test_all=COLLECT)
    (pytester.path / "features").mkdir()
    (pytester.path / "features" / "Latency.feature").write_text(SLOW)
    result = pytester.runpytest_subprocess(*_EXTRA)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*PerformanceRegression: Performance regression: ttft_p50_ms*vs pinned*16.4ms*",
            "*Stack layer: PERFORMANCE*",
            "*Regression: ttft_p50_ms*budget 10%*",
        ]
    )
    metrics = json.loads((pytester.path / "reports/run-metrics.json").read_text())
    assert metrics["summary"]["perf_regressions"] == 1
