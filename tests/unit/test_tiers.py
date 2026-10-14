import json
from importlib.metadata import entry_points

import pytest

from featureforge.cli import main
from featureforge.core.settings import Settings
from featureforge.core.tiers import Tier, TierSettings, parse_tiers, tier_from_tags

_BDD_INSTALLED = any(ep.name == "pytest-bdd" for ep in entry_points(group="pytest11"))
_EXTRA = () if _BDD_INSTALLED else ("-p", "pytest_bdd.plugin")


def test_tiers_are_ordered_by_cost():
    assert [t.value for t in sorted(Tier, key=lambda t: t.rank)] == [
        "unit",
        "sim",
        "single_card",
        "multi_card",
        "rack",
    ]


def test_parse_tiers_accepts_commas_dashes_and_repeats():
    assert parse_tiers(["sim,single-card", "rack", "sim"]) == [
        Tier.SIM,
        Tier.SINGLE_CARD,
        Tier.RACK,
    ]
    with pytest.raises(ValueError, match="Unknown tier 'gpu'"):
        parse_tiers(["gpu"])


def test_tier_from_tags_picks_the_costliest():
    assert tier_from_tags(["@smoke", "@tier-sim"]) is Tier.SIM
    assert tier_from_tags(["tier_sim", "tier_rack"]) is Tier.RACK
    assert tier_from_tags(["@smoke"]) is None
    with pytest.raises(ValueError, match="Unknown tier tag"):
        tier_from_tags(["@tier-gpu"])


def test_available_tiers_come_from_pyproject_and_env(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.featureforge.tiers]\navailable = ["unit"]\n'
        '[tool.featureforge.tiers.schedules]\nsmoke = ["unit"]\n'
    )
    assert Settings.load(pyproject, environ={}).tiers.available == [Tier.UNIT]
    env = {"FF_TIERS_AVAILABLE": "unit,sim,single_card"}
    settings = Settings.load(pyproject, environ=env)
    assert settings.tiers.available == [Tier.UNIT, Tier.SIM, Tier.SINGLE_CARD]
    assert settings.tiers.schedule("smoke") == [Tier.UNIT]
    with pytest.raises(KeyError, match="Known: smoke"):
        settings.tiers.schedule("weekly")


def test_default_schedules_grow_with_cost():
    schedules = TierSettings().schedules
    assert set(schedules["commit"]) < set(schedules["nightly"]) < set(schedules["release"])


PYPROJECT = """
[tool.pytest.ini_options]
addopts = "-p featureforge.plugin"

[tool.featureforge]
packs = []
reports_dir = "reports"

[tool.featureforge.tiers]
available = ["unit", "sim"]
"""

FEATURE = """Feature: Tiered
  Scenario: Untagged runs in the default tier
    Given a step

  @tier-single_card
  Scenario: Needs one card
    Given a step

  @tier-rack
  Scenario: Needs a rack
    Given a step
"""

TESTS = """
import pytest
from pytest_bdd import given, scenarios

scenarios("tiered.feature")


@given("a step")
def a_step():
    pass


def test_plain_unit():
    pass


@pytest.mark.tier_multi_card
def test_plain_multi_card():
    pass
"""


@pytest.fixture
def project(pytester):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makefile(".feature", tiered=FEATURE)
    pytester.makepyfile(test_tiered=TESTS)
    return pytester


def test_unavailable_tiers_skip_with_the_reason(project):
    result = project.runpytest_subprocess(*_EXTRA, "-rs")
    result.assert_outcomes(passed=2, skipped=3)
    result.stdout.fnmatch_lines(["*needs tier 'rack'; this runner provides: unit, sim*"])
    metrics = json.loads((project.path / "reports/run-metrics.json").read_text())
    tiers = {s["scenario"]: (s["tier"], s["outcome"]) for s in metrics["scenarios"]}
    assert tiers == {
        "Untagged runs in the default tier": ("sim", "passed"),
        "Needs one card": ("single_card", "skipped"),
        "Needs a rack": ("rack", "skipped"),
    }
    assert metrics["summary"]["by_tier"]["rack"] == {"skipped": 1}


def test_ff_tier_selects_plain_tests_and_scenarios(project):
    result = project.runpytest_subprocess(*_EXTRA, "--ff-tier", "unit")
    result.assert_outcomes(passed=1, deselected=4)


def test_schedule_runs_its_tiers_where_hardware_exists(project, monkeypatch):
    monkeypatch.setenv("FF_TIERS_AVAILABLE", "unit,sim,single_card,multi_card")
    result = project.runpytest_subprocess(*_EXTRA, "--ff-schedule", "nightly")
    result.assert_outcomes(passed=4, deselected=1)  # rack is release-only


def test_plan_counts_scenarios_per_tier(project, capsys, monkeypatch):
    monkeypatch.chdir(project.path)
    (project.path / "features").mkdir()
    (project.path / "features/tiered.feature").write_text(FEATURE)
    assert main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "commit: 1 scenario(s)" in out
    assert "release: 3 scenario(s), 2 skipped on this runner" in out
