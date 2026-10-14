"""The project's own features/ and steps/ folders."""

import json
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from featureforge.cli import main
from featureforge.collect import feature_paths, is_selected, step_modules
from featureforge.core.settings import Settings

_BDD_INSTALLED = any(ep.name == "pytest-bdd" for ep in entry_points(group="pytest11"))
_EXTRA = () if _BDD_INSTALLED else ("-p", "pytest_bdd.plugin")

PYPROJECT = """
[tool.pytest.ini_options]
addopts = "-p featureforge.plugin"

[tool.featureforge]
packs = []
reports_dir = "reports"
"""

COLLECT = "from pytest_bdd import scenarios\nfrom featureforge.collect import feature_paths\n"
COLLECT += "scenarios(*feature_paths())\n"

BOARD_FEATURE = """Feature: Board inventory
  Scenario: A board reports its card count
    Given a board with 4 cards
    Then the board should report 4 cards
"""

FIRMWARE_FEATURE = """Feature: Firmware version
  Scenario: Firmware reports a version
    Given a board with 1 cards
    Then the firmware version should be "1.4.2"
"""

HELPERS = "class Board:\n    def __init__(self, cards):\n        self.cards = cards\n"
HELPERS += "        self.firmware = '1.4.2'\n"

BOARD_STEPS = """
from pytest_bdd import given, parsers, then
from steps.helpers import Board


@given(parsers.parse("a board with {cards:d} cards"), target_fixture="board")
def board(cards):
    return Board(cards)


@then(parsers.parse("the board should report {cards:d} cards"))
def board_reports(board, cards):
    assert board.cards == cards
"""

FIRMWARE_STEPS = """
from pytest_bdd import parsers, then


@then(parsers.parse('the firmware version should be "{version}"'))
def firmware_version(board, version):
    assert board.firmware == version
"""


@pytest.fixture
def project(pytester):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makepyfile(test_all=COLLECT)
    for folder in ("features/board", "features/firmware", "steps/firmware"):
        (pytester.path / folder).mkdir(parents=True)
    (pytester.path / "features/board/Inventory.feature").write_text(BOARD_FEATURE)
    (pytester.path / "features/firmware/Version.feature").write_text(FIRMWARE_FEATURE)
    (pytester.path / "steps/helpers.py").write_text(HELPERS)
    (pytester.path / "steps/board_steps.py").write_text(BOARD_STEPS)
    (pytester.path / "steps/firmware/version_steps.py").write_text(FIRMWARE_STEPS)
    return pytester


def test_project_features_and_nested_steps_load_without_registration(project):
    result = project.runpytest_subprocess(*_EXTRA)
    result.assert_outcomes(passed=2)
    metrics = json.loads((project.path / "reports/run-metrics.json").read_text())
    assert {s["feature"] for s in metrics["scenarios"]} == {"Board inventory", "Firmware version"}


def test_ff_feature_selects_a_folder(project):
    result = project.runpytest_subprocess(*_EXTRA, "--ff-feature", "features/firmware")
    result.assert_outcomes(passed=1, deselected=1)


def test_ff_feature_selects_a_single_file(project):
    result = project.runpytest_subprocess(
        *_EXTRA, "--ff-feature", "features/board/Inventory.feature"
    )
    result.assert_outcomes(passed=1, deselected=1)


def test_step_modules_walk_subfolders_and_skip_init(tmp_path):
    steps = tmp_path / "steps"
    (steps / "compiler").mkdir(parents=True)
    for name in ("__init__.py", "a_steps.py", "compiler/__init__.py", "compiler/b_steps.py"):
        (steps / name).write_text("")
    assert step_modules(steps) == ["steps.a_steps", "steps.compiler.b_steps"]
    assert step_modules(tmp_path / "missing") == []


def test_is_selected_matches_files_and_folders(tmp_path):
    feature = tmp_path / "features" / "compiler" / "Lowering.feature"
    assert is_selected(feature, [tmp_path / "features"])
    assert is_selected(feature, [feature])
    assert not is_selected(feature, [tmp_path / "features" / "runtime"])


def test_feature_paths_add_project_folder_only_when_it_has_features(tmp_path):
    settings = Settings(packs=["ai"], features_dir=tmp_path / "features")
    assert len(feature_paths(settings)) == 1
    (tmp_path / "features").mkdir()
    assert len(feature_paths(settings)) == 1
    (tmp_path / "features" / "X.feature").write_text("Feature: X\n")
    paths = feature_paths(settings)
    assert len(paths) == 2 and Path(paths[1]) == (tmp_path / "features").resolve()


def test_settings_resolve_project_folders_next_to_pyproject(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.featureforge]\nfeatures_dir = "qual/features"\n')
    settings = Settings.load(pyproject, environ={})
    assert settings.features_dir == (tmp_path / "qual/features").resolve()
    assert settings.steps_dir == (tmp_path / "steps").resolve()


def test_cli_run_rejects_a_missing_feature_path(capsys):
    assert main(["run", "features/does-not-exist"]) == 4
    assert "No such feature file or folder" in capsys.readouterr().err
