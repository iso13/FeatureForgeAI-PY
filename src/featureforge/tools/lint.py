"""Gherkin lint. TS equivalent: ``npm run lint:gherkin`` (+ ``.gherkin-lintrc``).

Implements the rules from the TS ``.gherkin-lintrc`` that matter for review
quality; limits come from ``[tool.featureforge.lint]``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from featureforge.core.settings import LintSettings
from featureforge.tools.gherkin import Feature, Scenario, find_feature_files, read_feature

_Add = Callable[[int, str, str], None]


@dataclass(frozen=True)
class Issue:
    path: Path | None
    line: int
    rule: str
    message: str

    def render(self, root: Path | None = None) -> str:
        location = "<string>"
        if self.path is not None:
            location = str(self.path.relative_to(root) if root else self.path)
        return f"{location}:{self.line}: [{self.rule}] {self.message}"


def lint_feature(feature: Feature, rules: LintSettings | None = None) -> list[Issue]:
    cfg = rules or LintSettings()
    issues: list[Issue] = []

    def add(line: int, rule: str, message: str) -> None:
        issues.append(Issue(feature.path, line, rule, message))

    if not feature.raw.strip():
        add(1, "no-empty-file", "Feature file is empty")
        return issues

    # whitespace rules
    blank_run = 0
    for number, line in enumerate(feature.lines, start=1):
        if line != line.rstrip():
            add(number, "no-trailing-spaces", "Trailing whitespace")
        blank_run = blank_run + 1 if not line.strip() else 0
        if blank_run == 2:
            add(number, "no-multiple-empty-lines", "Multiple consecutive empty lines")
    if not feature.raw.endswith("\n"):
        add(len(feature.lines), "new-line-at-eof", "File should end with a newline")

    # feature-level rules
    if not feature.name:
        add(feature.line or 1, "no-unnamed-features", "Feature has no name")
    elif len(feature.name) > cfg.max_feature_name:
        add(feature.line, "name-length", f"Feature name longer than {cfg.max_feature_name}")
    _check_tags(feature.tags, feature.line, cfg, add)

    if not feature.scenarios:
        add(feature.line or 1, "no-files-without-scenarios", "Feature has no scenarios")

    count = sum(s.example_count if s.is_outline else 1 for s in feature.scenarios)
    if count > cfg.max_scenarios_per_file:
        add(
            feature.line,
            "max-scenarios-per-file",
            f"{count} scenarios (outline examples counted); limit is {cfg.max_scenarios_per_file}",
        )

    if feature.background is not None:
        background = feature.background
        if not background.steps:
            add(background.line, "no-empty-background", "Background has no steps")
        elif len(background.steps) > cfg.max_background_steps:
            add(
                background.line,
                "scenario-size",
                f"Background has {len(background.steps)} steps; limit is "
                f"{cfg.max_background_steps}",
            )
        if len(feature.scenarios) == 1:
            add(background.line, "no-background-only-scenario", "Background used by one scenario")
        _check_steps(background, cfg, add)

    names = Counter(s.name for s in feature.scenarios if s.name)
    for scenario in feature.scenarios:
        if not scenario.name:
            add(scenario.line, "no-unnamed-scenarios", "Scenario has no name")
        elif names[scenario.name] > 1:
            add(scenario.line, "no-dupe-scenario-names", f'Duplicate scenario "{scenario.name}"')
        elif len(scenario.name) > cfg.max_scenario_name:
            add(scenario.line, "name-length", f"Scenario name longer than {cfg.max_scenario_name}")
        if len(scenario.steps) > cfg.max_scenario_steps:
            add(
                scenario.line,
                "scenario-size",
                f"{len(scenario.steps)} steps; limit is {cfg.max_scenario_steps}",
            )
        _check_tags(scenario.tags, scenario.line, cfg, add)
        _check_steps(scenario, cfg, add)
        if scenario.is_outline:
            _check_outline(scenario, add)
    return issues


def _check_tags(tags: list[str], line: int, cfg: LintSettings, add: _Add) -> None:
    for tag, seen in Counter(tags).items():
        if seen > 1:
            add(line, "no-duplicate-tags", f"Duplicate tag {tag}")
        if tag in cfg.restricted_tags:
            add(line, "no-restricted-tags", f"Restricted tag {tag}")


def _check_steps(scenario: Scenario, cfg: LintSettings, add: _Add) -> None:
    previous = ""
    for step in scenario.steps:
        if len(step.text) > cfg.max_step_length:
            add(step.line, "name-length", f"Step longer than {cfg.max_step_length} characters")
        if step.keyword in ("Given", "When", "Then") and step.keyword == previous:
            add(step.line, "use-and", f'Use "And" instead of repeating "{step.keyword}"')
        if step.keyword in ("Given", "When", "Then"):
            previous = step.keyword


def _check_outline(scenario: Scenario, add: _Add) -> None:
    if not scenario.examples or scenario.example_count == 0:
        add(scenario.line, "no-scenario-outlines-without-examples", "Outline has no examples")
        return
    used = {name for step in scenario.steps for name in step.placeholders}
    columns = {name for block in scenario.examples for name in block.header}
    for name in sorted(used - columns):
        add(scenario.line, "no-missing-placeholder", f"<{name}> is not an Examples column")
    for name in sorted(columns - used):
        add(scenario.line, "no-unused-variables", f"Examples column '{name}' is never used")


def lint_paths(paths: list[Path], rules: LintSettings | None = None) -> list[Issue]:
    issues: list[Issue] = []
    for path in find_feature_files(paths):
        issues.extend(lint_feature(read_feature(path), rules))
    return issues
