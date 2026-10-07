"""A small, dependency-free Gherkin reader for tooling (lint, scaffold).

Test execution uses pytest-bdd's full parser; this one only needs the structure
that tooling cares about — names, tags, steps, examples and line numbers — and
keeps the tools usable without importing pytest. Handles Background, Scenario,
Scenario Outline / Template, Examples, Rule, data tables, doc strings and
comments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

STEP_KEYWORDS = ("Given", "When", "Then", "And", "But", "*")
_PLACEHOLDER = re.compile(r"<([^<>]+)>")


@dataclass
class Step:
    keyword: str
    text: str
    line: int

    @property
    def placeholders(self) -> list[str]:
        return _PLACEHOLDER.findall(self.text)


@dataclass
class Examples:
    line: int
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class Scenario:
    kind: str  # "Background" | "Scenario" | "Scenario Outline"
    name: str
    line: int
    tags: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    examples: list[Examples] = field(default_factory=list)

    @property
    def is_outline(self) -> bool:
        return self.kind == "Scenario Outline"

    @property
    def example_count(self) -> int:
        return sum(len(block.rows) for block in self.examples)


@dataclass
class Feature:
    path: Path | None
    name: str = ""
    line: int = 0
    tags: list[str] = field(default_factory=list)
    background: Scenario | None = None
    scenarios: list[Scenario] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    raw: str = ""


_KEYWORD_LINE = re.compile(
    r"^(Feature|Rule|Background|Scenario Outline|Scenario Template|Scenario|Example|Examples|"
    r"Scenarios):\s*(.*)$"
)


def _split_cells(line: str) -> list[str]:
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") else inner
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", inner)]


def parse_feature(text: str, path: Path | None = None) -> Feature:
    feature = Feature(path=path, lines=text.splitlines(), raw=text)
    pending_tags: list[str] = []
    current: Scenario | None = None
    current_examples: Examples | None = None
    in_docstring: str | None = None

    for number, raw_line in enumerate(feature.lines, start=1):
        line = raw_line.strip()
        if in_docstring is not None:
            if line.startswith(in_docstring):
                in_docstring = None
            continue
        if line.startswith(('"""', "```")):
            in_docstring = line[:3]
            continue
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            pending_tags.extend(tag for tag in line.split("#")[0].split() if tag.startswith("@"))
            continue
        if line.startswith("|"):
            if current_examples is not None:
                cells = _split_cells(line)
                if not current_examples.header:
                    current_examples.header = cells
                else:
                    current_examples.rows.append(cells)
            continue  # step data tables are not needed by the tools

        match = _KEYWORD_LINE.match(line)
        if match:
            keyword, name = match.group(1), match.group(2).strip()
            tags, pending_tags = pending_tags, []
            if keyword == "Feature":
                feature.name, feature.line, feature.tags = name, number, tags
                current, current_examples = None, None
            elif keyword == "Rule":
                current, current_examples = None, None
            elif keyword == "Background":
                current = Scenario("Background", name, number, tags)
                feature.background, current_examples = current, None
            elif keyword in ("Scenario Outline", "Scenario Template"):
                current = Scenario("Scenario Outline", name, number, tags)
                feature.scenarios.append(current)
                current_examples = None
            elif keyword in ("Scenario", "Example"):
                current = Scenario("Scenario", name, number, tags)
                feature.scenarios.append(current)
                current_examples = None
            elif current is not None:  # Examples / Scenarios
                current_examples = Examples(line=number)
                current.examples.append(current_examples)
            continue

        keyword = next((k for k in STEP_KEYWORDS if line.startswith(f"{k} ")), None)
        if keyword is not None and current is not None:
            current.steps.append(Step(keyword, line[len(keyword) :].strip(), number))
            current_examples = None
        # anything else is free-form description text

    return feature


def read_feature(path: Path) -> Feature:
    return parse_feature(path.read_text(encoding="utf-8"), path)


def find_feature_files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("*.feature")))
        elif path.suffix == ".feature":
            found.append(path)
    return found


def effective_keyword(steps: list[Step], index: int) -> str:
    """Resolve And/But/* to the Given/When/Then they continue."""
    for step in reversed(steps[: index + 1]):
        if step.keyword in ("Given", "When", "Then"):
            return step.keyword
    return "Given"
