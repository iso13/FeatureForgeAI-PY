"""Step scaffolder. TS equivalent: ``src/generators/scaffolder`` (stepWriter, scaffolder).

Reads feature files, finds steps that no existing step definition matches, and
writes pytest-bdd stubs for them. Existing definitions are read statically with
``ast`` (``@given("…")``, ``@when(parsers.parse("…"))``, ``@then(parsers.re(r"…"))``),
so scaffolding never imports or runs the test code.
"""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass
from pathlib import Path

from featureforge.tools.gherkin import Feature, effective_keyword, find_feature_files, read_feature

_FIELD = re.compile(r"\{(\w*)(?::([^{}]*))?\}")
_QUOTED = re.compile(r'"([^"]*)"')
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


@dataclass(frozen=True)
class StepPattern:
    keyword: str  # given | when | then
    kind: str  # literal | parse | re
    pattern: str
    source: Path | None = None

    def matches(self, text: str) -> bool:
        if self.kind == "literal":
            return self.pattern == text
        if self.kind == "re":
            return re.fullmatch(self.pattern, text) is not None
        return re.fullmatch(parse_pattern_to_regex(self.pattern), text) is not None


def parse_pattern_to_regex(pattern: str) -> str:
    """Translate a ``parse``-style pattern into an equivalent regular expression."""
    out, position = [], 0
    for match in _FIELD.finditer(pattern):
        out.append(re.escape(pattern[position : match.start()]))
        fmt = match.group(2) or ""
        if fmt.endswith("d"):
            out.append(r"[-+]?\d+")
        elif fmt[-1:] in ("f", "g", "e", "n"):
            out.append(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
        elif fmt == "w":
            out.append(r"\w+")
        else:
            out.append(r".+?")
        position = match.end()
    out.append(re.escape(pattern[position:]))
    return "".join(out)


# ─── reading existing definitions ────────────────────────────────────────────


def _string_arg(node: ast.expr) -> tuple[str, str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "literal", node.value
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if name in ("parse", "cfparse"):
                return "parse", first.value
            if name == "re":
                return "re", first.value
    return None


def collect_step_patterns(paths: list[Path]) -> list[StepPattern]:
    patterns: list[StepPattern] = []
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    for file in files:
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not decorator.args:
                    continue
                func = decorator.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name not in ("given", "when", "then", "step"):
                    continue
                parsed = _string_arg(decorator.args[0])
                if parsed is None:
                    continue
                kinds = ("given", "when", "then") if name == "step" else (name,)
                for kind in kinds:
                    patterns.append(StepPattern(kind, parsed[0], parsed[1], file))
    return patterns


# ─── finding gaps ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MissingStep:
    keyword: str
    text: str  # as written in the feature (may contain <placeholders>)
    sample: str  # with the first Examples row substituted
    feature: Path | None
    line: int


def _substitute(text: str, header: list[str], row: list[str]) -> str:
    values = dict(zip(header, row, strict=False))
    return re.sub(r"<([^<>]+)>", lambda m: values.get(m.group(1), m.group(0)), text)


def missing_steps(features: list[Feature], patterns: list[StepPattern]) -> list[MissingStep]:
    found: dict[tuple[str, str], MissingStep] = {}
    for feature in features:
        blocks = [feature.background] if feature.background else []
        for scenario in [*blocks, *feature.scenarios]:
            header, row = [], []
            if scenario.examples and scenario.examples[0].rows:
                header, row = scenario.examples[0].header, scenario.examples[0].rows[0]
            for index, step in enumerate(scenario.steps):
                kind = effective_keyword(scenario.steps, index).lower()
                sample = _substitute(step.text, header, row)
                if any(p.keyword == kind and p.matches(sample) for p in patterns):
                    continue
                found.setdefault(
                    (kind, step.text),
                    MissingStep(kind, step.text, sample, feature.path, step.line),
                )
    return list(found.values())


# ─── writing stubs ───────────────────────────────────────────────────────────


def _identifier(text: str, used: set[str]) -> str:
    words = re.findall(r"[a-z0-9]+", re.sub(r"<[^<>]+>|\"[^\"]*\"", " ", text.lower()))
    base = "_".join(words[:8]) or "step"
    if base[0].isdigit() or keyword.iskeyword(base):
        base = f"step_{base}"
    name, counter = base, 2
    while name in used:
        name, counter = f"{base}_{counter}", counter + 1
    used.add(name)
    return name


def step_signature(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (parse pattern, [(param, type)]) for a step text."""
    params: list[tuple[str, str]] = []
    names: set[str] = set()

    def unique(candidate: str) -> str:
        candidate = re.sub(r"\W+", "_", candidate).strip("_").lower() or "value"
        if candidate[0].isdigit() or keyword.iskeyword(candidate):
            candidate = f"arg_{candidate}"
        name, counter = candidate, 2
        while name in names:
            name, counter = f"{candidate}_{counter}", counter + 1
        names.add(name)
        return name

    def quoted(match: re.Match[str]) -> str:
        inner = match.group(1)
        placeholder = re.fullmatch(r"<([^<>]+)>", inner)
        name = unique(placeholder.group(1) if placeholder else f"text{len(params) + 1}")
        params.append((name, "str"))
        return f'"{{{name}}}"'

    pattern = _QUOTED.sub(quoted, text)

    def bare_placeholder(match: re.Match[str]) -> str:
        name = unique(match.group(1))
        params.append((name, "str"))
        return f"{{{name}}}"

    pattern = re.sub(r"(?<!\{)<([^<>]+)>", bare_placeholder, pattern)

    pieces, position = [], 0
    for match in _NUMBER.finditer(pattern):
        if _inside_field(pattern, match.start()):
            continue
        name = unique(f"number{len(params) + 1}")
        params.append((name, "float"))
        pieces.append(pattern[position : match.start()] + f"{{{name}:g}}")
        position = match.end()
    pieces.append(pattern[position:])
    return "".join(pieces), params


def _inside_field(text: str, index: int) -> bool:
    return text.rfind("{", 0, index) > text.rfind("}", 0, index)


def render_stubs(missing: list[MissingStep], existing_names: set[str] | None = None) -> str:
    used = set(existing_names or ())
    keywords = sorted({m.keyword for m in missing}) or ["given", "then", "when"]
    needs_parsers = any(step_signature(m.text)[1] for m in missing)
    imports = ", ".join(sorted([*keywords, *(["parsers"] if needs_parsers else [])]))
    lines = [
        '"""Step definitions scaffolded by `featureforge scaffold`. Fill in the bodies."""',
        "",
        "from __future__ import annotations",
        "",
        f"from pytest_bdd import {imports}",
        "",
        "from featureforge.core.context import ScenarioContext",
        "",
    ]
    for step in missing:
        pattern, params = step_signature(step.text)
        decorator_arg = f"parsers.parse({pattern!r})" if params else repr(pattern)
        args = ", ".join(["ctx: ScenarioContext", *(f"{n}: {t}" for n, t in params)])
        where = f"{step.feature.name if step.feature else 'feature'}:{step.line}"
        lines += [
            "",
            f"@{step.keyword}({decorator_arg})",
            f"def {_identifier(step.text, used)}({args}) -> None:",
            f'    """Defined in {where}."""',
            f"    raise NotImplementedError({step.text!r})",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def scaffold(feature_paths: list[Path], steps_paths: list[Path]) -> tuple[list[MissingStep], str]:
    features = [read_feature(path) for path in find_feature_files(feature_paths)]
    patterns = collect_step_patterns([p for p in steps_paths if p.exists()])
    missing = missing_steps(features, patterns)
    return missing, render_stubs(missing)
