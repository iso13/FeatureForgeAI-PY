"""The FeatureForge pytest plugin — the runtime shell.

TS equivalents: ``src/core/support/hooks.ts`` (Before/BeforeStep/AfterStep/After/
AfterAll), ``world.ts`` (setWorldConstructor) and ``cucumber.mjs``.

Cucumber hook        → pytest / pytest-bdd hook used here
Before               → pytest_bdd_before_scenario
BeforeStep           → pytest_bdd_before_step
AfterStep            → pytest_bdd_after_step
AfterStep (failed)   → pytest_bdd_step_error  (Step Intelligence Layer)
After                → pytest_bdd_after_scenario + pytest_runtest_makereport
AfterAll             → pytest_sessionfinish
World                → the ``ctx`` fixture (ScenarioContext)
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

from featureforge.core.context import ScenarioContext
from featureforge.core.intelligence import (
    classify_failure,
    generate_developer_brief,
    reconcile_with_layers,
    render_console,
    render_html,
    snapshot_page,
    validate_all_layers,
    write_brief,
)
from featureforge.core.intelligence.models import utc_now
from featureforge.core.llm import LLMClient, build_llm
from featureforge.core.metrics import RunMetrics, ScenarioResult
from featureforge.core.settings import Settings, find_pyproject
from featureforge.core.telemetry import init_telemetry, shutdown_telemetry, start_span
from featureforge.packs import get_pack

SETTINGS_KEY = pytest.StashKey[Settings]()
METRICS_KEY = pytest.StashKey[RunMetrics]()
SCENARIO_KEY = pytest.StashKey[dict[str, Any]]()
CTX_KEY = pytest.StashKey[ScenarioContext]()

CORE_PLUGINS = ("featureforge.core.browser", "featureforge.core.steps")


def normalize_tag(tag: str) -> str:
    """Gherkin tag → pytest marker name (``@no-browser`` → ``no_browser``)."""
    return tag.lstrip("@").replace("-", "_")


# ─── options & configuration ────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("featureforge")
    group.addoption("--ff-env", help="Target environment key (overrides ENV)")
    group.addoption("--ff-headed", action="store_true", help="Run the browser headed")
    group.addoption("--ff-pack", action="append", help="Enable a pack (repeatable)")
    group.addoption("--ff-reports-dir", help="Where reports, briefs and metrics go")


def pytest_configure(config: pytest.Config) -> None:
    overrides: dict[str, Any] = {
        "env": config.getoption("--ff-env"),
        "packs": config.getoption("--ff-pack"),
        "reports_dir": config.getoption("--ff-reports-dir"),
    }
    if config.getoption("--ff-headed"):
        overrides["headless"] = False
    settings = Settings.load(find_pyproject(config.rootpath), overrides=overrides)
    if not settings.reports_dir.is_absolute():
        settings = settings.model_copy(
            update={"reports_dir": (config.rootpath / settings.reports_dir).resolve()}
        )
    config.stash[SETTINGS_KEY] = settings
    config.stash[METRICS_KEY] = RunMetrics(environment=settings.env)
    init_telemetry(settings)

    for module in CORE_PLUGINS:
        config.pluginmanager.import_plugin(module)
    for name in settings.packs:
        config.pluginmanager.import_plugin(get_pack(name).plugin)


def pytest_bdd_apply_tag(tag: str, function: Any) -> bool:
    getattr(pytest.mark, normalize_tag(tag))(function)
    return True


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ff_settings(pytestconfig: pytest.Config) -> Settings:
    return pytestconfig.stash[SETTINGS_KEY]


@pytest.fixture(scope="session")
def ff_llm(ff_settings: Settings) -> LLMClient:
    return build_llm(ff_settings)


@pytest.fixture
def ctx(request: pytest.FixtureRequest, ff_settings: Settings) -> Iterator[ScenarioContext]:
    """Per-scenario context — the Python counterpart of the Cucumber World."""
    callspec = getattr(request.node, "callspec", None)
    context = ScenarioContext(settings=ff_settings, variant=callspec.id if callspec else "")
    request.node.stash[CTX_KEY] = context
    yield context


# ─── scenario & step hooks ──────────────────────────────────────────────────


def _feature_name(feature: Any) -> str:
    if getattr(feature, "name", ""):
        return str(feature.name)
    stem = Path(str(getattr(feature, "filename", "Unknown"))).stem
    return " ".join(part.capitalize() for part in stem.replace("-", "_").split("_")) or "Unknown"


def pytest_bdd_before_scenario(request: pytest.FixtureRequest, feature: Any, scenario: Any) -> None:
    context: ScenarioContext = request.getfixturevalue("ctx")
    context.feature_name = _feature_name(feature)
    context.scenario_name = str(scenario.name) or "Unknown Scenario"
    context.tags = frozenset(set(feature.tags) | set(scenario.tags))
    request.node.stash[SCENARIO_KEY] = {
        "feature": context.feature_name,
        "scenario": context.scenario_name,
        "tags": sorted(context.tags),
        "started": time.perf_counter(),
    }
    context.scenario_span = start_span(
        f"Feature: {context.feature_name} - Scenario: {context.scenario_name}",
        {
            "bdd.feature": context.feature_name,
            "bdd.scenario": context.scenario_name,
            "bdd.tags": ",".join(sorted(context.tags)),
            "test.framework": "pytest-bdd",
        },
    )


def pytest_bdd_before_step(
    request: pytest.FixtureRequest, feature: Any, scenario: Any, step: Any, step_func: Any
) -> None:
    context: ScenarioContext = request.getfixturevalue("ctx")
    context.current_step = f"{step.keyword} {step.name}".strip()
    context.step_index += 1
    if context.network is not None:
        context.network.mark()
    context.step_span = start_span(
        f"{context.feature_name} - {step.name}",
        {"bdd.step": step.name, "bdd.scenario": context.scenario_name},
    )
    if context.settings.verbose:
        print(f"  → {context.current_step}")


def pytest_bdd_after_step(
    request: pytest.FixtureRequest,
    feature: Any,
    scenario: Any,
    step: Any,
    step_func: Any,
    step_func_args: dict[str, Any],
) -> None:
    context: ScenarioContext = request.getfixturevalue("ctx")
    if context.step_span is not None:
        context.step_span.end()
        context.step_span = None


def pytest_bdd_step_error(
    request: pytest.FixtureRequest,
    feature: Any,
    scenario: Any,
    step: Any,
    step_func: Any,
    step_func_args: dict[str, Any],
    exception: BaseException,
) -> None:
    """Step Intelligence Layer: classify → validate layers → Developer Brief."""
    context: ScenarioContext = request.getfixturevalue("ctx")
    if context.step_span is not None:
        context.step_span.record_exception(exception)
        context.step_span.end()
        context.step_span = None

    step_text = context.current_step or str(step.name)
    error_message = f"{type(exception).__name__}: {exception}".strip()
    stack = "".join(traceback.format_exception(exception))
    try:
        llm: LLMClient = request.getfixturevalue("ff_llm")
        requests_seen = context.network.all() if context.network else []
        classification = classify_failure(
            step_text, context.scenario_name, context.feature_name, error_message, stack, llm
        )
        report = validate_all_layers(
            step_text, error_message, snapshot=snapshot_page(context.page), requests=requests_seen
        )
        classification = reconcile_with_layers(classification, report)
        brief = generate_developer_brief(classification, report, llm)
        write_brief(brief, context.settings.reports_dir / "intelligence")
        context.brief = brief
        context.attach(render_html(brief), "text/html", "developer_brief")
    except Exception as err:  # the analysis must never mask the real failure
        print(f"Step Intelligence Layer error: {err!r}")

    if context.page is not None:
        try:
            context.attach(context.page.screenshot(full_page=True), "image/png", "failure")
        except Exception as err:
            print(f"Failed to capture screenshot: {err!r}")


def pytest_bdd_step_func_lookup_error(
    request: pytest.FixtureRequest, feature: Any, scenario: Any, step: Any, exception: Exception
) -> None:
    info = request.node.stash.get(SCENARIO_KEY, None)
    if info is not None:
        info["missing_step"] = f"{step.keyword} {step.name}".strip()


def pytest_bdd_after_scenario(request: pytest.FixtureRequest, feature: Any, scenario: Any) -> None:
    context: ScenarioContext = request.getfixturevalue("ctx")
    if context.scenario_span is not None:
        context.scenario_span.end()
        context.scenario_span = None


# ─── reporting & metrics ────────────────────────────────────────────────────


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    _record(item, report)
    return report


def _record(item: pytest.Item, report: pytest.TestReport) -> None:
    info = item.stash.get(SCENARIO_KEY, None)
    if info is None:
        return  # not a BDD scenario

    context = item.stash.get(CTX_KEY, None)
    if context is not None and context.brief is not None and report.failed:
        report.sections.append(("FeatureForge Developer Brief", render_console(context.brief)))
    if info.get("missing_step") and report.failed:
        report.sections.append(
            (
                "FeatureForge",
                f"No step definition for: {info['missing_step']}\n"
                "Generate a stub with: featureforge scaffold <feature file>",
            )
        )

    is_final = report.when == "call" or (report.when == "setup" and not report.passed)
    if not is_final:
        return
    if report.passed:
        result = "passed"
    elif report.skipped:
        result = "skipped"
    else:
        result = "failed" if report.when == "call" else "error"

    brief = context.brief if context is not None else None
    item.config.stash[METRICS_KEY].scenarios.append(
        ScenarioResult(
            nodeid=item.nodeid,
            feature=info["feature"],
            scenario=info["scenario"],
            example=context.variant if context is not None else "",
            tags=info["tags"],
            outcome=result,
            duration_s=round(time.perf_counter() - info["started"], 4),
            failed_step=brief.gherkin_step if brief else None,
            root_cause=brief.root_cause.value if brief else None,
            brief_id=brief.id if brief else None,
            missing_step=info.get("missing_step"),
        )
    )


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    metrics = config.stash.get(METRICS_KEY, None)
    if metrics is None or not metrics.scenarios:
        return
    terminalreporter.section("FeatureForge")
    for line in metrics.render().splitlines():
        terminalreporter.write_line(line)
    settings = config.stash[SETTINGS_KEY]
    terminalreporter.write_line(f"Metrics: {settings.reports_dir / 'run-metrics.json'}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    metrics = config.stash.get(METRICS_KEY, None)
    if metrics is not None and metrics.scenarios:
        metrics.finished_at = utc_now()
        metrics.write(config.stash[SETTINGS_KEY].reports_dir / "run-metrics.json")
    shutdown_telemetry()
