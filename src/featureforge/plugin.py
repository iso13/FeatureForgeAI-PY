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

import json
import os
import time
import traceback
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

from featureforge.collect import (
    is_selected,
    load_project_steps,
    scenario_feature_file,
    scenario_template,
)
from featureforge.core.context import ScenarioContext
from featureforge.core.history import RunHistory
from featureforge.core.intelligence import (
    StackAttribution,
    attribute_failure,
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
from featureforge.core.manifest import BuildManifest
from featureforge.core.metrics import RunMetrics, ScenarioResult
from featureforge.core.settings import Settings, find_pyproject
from featureforge.core.stack import StackError
from featureforge.core.telemetry import init_telemetry, shutdown_telemetry, start_span
from featureforge.core.tiers import Tier, parse_tiers, tier_from_tags
from featureforge.packs import get_pack

SETTINGS_KEY = pytest.StashKey[Settings]()
METRICS_KEY = pytest.StashKey[RunMetrics]()
SCENARIO_KEY = pytest.StashKey[dict[str, Any]]()
CTX_KEY = pytest.StashKey[ScenarioContext]()
TIER_KEY = pytest.StashKey[Tier]()
ATTEMPT_KEY = pytest.StashKey[int]()
FIRST_FAILURE_KEY = pytest.StashKey[StackAttribution]()
# Manifest component -> (last known-good version, this run's version); filled by run history.
CHANGES_KEY = pytest.StashKey[dict[str, tuple[str, str]]]()
HISTORY_KEY = pytest.StashKey[RunHistory]()

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
    group.addoption(
        "--ff-feature",
        action="append",
        default=[],
        help="Only run scenarios from this .feature file or folder (repeatable)",
    )
    group.addoption(
        "--ff-tier",
        action="append",
        default=[],
        help="Only run these tiers: unit, sim, single_card, multi_card, rack (comma/repeat)",
    )
    group.addoption("--ff-schedule", help="Run the tiers of a named schedule (commit, nightly …)")
    group.addoption(
        "--ff-reruns",
        type=int,
        default=0,
        help="Rerun a failed test up to N times; a pass on rerun is recorded as FLAKY",
    )
    group.addoption(
        "--ff-no-history", action="store_true", help="Don't add this run to the run history"
    )


def pytest_configure(config: pytest.Config) -> None:
    overrides: dict[str, Any] = {
        "env": config.getoption("--ff-env"),
        "packs": config.getoption("--ff-pack"),
        "reports_dir": config.getoption("--ff-reports-dir"),
    }
    if config.getoption("--ff-headed"):
        overrides["headless"] = False
    settings = Settings.load(find_pyproject(config.rootpath), overrides=overrides)
    if not settings.reports_dir.is_absolute():  # no pyproject.toml: anchor at the rootdir
        settings = settings.relative_to(config.rootpath)
    config.stash[SETTINGS_KEY] = settings
    manifest = BuildManifest.collect(
        settings.manifest, settings.manifest_file, os.environ, cwd=config.rootpath
    )
    history = RunHistory(settings.history_dir, keep=settings.history_keep)
    config.stash[HISTORY_KEY] = history
    scope = run_scope(config)
    good = history.last_known_good(scope=scope) or history.last_known_good()
    config.stash[CHANGES_KEY] = (
        manifest.changes_since(good.manifest) if good is not None and good.manifest else {}
    )
    config.stash[METRICS_KEY] = RunMetrics(
        environment=settings.env,
        manifest=manifest,
        schedule=config.getoption("--ff-schedule"),
        scope=scope,
    )
    init_telemetry(settings)

    for module in CORE_PLUGINS:
        config.pluginmanager.import_plugin(module)
    for name in settings.packs:
        config.pluginmanager.import_plugin(get_pack(name).plugin)
    load_project_steps(config.pluginmanager, settings.steps_dir)
    for tier in Tier:
        config.addinivalue_line("markers", f"{tier.marker}: test tier '{tier.value}'")


def run_scope(config: pytest.Config) -> str:
    """A stable description of which tests this run selected."""
    root = config.rootpath
    args = []
    for arg in config.args:
        path = Path(arg.split("::")[0])
        try:
            args.append(str((root / path).resolve().relative_to(root)) + arg[len(str(path)) :])
        except ValueError:
            args.append(arg)
    tiers = selected_tiers(config)
    features = sorted(
        str(Path(p).resolve().relative_to(root)) if Path(p).resolve().is_relative_to(root) else p
        for p in config.getoption("--ff-feature")
    )
    return json.dumps(
        {
            "args": sorted(args),
            "markers": config.getoption("markexpr", "") or "",
            "keywords": config.getoption("keyword", "") or "",
            "tiers": sorted(t.value for t in tiers) if tiers else "all",
            "features": features,
            "packs": sorted(config.stash[SETTINGS_KEY].packs),
        },
        sort_keys=True,
    )


def selected_tiers(config: pytest.Config) -> list[Tier] | None:
    """Tiers chosen with --ff-tier / --ff-schedule; None means every tier."""
    settings = config.stash[SETTINGS_KEY]
    tiers = parse_tiers(config.getoption("--ff-tier"))
    schedule = config.getoption("--ff-schedule")
    if schedule:
        tiers = list(dict.fromkeys([*tiers, *settings.tiers.schedule(schedule)]))
    return tiers or None


def item_tier(item: pytest.Item, default: Tier) -> Tier:
    tagged = tier_from_tags(marker.name for marker in item.iter_markers())
    if tagged is not None:
        return tagged
    return default if scenario_feature_file(item) is not None else Tier.UNIT


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Select by --ff-feature and tier; skip tiers this runner can't provide."""
    settings = config.stash[SETTINGS_KEY]
    features = [Path(p).resolve() for p in config.getoption("--ff-feature")]
    tiers = selected_tiers(config)
    config.stash[METRICS_KEY].tiers = [t.value for t in tiers] if tiers else []
    available = set(settings.tiers.available)
    keep: list[pytest.Item] = []
    dropped: list[pytest.Item] = []
    for item in items:
        tier = item_tier(item, settings.tiers.default)
        item.stash[TIER_KEY] = tier
        feature_file = scenario_feature_file(item)
        if features and (feature_file is None or not is_selected(feature_file, features)):
            dropped.append(item)
            continue
        if tiers is not None and tier not in tiers:
            dropped.append(item)
            continue
        if tier not in available:
            have = ", ".join(t.value for t in Tier if t in available) or "none"
            reason = f"needs tier '{tier.value}'; this runner provides: {have}"
            item.add_marker(pytest.mark.skip(reason=reason))
        keep.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep


def pytest_bdd_apply_tag(tag: str, function: Any) -> bool:
    getattr(pytest.mark, normalize_tag(tag))(function)
    return True


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ff_settings(pytestconfig: pytest.Config) -> Settings:
    return pytestconfig.stash[SETTINGS_KEY]


@pytest.fixture(scope="session")
def ff_history(pytestconfig: pytest.Config) -> RunHistory:
    return pytestconfig.stash[HISTORY_KEY]


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
    if context.page is None or isinstance(exception, StackError):
        # Stack attribution is for device/model scenarios; browser failures use the UI/API layers.
        try:
            context.stack = attribute_failure(
                exception,
                error_message,
                stack,
                changes=request.config.stash.get(CHANGES_KEY, {}),
            )
        except Exception as err:
            print(f"Stack attribution error: {err!r}")
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
        if context.stack is not None:
            brief = brief.model_copy(update={"stack": context.stack})
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


def _info_from_item(item: pytest.Item) -> dict[str, Any] | None:
    """Scenario details for a scenario that never started (e.g. skipped at setup)."""
    scenario = scenario_template(item)
    if scenario is None:
        return None
    tags = set(scenario.feature.tags) | set(scenario.tags)
    return {
        "feature": _feature_name(scenario.feature),
        "scenario": str(scenario.name),
        "tags": sorted(tags),
        "started": time.perf_counter(),
    }


def _plain_test_info(item: pytest.Item) -> dict[str, Any]:
    """A plain pytest test, recorded so unit-tier results reach metrics and gates."""
    return {
        "kind": "test",
        "feature": item.nodeid.split("::")[0],
        "scenario": item.name,
        "tags": sorted({m.name for m in item.iter_markers()}),
        "started": time.perf_counter(),
    }


def _record(item: pytest.Item, report: pytest.TestReport) -> None:
    if report.when == "teardown":
        if report.failed:
            _mark_error(item, "teardown failed")
        return
    info = item.stash.get(SCENARIO_KEY, None)
    if info is None and not report.passed:  # skipped or broken before the scenario started
        info = _info_from_item(item)
    if info is None:
        if scenario_feature_file(item) is not None:
            return  # a scenario still setting up; recorded once it reports
        info = item.stash.setdefault(SCENARIO_KEY, _plain_test_info(item))

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
    stack = context.stack if context is not None else None
    attempt = item.stash.get(ATTEMPT_KEY, 1)
    first_failure = item.stash.get(FIRST_FAILURE_KEY, None)
    flaky = result == "passed" and attempt > 1
    if flaky:
        stack = attribute_failure(None, passed_on_rerun=True, first_failure=first_failure)
    elif result in ("failed", "error") and stack is not None and first_failure is None:
        item.stash[FIRST_FAILURE_KEY] = stack
    if stack is not None and stack.layer.value == "UNKNOWN" and not stack.candidates:
        stack = None  # nothing learned; keep the record clean
    _upsert(
        item.config.stash[METRICS_KEY],
        ScenarioResult(
            nodeid=item.nodeid,
            kind=info.get("kind", "scenario"),
            feature=info["feature"],
            scenario=info["scenario"],
            example=context.variant if context is not None else _callspec_id(item),
            tags=info["tags"],
            outcome=result,
            tier=item.stash[TIER_KEY].value if TIER_KEY in item.stash else None,
            duration_s=round(time.perf_counter() - info["started"], 4),
            failed_step=brief.gherkin_step if brief else None,
            root_cause=brief.root_cause.value if brief else None,
            brief_id=brief.id if brief else None,
            missing_step=info.get("missing_step"),
            measurements=list(context.measurements) if context is not None else [],
            stack_layer=stack.layer.value if stack else None,
            stack_confidence=stack.confidence if stack else None,
            stack_method=stack.method if stack else None,
            stack_evidence=list(stack.evidence) if stack else [],
            attempts=attempt,
            flaky=flaky,
        ),
    )
    if stack is not None and report.failed and brief is None:  # the brief already shows it
        report.sections.append(("FeatureForge Stack Attribution", stack.render()))


def _mark_error(item: pytest.Item, reason: str) -> None:
    """A test that passed its call but broke afterwards (e.g. in teardown) is an error."""
    metrics = item.config.stash[METRICS_KEY]
    for index, existing in enumerate(metrics.scenarios):
        if existing.nodeid == item.nodeid:
            metrics.scenarios[index] = existing.model_copy(
                update={"outcome": "error", "flaky": False, "missing_step": None}
            )
            return
    info = _info_from_item(item) or _plain_test_info(item)
    metrics.scenarios.append(
        ScenarioResult(
            nodeid=item.nodeid,
            kind=info.get("kind", "scenario"),
            feature=info["feature"],
            scenario=info["scenario"],
            tags=info["tags"],
            outcome="error",
            tier=item.stash[TIER_KEY].value if TIER_KEY in item.stash else None,
            stack_evidence=[reason],
        )
    )


def _upsert(metrics: RunMetrics, result: ScenarioResult) -> None:
    """Add a result, replacing an earlier attempt of the same test (reruns)."""
    for index, existing in enumerate(metrics.scenarios):
        if existing.nodeid == result.nodeid:
            metrics.scenarios[index] = result
            return
    metrics.scenarios.append(result)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> bool | None:
    """``--ff-reruns N``: rerun failures; only the last attempt is reported."""
    reruns = int(item.config.getoption("--ff-reruns") or 0)
    if reruns <= 0:
        return None
    from _pytest.runner import runtestprotocol

    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    # Tear down only this test's own fixtures between attempts: module/session fixtures
    # (a device handle, a server) stay up, so a pass on rerun really is "same state".
    keep_parent: Any = item.parent
    for attempt in range(1, reruns + 2):
        item.stash[ATTEMPT_KEY] = attempt
        reports = runtestprotocol(item, nextitem=keep_parent, log=False)
        if any(r.failed for r in reports) and attempt <= reruns:
            init_request = getattr(item, "_initrequest", None)
            if init_request is not None:
                init_request()  # fresh function-scoped fixtures for the next attempt
            continue
        for report in reports:
            item.ihook.pytest_runtest_logreport(report=report)
        break
    try:  # now the broader teardown pytest would normally have done after this test
        item.session._setupstate.teardown_exact(nextitem)
    except Exception as err:
        report = pytest.TestReport(item.nodeid, item.location, {}, "failed", repr(err), "teardown")
        item.ihook.pytest_runtest_logreport(report=report)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def _callspec_id(item: pytest.Item) -> str:
    callspec = getattr(item, "callspec", None)
    return str(callspec.id) if callspec is not None else ""


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    metrics = config.stash.get(METRICS_KEY, None)
    if metrics is None or not metrics.scenarios:
        return
    terminalreporter.section("FeatureForge")
    for line in metrics.render().splitlines():
        terminalreporter.write_line(line)
    changes = config.stash.get(CHANGES_KEY, {})
    if changes:
        changed = ", ".join(f"{name} {old} → {new}" for name, (old, new) in changes.items())
        terminalreporter.write_line(f"Changed since last known-good: {changed}")
    settings = config.stash[SETTINGS_KEY]
    terminalreporter.write_line(f"Metrics: {settings.reports_dir / 'run-metrics.json'}")


WORKER_OUTPUT = "featureforge_metrics"


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    """pytest-xdist controller: collect each worker's results into this run's metrics."""
    payload = getattr(node, "workeroutput", {}).get(WORKER_OUTPUT)
    metrics = node.config.stash.get(METRICS_KEY, None)
    if not payload or metrics is None:
        return
    worker = RunMetrics.model_validate_json(payload)
    metrics.scenarios.extend(worker.scenarios)
    metrics.tiers = metrics.tiers or worker.tiers


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    metrics = config.stash.get(METRICS_KEY, None)
    workeroutput = getattr(config, "workeroutput", None)
    if workeroutput is not None:  # xdist worker: hand results to the controller
        if metrics is not None:
            workeroutput[WORKER_OUTPUT] = metrics.model_dump_json()
        shutdown_telemetry()
        return
    if metrics is not None and metrics.scenarios:
        metrics.finished_at = utc_now()
        metrics.write(config.stash[SETTINGS_KEY].reports_dir / "run-metrics.json")
        history = config.stash.get(HISTORY_KEY, None)
        if history is not None and not config.getoption("--ff-no-history"):
            history.archive(metrics)
    shutdown_telemetry()
