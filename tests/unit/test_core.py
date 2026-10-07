from dataclasses import dataclass

import pytest

from featureforge.cli import cucumber_tags_to_markers
from featureforge.core.context import ScenarioContext
from featureforge.core.llm import NullLLM, OpenAILLM, build_llm, parse_json_object
from featureforge.core.metrics import RunMetrics, ScenarioResult
from featureforge.core.settings import Settings
from featureforge.plugin import normalize_tag


def test_settings_layering(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.featureforge]\nenv = "dev"\nreports_dir = "out"\n'
        '[tool.featureforge.environments]\ndev = "https://dev.example/"\n'
        'prod = "https://example/"\n'
    )
    settings = Settings.load(pyproject, environ={"HEADLESS": "false"})
    assert settings.env == "dev"
    assert settings.base_url == "https://dev.example/"
    assert settings.headless is False
    assert settings.reports_dir == (tmp_path / "out").resolve()

    overridden = Settings.load(pyproject, environ={"ENV": "PROD"}, overrides={"env": None})
    assert overridden.base_url == "https://example/"


def test_unknown_environment_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown environment 'nope'"):
        _ = Settings(env="nope").base_url


def test_openai_key_enables_openai_like_the_ts_engine():
    settings = Settings.load(None, environ={"OPENAI_API_KEY": "sk-test"})
    assert settings.llm_provider == "openai"
    assert isinstance(build_llm(settings), OpenAILLM)
    explicit = Settings.load(None, environ={"OPENAI_API_KEY": "k", "FF_LLM_PROVIDER": "none"})
    assert isinstance(build_llm(explicit), NullLLM)


def test_api_key_is_not_in_repr():
    assert "sk-secret" not in repr(Settings(openai_api_key="sk-secret"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! {"a": 1} hope that helps', {"a": 1}),
        ("[1, 2]", None),
        ("nope", None),
    ],
)
def test_parse_json_object(text, expected):
    assert parse_json_object(text) == expected


@dataclass
class DemoState:
    value: int = 0


def test_context_typed_state_and_attachments(tmp_path):
    ctx = ScenarioContext(settings=Settings(reports_dir=tmp_path))
    ctx.feature_name, ctx.scenario_name = "My Feature", "Does a thing!"
    ctx.state(DemoState).value = 3
    assert ctx.state(DemoState).value == 3

    attachment = ctx.attach("hello", "text/plain", "greeting")
    assert attachment.path == tmp_path / "attachments/My_Feature__Does_a_thing/01_greeting.txt"
    assert attachment.path.read_text() == "hello"
    assert ctx.attach(b"\x89PNG", "image/png").path.suffix == ".png"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("not @wip and not @performanceTest", "not wip and not performanceTest"),
        ("@smoke or (@no-browser and not @demo)", "smoke or (no_browser and not demo)"),
    ],
)
def test_cucumber_tags_become_marker_expressions(expression, expected):
    assert cucumber_tags_to_markers(expression) == expected


def test_normalize_tag():
    assert normalize_tag("@no-browser") == "no_browser"
    assert normalize_tag("biasAudit") == "biasAudit"


def test_run_metrics_summary(tmp_path):
    metrics = RunMetrics(environment="qa")
    metrics.scenarios += [
        ScenarioResult(nodeid="a", feature="F", scenario="ok", outcome="passed", duration_s=0.2),
        ScenarioResult(
            nodeid="b", feature="F", scenario="bad", outcome="failed", root_cause="UI", duration_s=1
        ),
        ScenarioResult(
            nodeid="c", feature="F", scenario="gap", outcome="failed", missing_step="Given x"
        ),
        ScenarioResult(nodeid="d", feature="F", scenario="skip", outcome="skipped"),
    ]
    assert metrics.pass_rate == pytest.approx(1 / 3, abs=1e-4)
    assert metrics.failures_by_layer == {"UI": 1, "MISSING_STEP": 1}
    assert metrics.slowest(1)[0].scenario == "bad"

    path = metrics.write(tmp_path / "run-metrics.json")
    again = RunMetrics.read(path)
    assert again.counts == {"passed": 1, "failed": 2, "error": 0, "skipped": 1}
    assert "pass rate 33.3%" in again.render()
