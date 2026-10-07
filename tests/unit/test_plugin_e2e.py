"""Run real pytest sessions through the plugin (pytester) and check the outputs."""

import json
import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest

# pytest-bdd registers itself through an entry point when installed. In a source
# checkout without that metadata, load it explicitly in the child sessions.
_BDD_INSTALLED = any(ep.name == "pytest-bdd" for ep in entry_points(group="pytest11"))
_EXTRA = () if _BDD_INSTALLED else ("-p", "pytest_bdd.plugin")

PYPROJECT = """
[tool.pytest.ini_options]
addopts = "-p featureforge.plugin -m 'not wip and not performanceTest and not demo'"
filterwarnings = ["ignore::pytest.PytestUnknownMarkWarning"]

[tool.featureforge]
packs = ["ai"]
reports_dir = "reports"
"""

PACK_TESTS = """
from pytest_bdd import scenarios
from featureforge.packs import get_pack
scenarios(str(get_pack("ai").features_dir))
"""


@pytest.fixture
def project(pytester):
    pytester.makefile(".toml", pyproject=PYPROJECT)
    pytester.makepyfile(test_packs=PACK_TESTS)
    return pytester


def test_default_run_passes_and_writes_metrics(project):
    result = project.runpytest_subprocess(*_EXTRA)
    result.assert_outcomes(passed=10, deselected=2)
    result.stdout.fnmatch_lines(["*FeatureForge*", "*pass rate 100.0%*"])
    metrics = json.loads((project.path / "reports/run-metrics.json").read_text())
    assert metrics["summary"]["counts"]["passed"] == 10
    outline = [s for s in metrics["scenarios"] if "parity" in s["scenario"]]
    assert len(outline) == 7 and all("no-browser" in s["tags"] for s in outline)
    attachments = list((project.path / "reports/attachments").rglob("*fairness_check.txt"))
    assert len(attachments) == 7


def test_failure_produces_developer_brief(project):
    result = project.runpytest_subprocess(*_EXTRA, "-m", "demo")
    result.assert_outcomes(failed=1, deselected=11)
    result.stdout.fnmatch_lines(
        [
            "*FeatureForge Developer Brief*",
            "*Root cause: BUSINESS_LOGIC (85% confidence, heuristic)*",
            "*Failures by root-cause layer: BUSINESS_LOGIC=1*",
        ]
    )
    briefs = [p for p in (project.path / "reports/intelligence").glob("brief-*.json")]
    assert len(briefs) == 1
    brief = json.loads(briefs[0].read_text())
    assert brief["gherkin_step"].startswith("Then the decision outcome")
    assert brief["layer_analysis"]["business_logic"]["passed"] is False
    html = list((project.path / "reports/attachments").rglob("*developer_brief.html"))
    assert html and "Developer Brief" in html[0].read_text()


def test_missing_step_points_to_scaffold(project):
    project.makefile(
        ".feature",
        gap="Feature: Gap\n  Scenario: Undefined\n    Given a step nobody wrote\n",
    )
    project.makepyfile(test_gap='from pytest_bdd import scenarios\nscenarios("gap.feature")\n')
    result = project.runpytest_subprocess(*_EXTRA, "test_gap.py")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*No step definition for: Given a step nobody wrote*"])
    metrics = json.loads((project.path / "reports/run-metrics.json").read_text())
    assert metrics["summary"]["failures_by_layer"] == {"MISSING_STEP": 1}


def test_cucumber_json_is_compatible_output(project):
    project.runpytest_subprocess(*_EXTRA, "--cucumberjson=reports/cucumber.json")
    report = json.loads((project.path / "reports/cucumber.json").read_text())
    names = {feature["name"] for feature in report}
    assert "Model fairness across demographic attributes" in names


def _browser_available():
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            playwright.chromium.launch().close()
    except Exception:
        return False
    return True


def _playwright_browsers_path():
    """Where Playwright keeps browsers for the *real* user.

    pytester points HOME at a temp dir for child sessions, so without this the
    child looks for browsers under the temp HOME and fails to launch.
    """
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if configured:
        return configured
    home = Path.home()
    if sys.platform == "darwin":
        return str(home / "Library/Caches/ms-playwright")
    if sys.platform == "win32":
        return str(Path(os.environ.get("LOCALAPPDATA", home)) / "ms-playwright")
    return str(home / ".cache/ms-playwright")


UI_PYPROJECT = (
    PYPROJECT.replace('packs = ["ai"]', "packs = []")
    + """
[tool.featureforge.environments]
qa = "http://policy.test/"
"""
)

UI_FEATURE = """Feature: Policy dashboard
  Scenario: Dashboard shows the policy summary
    Given the policy API is failing
    And I navigate to homepage
    Then I should see the policy summary
"""

UI_STEPS = '''
from playwright.sync_api import expect
from pytest_bdd import given, then

PAGE = """<title>Policy</title><body>Loading…<script>
fetch('/api/policy', {method: 'POST'}).then(r => {
  document.body.innerHTML = r.ok ? '<h1>Policy summary</h1>'
                                 : '<div role="alert">Could not load policy</div>';
});</script></body>"""


@given("the policy API is failing")
def api_failing(page):
    page.route(
        "http://policy.test/",
        lambda route: route.fulfill(body=PAGE, content_type="text/html"),
    )
    page.route("**/api/policy", lambda route: route.fulfill(status=500, json={"error": "db down"}))


@then("I should see the policy summary")
def see_summary(page):
    page.get_by_role("alert").wait_for(timeout=2000)
    expect(page.get_by_text("Policy summary")).to_be_visible(timeout=200)
'''


@pytest.mark.skipif(not _browser_available(), reason="no Playwright browser available")
def test_ui_failure_traces_symptom_to_api(pytester, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", _playwright_browsers_path())
    pytester.makefile(".toml", pyproject=UI_PYPROJECT)
    pytester.makefile(".feature", dashboard=UI_FEATURE)
    pytester.makeconftest(UI_STEPS)
    pytester.makepyfile(
        test_ui='from pytest_bdd import scenarios\nscenarios("dashboard.feature")\n'
    )
    result = pytester.runpytest_subprocess(*_EXTRA)
    result.assert_outcomes(failed=1)
    brief_path = next((pytester.path / "reports/intelligence").glob("brief-*.json"))
    brief = json.loads(brief_path.read_text())
    assert brief["root_cause"] == "API_CONTRACT"
    assert brief["classification_method"] == "heuristic+layers"
    assert "UI" in brief["failure_layers"]
    assert brief["layer_analysis"]["ui"]["passed"] is False
    assert "Could not load policy" in brief["layer_analysis"]["ui"]["evidence"]
    assert "POST http://policy.test/api/policy → 500" in brief["layer_analysis"]["api"]["evidence"]
    screenshots = list((pytester.path / "reports/attachments").rglob("*failure.png"))
    assert screenshots and screenshots[0].stat().st_size > 1000
