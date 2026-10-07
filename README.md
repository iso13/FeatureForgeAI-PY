# FeatureForge AI — Python engine

AI-powered BDD quality engineering on **pytest + pytest-bdd + Playwright**.
This is the Python counterpart of the TypeScript FeatureForgeAI shell: the same
`.feature` files, tags, Cucumber JSON output and Developer Brief format, with a
Python-native runtime. See [MAPPING.md](MAPPING.md) for the file-by-file mapping.

## What it does

- **Runs Gherkin scenarios on pytest.** Packs ship features and step definitions;
  any pytest tooling (xdist, markers, fixtures, JUnit, CI widgets) works as is.
- **Explains failures (Step Intelligence Layer).** When a step fails, the plugin
  classifies the failure, checks the UI, API and business-logic layers, traces
  symptoms to their cause, and writes a Developer Brief (console, HTML, JSON).
  An LLM (OpenAI or local Ollama) refines ambiguous cases; without one, the
  heuristics still run offline.
- **Measures quality per run.** `reports/run-metrics.json` records outcomes,
  pass rate, durations and failures by root-cause layer for trend tracking.
- **Keeps state typed.** Each pack owns a dataclass for its scenario state
  (`ctx.state(FairnessState)`) instead of one shared, untyped World object.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python -m playwright install chromium   # only needed for UI scenarios

pytest                                  # unit tests + BDD scenarios
featureforge run                        # BDD scenarios with Cucumber JSON + JUnit
featureforge run --tags "@demo"         # see a Developer Brief for a failing step
featureforge metrics                    # summary of the last run
```

## Layout

```
src/featureforge/
  plugin.py              pytest plugin: hooks, ctx fixture, briefs, metrics
  cli.py                 featureforge run | lint | scaffold | metrics | packs
  core/
    settings.py          typed config (pyproject → env vars → CLI)
    context.py           ScenarioContext (the World) + attachments
    browser.py           lazy Playwright fixtures (browser/context/page)
    network.py           per-scenario network capture
    telemetry.py         optional OpenTelemetry spans
    llm.py               LLMClient: OpenAI | Ollama | none
    metrics.py           run metrics
    pages/               page objects
    intelligence/        classifier, layer validator, developer brief
  packs/
    ai/                  fairness + RAG pack (features, steps, domain models)
  tools/                 Gherkin reader, linter, step scaffolder
tests/
  bdd/test_packs.py      collects every pack scenario
  unit/                  unit tests + end-to-end plugin tests (pytester)
```

## Configuration

`[tool.featureforge]` in `pyproject.toml`, overridden by environment variables:

| Setting | Env var | Default |
|---|---|---|
| `env` | `ENV` | `qa` |
| `headless` | `HEADLESS` | `true` (or `--ff-headed`) |
| `enable_tracing` | `ENABLE_TRACING` | `false` |
| `otlp_endpoint` | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `http://localhost:4318/v1/traces` |
| `llm_provider` | `FF_LLM_PROVIDER` | `none` (`openai` if `OPENAI_API_KEY` is set) |
| `llm_model` | `FF_LLM_MODEL` | `gpt-4o` / `llama3.1` |
| `packs` | `FF_PACKS` | `["ai"]` |
| `reports_dir` | `FF_REPORTS_DIR` | `reports` |

## Writing a pack

```python
# my_pack/__init__.py
from pathlib import Path
from featureforge.packs import Pack

PACK = Pack("billing", "Billing scenarios", Path(__file__).parent / "features", "my_pack.steps")
```

```toml
# my_pack's pyproject.toml
[project.entry-points."featureforge.packs"]
billing = "my_pack:PACK"
```

Enable it with `packs = ["ai", "billing"]` and collect it with
`scenarios(str(get_pack("billing").features_dir))`. Missing steps can be
generated with `featureforge scaffold my_pack/features --out my_pack/steps.py`.

## License

Business Source License 1.1 — see [LICENSE.txt](LICENSE.txt).
