# FeatureForge AI — Python engine

AI-powered BDD quality engineering on **pytest + pytest-bdd + Playwright**.
This is the Python counterpart of the TypeScript FeatureForgeAI shell: the same
`.feature` files, tags, Cucumber JSON output and Developer Brief format, with a
Python-native runtime. See [MAPPING.md](MAPPING.md) for the file-by-file mapping
and [docs/QUALITY_MODEL.md](docs/QUALITY_MODEL.md) for how tiers, tolerances,
attribution, history and gates fit together.

## What it does

- **Runs Gherkin scenarios on pytest.** Packs ship features and step definitions;
  the project adds its own in `features/` and `steps/`. Any pytest tooling
  (xdist, markers, fixtures, JUnit, CI widgets) works as is.
- **Treats correctness as numerical.** The inference pack compares a model's
  logits with a reference under a versioned tolerance policy (allclose limits,
  cosine, top-k agreement, two-sided perplexity) instead of `==`.
- **Treats performance as correctness.** Latency and throughput are judged
  against a pinned baseline *and* the last known-good run, with budgets in Gherkin.
- **Spends scarce hardware deliberately.** Tests are tiered
  (`unit → sim → single_card → multi_card → rack`); schedules decide what runs
  on commit, nightly and release, and missing hardware means a skip with a reason.
- **Says where it broke.** Failures are attributed to a stack layer (firmware,
  compiler, runtime, serving, hardware, test, flaky) from explicit errors, what
  changed since green, error signals and reruns. Browser scenarios keep the
  UI/API/business-logic Developer Brief.
- **Decides releases on evidence.** `featureforge gate` turns the run, its history
  and the escape log into PASS/BLOCK with every check's measured value and limit.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python -m playwright install chromium   # only needed for UI scenarios

pytest                                  # unit tests + BDD scenarios
featureforge plan                       # scenarios per tier, what each schedule runs
pytest --ff-schedule commit             # the tiers a commit can afford (unit + sim)
featureforge gate --policy commit       # PASS / BLOCK on that run
featureforge run --tier sim             # BDD scenarios only, with Cucumber JSON + JUnit
featureforge history                    # recent runs, what changed, flaky tests
featureforge run --tags "@demo"         # see a Developer Brief for a failing step
```

Try the hardware tiers without hardware (a simulated card backend):

```bash
FF_TIERS_AVAILABLE=unit,sim,single_card,multi_card \
FF_DEVICE_BACKEND=featureforge.packs.inference.backends:simulated_card \
pytest --ff-schedule nightly && featureforge gate --policy nightly

FF_SIM_FAULTY_CARD=2 ...  # same, with a bad board: attributed to HARDWARE, card 2 named
```

## Commands

| Command | What it does |
|---|---|
| `featureforge run [FEATURES…]` | Run BDD scenarios (`tests/bdd`); `--tags`, `--schedule`, `--tier`, `--env`, `--parallel`. Gates that require the `unit` tier need a plain `pytest` run |
| `featureforge plan` | Scenarios per tier and which schedule runs them |
| `featureforge gate --policy NAME` | PASS/BLOCK on the last run (exit 1 = BLOCK), report in `reports/gate-NAME.json` |
| `featureforge history` | Recent runs, failures by stack layer, regressions, changes since green, flaky tests |
| `featureforge escapes` | Post-release defects by layer and tier, and whether each has a test |
| `featureforge metrics` | Summary of the last run |
| `featureforge lint` / `scaffold` | Lint feature files / generate stubs for undefined steps |
| `featureforge packs` | Installed packs |

pytest options: `--ff-schedule`, `--ff-tier`, `--ff-feature`, `--ff-reruns N`,
`--ff-no-history`, `--ff-env`, `--ff-pack`, `--ff-headed`, `--ff-reports-dir`.

## Layout

```
features/                project feature files (subfolders by domain)
steps/                   project step definitions, loaded automatically
quality/escapes.toml     escape log (post-release defects → regression tests)
src/featureforge/
  plugin.py              pytest plugin: hooks, ctx, tiers, reruns, briefs, metrics, history
  cli.py                 featureforge run | plan | gate | history | escapes | lint | …
  collect.py             where scenarios and steps come from
  core/
    settings.py          typed config (pyproject → env vars → CLI), paths from project root
    context.py           ScenarioContext (the World): state, attachments, ctx.measure()
    tiers.py             test tiers and schedules
    stack.py             StackLayer + errors that name their layer
    manifest.py          build manifest: component versions, board, commit
    history.py           run history: last known-good, flakiness, new failures
    perf.py              latency summaries, baselines, regression judgement
    gate.py              release gates
    escapes.py           escape log
    metrics.py           run metrics
    intelligence/        classifier, layer validator, developer brief, stack attribution
    browser.py, network.py, telemetry.py, llm.py, pages/
  packs/
    ai/                  fairness + RAG
    inference/           accuracy vs reference, determinism, cards, latency
      tolerances.toml    the tolerance policy
      baselines.toml     pinned performance baselines
  tools/                 Gherkin reader, linter, step scaffolder
tests/
  bdd/test_packs.py      collects pack + project scenarios
  unit/                  unit tests + end-to-end plugin tests (pytester)
```

## Configuration

`[tool.featureforge]` in `pyproject.toml`, overridden by environment variables.
Relative paths resolve from the project root.

| Setting | Env var | Default |
|---|---|---|
| `env` | `ENV` | `qa` |
| `packs` | `FF_PACKS` | `["ai"]` (this repo: `["ai", "inference"]`) |
| `reports_dir` | `FF_REPORTS_DIR` | `reports` |
| `features_dir` / `steps_dir` | | `features` / `steps` |
| `tiers.available` | `FF_TIERS_AVAILABLE` | `["unit", "sim"]` |
| `tiers.default`, `tiers.schedules` | | `sim`; commit / nightly / release |
| `device_backend` | `FF_DEVICE_BACKEND` | none (card tiers skip) |
| `tolerance_policy` | `FF_TOLERANCE_POLICY` | the inference pack's `tolerances.toml` |
| `perf_baselines` | | the inference pack's `baselines.toml` |
| `manifest` (components) | `FF_COMPONENT_<NAME>`, `FF_HARDWARE`, `FF_MANIFEST` (JSON) | git commit only |
| `history_dir`, `history_keep` | `FF_HISTORY_DIR` | `.featureforge/history`, 200 runs |
| `gates.<name>` | | built-in commit / nightly / release |
| `escapes_file` | | `quality/escapes.toml` |
| `headless` | `HEADLESS` | `true` (or `--ff-headed`) |
| `enable_tracing`, `otlp_endpoint` | `ENABLE_TRACING`, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | off |
| `llm_provider`, `llm_model` | `FF_LLM_PROVIDER`, `FF_LLM_MODEL` | `none` (`openai` if `OPENAI_API_KEY`) |

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

Enable it with `packs = ["ai", "billing"]`; `tests/bdd/test_packs.py` collects every
enabled pack. Missing steps can be generated with
`featureforge scaffold my_pack/features --out my_pack/steps.py`.

To qualify real hardware, implement a card factory
`(model, dtype, card) -> Backend` that raises `featureforge.core.stack` errors
(`FirmwareError`, `CompilerError`, `RuntimeFault`, `DeviceError` …) and point
`device_backend` at it.

## License

Business Source License 1.1 — see [LICENSE.txt](LICENSE.txt).
