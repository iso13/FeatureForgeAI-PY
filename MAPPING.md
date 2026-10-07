# TypeScript → Python mapping

How the FeatureForgeAI TypeScript shell (`iso13/FeatureForgeAI`, commit `6c6aa4b`)
maps onto this Python engine. The two engines share the same contract: Gherkin
`.feature` files, the tag conventions, Cucumber JSON output and the Developer
Brief JSON schema.

## Runtime

| TypeScript | Python | Notes |
|---|---|---|
| `cucumber.mjs` | `[tool.pytest.ini_options]` in `pyproject.toml` | Same default tag filter (`not @wip and not @performanceTest`), plus `@demo` |
| `src/core/support/world.ts` (`CustomWorld`) | `featureforge/core/context.py` (`ScenarioContext`, the `ctx` fixture) | ~60 optional, mostly `any` fields replaced by typed per-pack state: `ctx.state(FairnessState)` |
| `src/core/support/hooks.ts` | `featureforge/plugin.py` | `Before`→`pytest_bdd_before_scenario`, `BeforeStep`→`pytest_bdd_before_step`, failed `AfterStep`→`pytest_bdd_step_error`, `After`→`pytest_bdd_after_scenario` + `pytest_runtest_makereport`, `AfterAll`→`pytest_sessionfinish` |
| `launchBrowser` / `pageFixture.ts` | `featureforge/core/browser.py` | Lazy: a browser starts only when a step asks for `page`. One browser per session, one context per scenario. `@no-browser` is no longer needed |
| `src/core/support/environments.ts`, `process.env.*` | `featureforge/core/settings.py` | One typed `Settings` model: defaults → `[tool.featureforge]` → env vars (same names: `ENV`, `HEADLESS`, `ENABLE_TRACING`, `OPENAI_API_KEY`…) → CLI flags |
| `tracer.ts`, `plugins/telemetry/traceHelper.ts` (`withSpan`) | `featureforge/core/telemetry.py` (`span`, `start_span`) | No-op unless tracing is on **and** the `telemetry` extra is installed |
| `new OpenAI()` in several modules | `featureforge/core/llm.py` (`LLMClient`, `OpenAILLM`, `OllamaLLM`, `NullLLM`) | Injected, not created at import time. No key → heuristics only, no network |
| `src/core/pages/BasePage.ts`, `loginPage.ts` | `featureforge/core/pages/` | Role/label locators; URLs relative to the environment base URL |
| `src/core/steps/env.steps.ts` | `featureforge/core/steps.py` | |
| `src/core/support/metrics.ts` (Prometheus histogram) | `featureforge/core/metrics.py` → `reports/run-metrics.json` | Pass rate, durations, failures by root-cause layer, per run |

## Step Intelligence Layer

| TypeScript | Python |
|---|---|
| `intelligence/failureClassifier.ts` | `core/intelligence/classifier.py` |
| `intelligence/layerValidator.ts` | `core/intelligence/layer_validator.py` + `core/network.py` |
| `intelligence/developerBrief.ts` | `core/intelligence/brief.py` |
| interfaces in each file | `core/intelligence/models.py` (Pydantic) |

Same decision rules (one layer → 0.85, two → FULL_STACK 0.60, otherwise LLM or
fallback 0.30) and the same brief fields, plus:

- **Symptom → cause reconciliation.** If the API layer failed, a UI or assertion
  failure is re-attributed to `API_CONTRACT` (method `heuristic+layers`).
- **Scenario-wide network evidence.** The failing step often isn't the one that
  made the bad call, so the validator sees every request in the scenario.

## Bugs found while porting (present in the TS engine)

1. `'500'` substring matched `Timeout 5000ms exceeded`, so UI timeouts were
   classified as API failures. Signals are now word-bounded regexes.
2. The business-logic signal `'tobе'` contains a Cyrillic `е` and can never match.
3. `BeforeStep` called `attachNetworkCapture` on every step, stacking listeners
   and duplicating entries.
4. Responses were matched to requests by URL + method, so repeated calls to one
   endpoint all updated the first entry.
5. The Developer Brief HTML interpolated error text and page text unescaped.
6. `RAGEngine.getRetrievedClauseIds()` returned every indexed clause, so the
   "retrieves relevant context" assertion could not fail. `rag.steps.ts` also has
   no matching feature file; `ContractSummary.feature` now exercises it.
7. Playwright/assertion failures with no API traffic were reported as
   "no business logic violations"; they now count as business-logic evidence.

## Packs (the plugin layer)

| TypeScript | Python |
|---|---|
| `src/features/ai/**`, `src/steps/ai/**`, `src/plugins/ai/**` | `featureforge/packs/ai/` (features, `steps.py`, `models.py`) |
| `implementations/{customer}/` (restructure plan) | A separate package registered under the `featureforge.packs` entry-point group |

Ported so far: `BiasAudit.feature` and the RAG contract-summary steps.

## Tooling

| npm script | Python |
|---|---|
| `npm run cucumber[:dev\|:staging]` | `featureforge run [--env dev] [--tags "@smoke"]` |
| `npm run lint:gherkin` (+ `.gherkin-lintrc`) | `featureforge lint` (rules configured in `[tool.featureforge.lint]`) |
| `npm run scaffold:steps` | `featureforge scaffold <features> [--out file]` |
| `multiple-cucumber-html-reporter` | `--cucumberjson` output is compatible; reuse the same reporter |
| — | `featureforge metrics`, `featureforge packs` |

## Not ported yet

Generators (LLM feature generator, DOM generator, OpenAPI flow scanner,
PDF-to-feature, agentic), `enforceDeclarative`, Weaviate RAG, k6 performance,
axe accessibility, HIPAA/compliance, HiL device simulators, and the remaining
AI feature files (RAG, AgenticAI, FairnessAndBias). The pack and LLM interfaces
are in place for each of them.
