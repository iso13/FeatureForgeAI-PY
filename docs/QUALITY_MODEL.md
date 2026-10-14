# Quality model

How FeatureForge decides whether an inference stack works, and what it does
when it doesn't. A model's answer passes through firmware, the compiler, the
runtime and the serving stack before a token comes back; a regression anywhere
shows up as a wrong answer, a slow answer, or an answer that was right yesterday.

## 1. Correctness is numerical

Quantized inference is not bit-identical to the reference, so "correct" is a set
of limits, not `==`. They live in `packs/inference/tolerances.toml`:

| Limit | Why |
|---|---|
| `atol` + `rtol` per element, `max_mismatch_fraction` | `numpy.allclose` semantics; catches small systematic shifts |
| `min_cosine` | overall direction of the logits |
| `min_top1_agreement`, `min_topk_overlap` | what users see: the chosen and candidate tokens |
| `max_perplexity_delta_pct` (two-sided) | quality on the reference's own tokens |
| NaN / inf | always a violation |

Rules resolve `model/dtype → */dtype → model/* → defaults` and only list what
they change. Each run records the policy version and fingerprint, so a loosened
limit is visible in history.

The perplexity check is two-sided on purpose: a 2% int8 scale drift lowers
perplexity by 1.4% (the model gets overconfident) while 32% of elements are out
of tolerance. A one-sided "quality got worse" check would pass that build.

Limits are calibrated on the smoke set with ~1.5-2x headroom over a healthy
build's measured error. The pack proves its own checks work: scenarios inject a
drift, nondeterminism and a slowdown and assert they are caught.

## 2. Performance is correctness

Each latency/throughput number is compared with two baselines:

- **pinned**: reviewed numbers in `baselines.toml` (the last release),
- **last known-good**: the same measurement from the newest green run.

Worse than either by more than the budget in the Gherkin step fails the scenario
with a `PerformanceRegression`. The pinned baseline stops the ratchet: at 3% per
week, last-known-good alone never trips a 10% budget, because each green week
becomes the new baseline. Fault-injected runs record under their own key so they
never become anyone's baseline.

## 3. Tiers: test capacity is physical

```
unit → sim → single_card → multi_card → rack
```

Tag scenarios `@tier-<name>`. Each runner declares what it has
(`FF_TIERS_AVAILABLE`); each schedule declares what it runs:

| Schedule | Tiers | Where |
|---|---|---|
| commit | unit, sim | any runner, every push |
| nightly | + single_card, multi_card | card runners, `--ff-reruns 1` |
| release | + rack | rack runner, tags |

A selected test whose tier isn't available is skipped with the reason and still
recorded, so a gate can say "rack: 0 ran, 3 skipped" instead of silently passing.
`featureforge plan` shows the counts.

## 4. Attribution: where it broke

Every failure gets a stack layer, a confidence, a method and evidence. Strongest
evidence wins:

1. **rerun**: failed, then passed with nothing changed → `FLAKY`
2. **explicit**: a backend raised `CompilerError`, `DeviceError` …
3. **test code**: a Python error from step/test files → `TEST`
4. **change**: exactly one component changed since last known-good → that layer
5. **signals**: `MLIR`, `mailbox`, `DMA`, `HTTP 503`, `ECC` … in the error

Numerical or performance failures that can't be pinned stay `MODEL_NUMERICS` /
`PERFORMANCE` with the candidate layers and a "bisect" hint. A card that
disagrees with the others is `HARDWARE` and named; with no majority, no single
card is blamed.

Reruns keep module/session fixtures up between attempts, so "passed on rerun"
really means the same state.

## 5. History: what changed since green

Each run is archived with its **build manifest** (component versions, board,
commit) and its **scope** (args, markers, tiers, features). The last known-good
run is the newest green run of the same scope; a narrow green run doesn't vouch
for the full suite.

- **flaky** = outcome flips on the *same* build (or a pass on rerun); flips across
  builds are regressions, not flakes.
- `featureforge history` shows runs, layers, regressions, changes since green and
  the flakiest tests.

In CI, persist `.featureforge/history` between pipelines.

## 6. Gates: what blocks a release

`featureforge gate --policy commit|nightly|release`, configurable under
`[tool.featureforge.gates.<name>]`:

| Check | commit | nightly | release |
|---|---|---|---|
| pass rate | 100% | 100% | 100% |
| required tiers ran, none skipped | unit, sim | + cards | all |
| new failures vs last known-good | 0 | 0 | 0 |
| flaky | - | ≤ 2 | 0 |
| perf regressions | 0 | 0 | 0 |
| escapes without a regression test | - | - | 0 |
| every stack component versioned | - | - | yes |

A known failure can be **quarantined** with a ticket and an expiry date; an
expired waiver blocks. Unknown keys are rejected, so a typo can't silently
remove a check.

## 7. Escapes: every escape is a missing test

`quality/escapes.toml` records each post-release defect: the layer it lived in,
the tier that should have caught it, and the regression test that now does.
`featureforge escapes` reports coverage by layer and tier, which shows where the
next tests should go.

## What's simulated

`SimulatedAccelerator` uses real bf16 / fp8 e4m3 / int8 arithmetic on a
deterministic reference model, plus a latency model, so every scenario runs
anywhere. Real hardware plugs in through `device_backend` (a card factory) and
real serving endpoints through `OpenAICompatibleBackend`; the scenarios,
tolerances, tiers, attribution and gates stay the same.
