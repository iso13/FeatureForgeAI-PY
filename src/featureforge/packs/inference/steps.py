"""Step definitions for the inference pack (loaded as a pytest plugin)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when

from featureforge.core.context import ScenarioContext
from featureforge.core.history import RunHistory
from featureforge.core.metrics import Measurement
from featureforge.core.perf import Baseline, LatencySummary, PinnedBaselines, judge
from featureforge.core.settings import Settings
from featureforge.core.stack import (
    DeviceError,
    NumericalMismatch,
    PerformanceRegression,
    RuntimeFault,
)
from featureforge.core.telemetry import span
from featureforge.packs.inference.backends import (
    Backend,
    CardFactory,
    Faults,
    Generation,
    ReferenceBackend,
    SimulatedAccelerator,
    load_card_factory,
)
from featureforge.packs.inference.compare import ComparisonResult, compare_logits, merge
from featureforge.packs.inference.tolerance import TolerancePolicy

DATA_DIR = Path(__file__).parent / "data"
PINNED_BASELINES = Path(__file__).parent / "baselines.toml"
POSITIONS = 16


@dataclass
class InferenceState:
    reference: ReferenceBackend | None = None
    candidate: Backend | None = None
    cards: dict[int, Backend] = field(default_factory=dict)
    card_runs: dict[int, Generation] = field(default_factory=dict)
    benchmark: dict[str, Measurement] = field(default_factory=dict)
    policy: TolerancePolicy | None = None
    comparison: ComparisonResult | None = None
    generations: list[Generation] = field(default_factory=list)


def load_prompts(name: str) -> list[str]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in DATA_DIR.glob("*.jsonl")))
        raise FileNotFoundError(f"No prompt set '{name}'. Known: {known}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["prompt"] for line in lines if line.strip()]


def _candidate(ctx: ScenarioContext) -> Backend:
    candidate = ctx.state(InferenceState).candidate
    assert candidate is not None, "No accelerator configured — add 'Given the accelerator runs …'"
    return candidate


def _faults(ctx: ScenarioContext) -> Faults:
    candidate = _candidate(ctx)
    assert isinstance(candidate, SimulatedAccelerator), "Faults can only be injected in simulation"
    return candidate.faults


@pytest.fixture
def ff_cards(ff_settings: Settings) -> CardFactory:
    """The configured card backend; skips the scenario when none is configured."""
    if not ff_settings.device_backend:
        pytest.skip("no device backend configured (set device_backend or FF_DEVICE_BACKEND)")
    return load_card_factory(ff_settings.device_backend)


# ─── setup ──────────────────────────────────────────────────────────────────


@given(parsers.parse('the "{model}" fp32 reference model'))
def reference_model(ctx: ScenarioContext, model: str) -> None:
    ctx.state(InferenceState).reference = ReferenceBackend(model=model)


@given("the tolerance policy is loaded")
def tolerance_policy(ctx: ScenarioContext) -> None:
    policy = TolerancePolicy.load(ctx.settings.tolerance_policy)
    ctx.state(InferenceState).policy = policy
    ctx.attach(
        f"Tolerance policy {policy.version} ({policy.fingerprint}) from {policy.source}",
        name="tolerance_policy",
    )


@given(parsers.parse('the accelerator runs "{model}" in "{dtype}"'))
def accelerator_runs(ctx: ScenarioContext, model: str, dtype: str) -> None:
    ctx.state(InferenceState).candidate = SimulatedAccelerator(model=model, dtype=dtype)


@given(parsers.parse("a {percent:g}% scale drift is injected"))
def inject_scale_drift(ctx: ScenarioContext, percent: float) -> None:
    _faults(ctx).scale_drift = percent / 100


@given("the accelerator output varies from call to call")
def inject_nondeterminism(ctx: ScenarioContext) -> None:
    faults = _faults(ctx)
    faults.noise_sigma, faults.nondeterministic = 0.5, True


@given(parsers.parse("a {percent:g}% slowdown is injected"))
def inject_slowdown(ctx: ScenarioContext, percent: float) -> None:
    _faults(ctx).slowdown = percent / 100


@given(parsers.parse('card {card:d} runs "{model}" in "{dtype}"'))
def card_runs(
    ctx: ScenarioContext, ff_cards: CardFactory, card: int, model: str, dtype: str
) -> None:
    ctx.state(InferenceState).candidate = ff_cards(model, dtype, card)


@given(parsers.parse('cards {first:d} to {last:d} run "{model}" in "{dtype}"'))
def cards_run(
    ctx: ScenarioContext, ff_cards: CardFactory, first: int, last: int, model: str, dtype: str
) -> None:
    ctx.state(InferenceState).cards = {
        card: ff_cards(model, dtype, card) for card in range(first, last + 1)
    }


# ─── numerical accuracy ─────────────────────────────────────────────────────


@when(parsers.parse('I compare logits on the "{prompt_set}" prompt set'))
def compare_on_prompt_set(ctx: ScenarioContext, prompt_set: str) -> None:
    state = ctx.state(InferenceState)
    candidate = _candidate(ctx)
    reference = state.reference or ReferenceBackend(model=candidate.model)
    policy = state.policy or TolerancePolicy.load(ctx.settings.tolerance_policy)
    rule = policy.rule_for(candidate.model, candidate.dtype)
    results = []
    with span("inference.compare", model=candidate.model, dtype=candidate.dtype):
        for prompt in load_prompts(prompt_set):
            results.append(
                compare_logits(
                    reference.logits(prompt, POSITIONS),
                    candidate.logits(prompt, POSITIONS),
                    rule,
                    model=candidate.model,
                    dtype=candidate.dtype,
                    policy_version=policy.version,
                    policy_fingerprint=policy.fingerprint,
                )
            )
    state.comparison = merge(results)
    ctx.attach(state.comparison.render(), name="numerical_comparison")
    ctx.attach(state.comparison.model_dump_json(indent=2), "application/json", "comparison")
    stats = state.comparison.stats
    ctx.measure("cosine", stats.cosine, higher_is_better=True)
    ctx.measure("top1_agreement", stats.top1_agreement, higher_is_better=True)
    ctx.measure("mismatch_fraction", stats.mismatch_fraction)
    ctx.measure("perplexity_delta_pct", abs(stats.perplexity_delta_pct), unit="%")


def _comparison(ctx: ScenarioContext) -> ComparisonResult:
    comparison = ctx.state(InferenceState).comparison
    assert comparison is not None, "No comparison yet — add 'When I compare logits …'"
    return comparison


@then("every limit in the tolerance policy should hold")
def tolerances_hold(ctx: ScenarioContext) -> None:
    comparison = _comparison(ctx)
    state = ctx.state(InferenceState)
    if state.policy is not None and not state.policy.has_rule(comparison.model, comparison.dtype):
        hint = f" (no rule for {comparison.model}/{comparison.dtype}; fp32 defaults applied)"
    else:
        hint = ""
    if not comparison.passed:
        raise NumericalMismatch(f"Outside the tolerance policy{hint}:\n{comparison.render()}")


@then(parsers.parse('the comparison should flag "{metric}"'))
def comparison_flags(ctx: ScenarioContext, metric: str) -> None:
    comparison = _comparison(ctx)
    flagged = ", ".join(v.metric for v in comparison.violations) or "nothing"
    assert comparison.violated(metric), f"Expected '{metric}' to be flagged; flagged: {flagged}"


# ─── performance ────────────────────────────────────────────────────────────


def measurement_key(backend: Backend) -> str:
    """Baselines match on this. Fault-injected runs get their own key so they never
    become anyone's baseline."""
    key = f"{backend.model}/{backend.dtype}"
    faults = getattr(backend, "faults", None)
    if faults is not None and faults != Faults():
        key += "+faults"
    return key


@when(
    parsers.parse(
        'I send {requests:d} requests of {tokens:d} tokens from the "{prompt_set}" prompt set'
    )
)
def run_benchmark(ctx: ScenarioContext, requests: int, tokens: int, prompt_set: str) -> None:
    backend = _candidate(ctx)
    prompts = load_prompts(prompt_set)
    with span("inference.benchmark", model=backend.model, dtype=backend.dtype, requests=requests):
        runs = [backend.generate(prompts[i % len(prompts)], tokens) for i in range(requests)]
    ttft = LatencySummary.of([r.ttft_ms for r in runs])
    tpot = LatencySummary.of([r.tpot_ms for r in runs])
    throughput = sum(len(r.tokens) for r in runs) / sum(r.total_ms for r in runs) * 1000
    key = measurement_key(backend)
    state = ctx.state(InferenceState)
    for name, value, unit, higher in (
        ("ttft_p50_ms", ttft.p50, "ms", False),
        ("ttft_p99_ms", ttft.p99, "ms", False),
        ("tpot_p50_ms", tpot.p50, "ms", False),
        ("throughput_tok_s", throughput, "tok/s", True),
    ):
        state.benchmark[name] = ctx.measure(name, value, unit, higher_is_better=higher, key=key)


def _baselines(settings: Settings, history: RunHistory, key: str, name: str) -> list[Baseline]:
    clean_key = key.removesuffix("+faults")  # fault runs are judged against the clean baseline
    found = []
    pinned = PinnedBaselines.load(settings.perf_baselines or PINNED_BASELINES).get(clean_key, name)
    if pinned is not None:
        found.append(Baseline(pinned, "pinned"))
    good = history.baseline_measurement(clean_key, name)
    if good is not None:
        measurement, run = good
        found.append(Baseline(measurement.value, "last-known-good", run.run_id))
    return found


def _judged(
    ctx: ScenarioContext, history: RunHistory, metric: str, budget_pct: float
) -> Measurement:
    measured = ctx.state(InferenceState).benchmark.get(metric)
    assert measured is not None, f"No '{metric}' measured — add 'When I send … requests …'"
    baselines = _baselines(ctx.settings, history, measured.key, metric)
    assert baselines, f"No baseline for {measured.key} {metric}: pin one in baselines.toml"
    return judge(measured, baselines, budget_pct)


def _describe(m: Measurement) -> str:
    return (
        f"{m.name} {m.value:.4g}{m.unit} vs {m.baseline_source} baseline {m.baseline:.4g}{m.unit}: "
        f"{m.delta_pct:+.1f}% (budget {m.budget_pct:g}%)"
    )


@then(
    parsers.re(
        r'"(?P<metric>\w+)" should not (?:regress|drop) more than (?P<pct>[\d.]+)% '
        r"against the baseline"
    )
)
def within_budget(ctx: ScenarioContext, ff_history: RunHistory, metric: str, pct: str) -> None:
    judged = _judged(ctx, ff_history, metric, float(pct))
    ctx.measurements = [judged if m.name == metric else m for m in ctx.measurements]
    if judged.regressed:
        raise PerformanceRegression(f"Performance regression: {_describe(judged)}")


@then(parsers.parse('"{metric}" should be flagged as a regression of more than {pct:g}%'))
def flagged_as_regression(
    ctx: ScenarioContext, ff_history: RunHistory, metric: str, pct: float
) -> None:
    judged = _judged(ctx, ff_history, metric, pct)
    ctx.attach(_describe(judged), name=f"{metric}_verdict")
    assert judged.regressed, f"Expected a regression to be flagged: {_describe(judged)}"


# ─── determinism ────────────────────────────────────────────────────────────


@when(parsers.parse("I generate {tokens:d} tokens for the same prompt {times:d} times"))
def generate_repeatedly(ctx: ScenarioContext, tokens: int, times: int) -> None:
    candidate = _candidate(ctx)
    prompt = load_prompts("smoke")[0]
    ctx.state(InferenceState).generations = [
        candidate.generate(prompt, tokens) for _ in range(times)
    ]


def first_divergence(runs: list[Generation]) -> tuple[int, int] | None:
    """(run index, token position) of the first difference from run 0, if any."""
    baseline = runs[0].tokens
    for index, run in enumerate(runs[1:], start=1):
        for position, (a, b) in enumerate(zip(baseline, run.tokens, strict=False)):
            if a != b:
                return index, position
        if len(run.tokens) != len(baseline):
            return index, min(len(run.tokens), len(baseline))
    return None


@then("every run should produce identical tokens")
def runs_identical(ctx: ScenarioContext) -> None:
    runs = ctx.state(InferenceState).generations
    assert len(runs) >= 2, "Need at least two generations to compare"
    divergence = first_divergence(runs)
    if divergence is not None:
        run, position = divergence
        raise RuntimeFault(
            f"Nondeterministic output: run {run} diverged from run 0 at token {position} "
            "on the same build, prompt and inputs"
        )


@when(parsers.parse("each card generates {tokens:d} tokens for the same prompt"))
def each_card_generates(ctx: ScenarioContext, tokens: int) -> None:
    state = ctx.state(InferenceState)
    assert state.cards, "No cards configured — add 'Given cards … run …'"
    prompt = load_prompts("smoke")[0]
    state.card_runs = {
        card: backend.generate(prompt, tokens) for card, backend in state.cards.items()
    }


def odd_cards_out(runs: dict[int, Generation]) -> list[int]:
    """Cards whose tokens differ from the majority output; all cards if there's no majority."""
    outputs = [tuple(run.tokens) for run in runs.values()]
    majority = max(set(outputs), key=outputs.count)
    if outputs.count(majority) * 2 <= len(outputs) and len(set(outputs)) > 1:
        return list(runs)  # e.g. 2 vs 2: can't tell which board is bad
    return [card for card, run in runs.items() if tuple(run.tokens) != majority]


@then("every card should produce identical tokens")
def cards_identical(ctx: ScenarioContext) -> None:
    runs = ctx.state(InferenceState).card_runs
    odd = odd_cards_out(runs)
    if odd:
        names = ", ".join(str(card) for card in odd)
        raise DeviceError(
            f"Card(s) {names} disagree with the other {len(runs) - len(odd)} card(s) on the "
            "same build and prompt: suspect the board, not the software"
        )


@then("the runs should not all be identical")
def runs_differ(ctx: ScenarioContext) -> None:
    runs = ctx.state(InferenceState).generations
    assert first_divergence(runs) is not None, "Expected the injected fault to change the output"
