"""Step definitions for the inference pack (loaded as a pytest plugin)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when

from featureforge.core.context import ScenarioContext
from featureforge.core.settings import Settings
from featureforge.core.stack import DeviceError, NumericalMismatch, RuntimeFault
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
POSITIONS = 16


@dataclass
class InferenceState:
    reference: ReferenceBackend | None = None
    candidate: Backend | None = None
    cards: dict[int, Backend] = field(default_factory=dict)
    card_runs: dict[int, Generation] = field(default_factory=dict)
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
        for position, (a, b) in enumerate(zip(baseline, run.tokens, strict=True)):
            if a != b:
                return index, position
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
    """Cards whose tokens differ from the most common output across cards."""
    outputs = [tuple(run.tokens) for run in runs.values()]
    majority = max(set(outputs), key=outputs.count)
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
