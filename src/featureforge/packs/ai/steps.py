"""Step definitions for the AI pack (loaded as a pytest plugin).

TS equivalents: ``src/steps/ai/bias.steps.ts`` and ``src/steps/ai/rag.steps.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pytest_bdd import given, parsers, then, when

from featureforge.core.context import ScenarioContext
from featureforge.core.telemetry import span
from featureforge.packs.ai.models import (
    CONTRACT_CLAUSES,
    INDEMNITY_CLAUSE,
    BiasSimulatingModel,
    RAGEngine,
    Sample,
    ScoringModel,
    check_parity,
)

DEFAULT_BASELINE = 0.5
DEFAULT_THRESHOLD = 0.06


@dataclass
class FairnessState:
    model: ScoringModel | None = None
    sample: Sample | None = None
    output: float | None = None
    baseline: float = DEFAULT_BASELINE


@dataclass
class RagState:
    engine: RAGEngine = field(default_factory=RAGEngine)
    summary: str = ""


# ─── Fairness ────────────────────────────────────────────────────────────────


@given("a model is loaded")
def model_is_loaded(ctx: ScenarioContext) -> None:
    ctx.state(FairnessState).model = BiasSimulatingModel()


@given(parsers.parse('I input a sample with demographic group "{group}"'))
def input_sample(ctx: ScenarioContext, group: str) -> None:
    ctx.state(FairnessState).sample = Sample(demographic=group)


@when("I get the model's decision")
def get_decision(ctx: ScenarioContext) -> None:
    state = ctx.state(FairnessState)
    assert state.model is not None, "Model not available — add 'Given a model is loaded'"
    assert state.sample is not None, "No input sample provided"
    with span("model.predict", demographic=state.sample.demographic):
        state.output = state.model.predict(state.sample)


def _assert_parity(ctx: ScenarioContext, threshold: float) -> None:
    state = ctx.state(FairnessState)
    assert state.sample is not None and state.output is not None, "No model decision recorded"
    result = check_parity(state.sample, state.output, state.baseline, threshold)
    ctx.attach(result.render(), "text/plain", "fairness_check")
    assert result.passed, (
        f"Fairness threshold exceeded for '{result.demographic}': "
        f"expected delta <= {threshold}, got {result.delta}"
    )


@then("the decision outcome should be within the fairness threshold")
def within_default_threshold(ctx: ScenarioContext) -> None:
    _assert_parity(ctx, DEFAULT_THRESHOLD)


@then(parsers.parse("the decision outcome should be within a fairness threshold of {threshold:g}"))
def within_threshold(ctx: ScenarioContext, threshold: float) -> None:
    _assert_parity(ctx, threshold)


# ─── RAG ─────────────────────────────────────────────────────────────────────


@given(
    "the RAG system has indexed contract documents containing clauses for "
    "confidentiality and payment terms"
)
def rag_indexed(ctx: ScenarioContext) -> None:
    ctx.state(RagState).engine.index_documents(CONTRACT_CLAUSES)


@given("a new indemnity clause is added to the contract")
def add_indemnity(ctx: ScenarioContext) -> None:
    ctx.state(RagState).engine.add_clause(INDEMNITY_CLAUSE)


@when("I request a summary of a contract")
def request_summary(ctx: ScenarioContext) -> None:
    state = ctx.state(RagState)
    state.summary = state.engine.request_summary(["confidentiality", "payment"])


@when("I request a summary of a contract missing indemnity clauses")
def request_missing(ctx: ScenarioContext) -> None:
    state = ctx.state(RagState)
    state.summary = state.engine.request_summary(["indemnity"])


@when("I re-request the summary")
def re_request(ctx: ScenarioContext) -> None:
    state = ctx.state(RagState)
    state.summary = state.engine.request_summary(["confidentiality", "payment", "indemnity"])


@then("the system should retrieve relevant context for each clause")
def retrieved_each_clause(ctx: ScenarioContext) -> None:
    retrieved = ctx.state(RagState).engine.retrieved_ids
    assert retrieved == ["C001", "C002"], f"Unexpected retrieval: {retrieved}"


@then("the system should generate a summary including confidentiality and payment terms")
def summary_has_terms(ctx: ScenarioContext) -> None:
    summary = ctx.state(RagState).summary
    assert "confidentiality" in summary
    assert "payment" in summary


@then("the summary should match the retrieved content")
def summary_matches(ctx: ScenarioContext) -> None:
    summary = ctx.state(RagState).summary
    for clause in CONTRACT_CLAUSES:
        assert clause.content in summary, f"Summary is missing clause {clause.id}"


@then("the system should log a warning about the missing context")
def warning_logged(ctx: ScenarioContext) -> None:
    logs = ctx.state(RagState).engine.logs
    assert any("warning" in line.lower() for line in logs), f"No warning in logs: {logs}"


@then("the generated summary should note the missing indemnity information")
def notes_missing(ctx: ScenarioContext) -> None:
    assert "incomplete" in ctx.state(RagState).summary.lower()


@then("the new clause should be included in the updated AI summary")
def includes_new_clause(ctx: ScenarioContext) -> None:
    summary = ctx.state(RagState).summary
    assert "indemnity" in summary
    assert INDEMNITY_CLAUSE.content in summary
