"""Domain logic for the AI pack, kept free of pytest so it is easy to unit-test.

TS equivalents: the mock model in ``src/steps/ai/bias.steps.ts`` and
``RAGEngine`` in ``src/plugins/ai/ragHelper.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# ─── Fairness ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sample:
    demographic: str
    income: int = 60_000
    age: int = 35


class ScoringModel(Protocol):
    def predict(self, sample: Sample) -> float: ...


@dataclass
class BiasSimulatingModel:
    """Deterministic stand-in with small, deliberate group skews."""

    scores: dict[str, float] = field(
        default_factory=lambda: {
            "white": 0.55,
            "black": 0.45,
            "asian": 0.50,
            "hispanic": 0.48,
            "nonbinary": 0.50,
            "female": 0.52,
            "male": 0.50,
        }
    )
    default: float = 0.5

    def predict(self, sample: Sample) -> float:
        return self.scores.get(sample.demographic.lower(), self.default)


@dataclass(frozen=True)
class ParityResult:
    demographic: str
    output: float
    baseline: float
    threshold: float

    @property
    def delta(self) -> float:
        return round(abs(self.output - self.baseline), 6)

    @property
    def passed(self) -> bool:
        return self.delta <= self.threshold

    def render(self) -> str:
        return (
            f"Demographic: {self.demographic}\n"
            f"Model output: {self.output}\n"
            f"Baseline: {self.baseline}\n"
            f"Delta: {self.delta}\n"
            f"Threshold: {self.threshold}\n"
            f"Result: {'PASS' if self.passed else 'FAIL'}"
        )


def check_parity(sample: Sample, output: float, baseline: float, threshold: float) -> ParityResult:
    return ParityResult(sample.demographic, output, baseline, threshold)


# ─── RAG ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Clause:
    id: str
    type: str
    content: str


@dataclass
class RAGEngine:
    """Keyword-routed retrieval + extractive summary over contract clauses.

    Fix over the TS version: ``retrieved_ids`` reports the clauses actually used
    for the last summary. ``getRetrievedClauseIds`` in TS returned every indexed
    clause, so the "retrieves relevant context" assertion could never fail.
    """

    index: list[Clause] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    summary: str = ""
    retrieved_ids: list[str] = field(default_factory=list)

    def index_documents(self, clauses: list[Clause]) -> None:
        self.index = list(clauses)
        self.logs.append(f"Indexed {len(clauses)} clauses.")

    def add_clause(self, clause: Clause) -> None:
        self.index.append(clause)
        self.logs.append(f"Added new clause of type {clause.type}")

    def request_summary(self, requested_types: list[str]) -> str:
        wanted = {t.lower() for t in requested_types}
        retrieved = [c for c in self.index if c.type.lower() in wanted]
        self.retrieved_ids = [c.id for c in retrieved]
        missing = sorted(wanted - {c.type.lower() for c in retrieved})
        if not retrieved:
            self.logs.append("Warning: No clauses retrieved for summary.")
            self.summary = "Summary incomplete due to missing clauses."
            return self.summary
        if missing:
            self.logs.append(f"Warning: missing clause types: {', '.join(missing)}")
        lines = [f"• {c.type}: {c.content}" for c in retrieved]
        if missing:
            lines.append(f"Note: summary incomplete — no {', '.join(missing)} clauses found.")
        self.summary = "\n".join(lines)
        self.logs.append(f"Generated summary with {len(retrieved)} clauses.")
        return self.summary


CONTRACT_CLAUSES = [
    Clause("C001", "confidentiality", "This contract includes a confidentiality clause."),
    Clause("C002", "payment", "Payment must be made within 30 days of invoice."),
]

INDEMNITY_CLAUSE = Clause("C003", "indemnity", "Indemnity clause added for liability coverage.")
