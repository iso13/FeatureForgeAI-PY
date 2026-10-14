"""Tolerance-aware comparison of candidate logits against a reference.

Pure Python on purpose: a few thousand elements per scenario do not need numpy,
and the pack then runs anywhere the framework does. Inputs are
``positions x vocabulary`` logits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from featureforge.packs.inference.tolerance import ToleranceRule

Matrix = Sequence[Sequence[float]]


class ComparisonStats(BaseModel):
    elements: int
    positions: int
    max_abs_err: float
    max_rel_err: float
    mismatch_fraction: float
    cosine: float
    top1_agreement: float
    topk_overlap: float
    perplexity_ref: float
    perplexity_candidate: float

    @property
    def perplexity_delta_pct(self) -> float:
        return (self.perplexity_candidate - self.perplexity_ref) / self.perplexity_ref * 100


class Violation(BaseModel):
    metric: str
    actual: float
    limit: float
    rule: str  # human-readable, e.g. "cosine >= 0.9995"

    def render(self) -> str:
        return f"{self.metric} = {self.actual:.6g} (limit: {self.rule})"


class ComparisonResult(BaseModel):
    model: str
    dtype: str
    stats: ComparisonStats
    rule: ToleranceRule
    violations: list[Violation] = Field(default_factory=list)
    policy_version: str = ""
    policy_fingerprint: str = ""

    @property
    def passed(self) -> bool:
        return not self.violations

    def violated(self, metric: str) -> bool:
        return any(v.metric == metric for v in self.violations)

    def render(self) -> str:
        s = self.stats
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"{verdict} {self.model}/{self.dtype} vs reference "
            f"(policy {self.policy_version}, {self.policy_fingerprint})",
            f"  max_abs_err {s.max_abs_err:.4g}  max_rel_err {s.max_rel_err:.4g}  "
            f"mismatch {s.mismatch_fraction:.4%}  cosine {s.cosine:.6f}",
            f"  top1 {s.top1_agreement:.2%}  top{self.rule.topk} overlap {s.topk_overlap:.2%}  "
            f"perplexity {s.perplexity_ref:.4f} → {s.perplexity_candidate:.4f} "
            f"({s.perplexity_delta_pct:+.3f}%)",
        ]
        lines += [f"  ✗ {v.render()}" for v in self.violations]
        return "\n".join(lines)


def _log_softmax(row: Sequence[float]) -> list[float]:
    peak = max(row)
    log_total = peak + math.log(sum(math.exp(x - peak) for x in row))
    return [x - log_total for x in row]


def _argmax(row: Sequence[float]) -> int:
    return max(range(len(row)), key=row.__getitem__)


def _topk(row: Sequence[float], k: int) -> set[int]:
    return set(sorted(range(len(row)), key=row.__getitem__, reverse=True)[:k])


def perplexity(logits: Matrix, targets: Sequence[int]) -> float:
    nll = [-_log_softmax(row)[t] for row, t in zip(logits, targets, strict=True)]
    return math.exp(sum(nll) / len(nll))


def compute_stats(reference: Matrix, candidate: Matrix, rule: ToleranceRule) -> ComparisonStats:
    if len(reference) != len(candidate) or not reference:
        raise ValueError(f"shape mismatch: {len(reference)} vs {len(candidate)} positions")
    max_abs = max_rel = 0.0
    mismatches = elements = 0
    dot = norm_r = norm_c = 0.0
    top1 = 0
    overlap = 0.0
    for ref_row, cand_row in zip(reference, candidate, strict=True):
        if len(ref_row) != len(cand_row):
            raise ValueError(f"vocabulary mismatch: {len(ref_row)} vs {len(cand_row)}")
        for r, c in zip(ref_row, cand_row, strict=True):
            err = abs(c - r)
            max_abs = max(max_abs, err)
            max_rel = max(max_rel, err / max(abs(r), 1e-12))
            mismatches += err > rule.atol + rule.rtol * abs(r)
            dot += r * c
            norm_r += r * r
            norm_c += c * c
            elements += 1
        top1 += _argmax(ref_row) == _argmax(cand_row)
        overlap += len(_topk(ref_row, rule.topk) & _topk(cand_row, rule.topk)) / rule.topk
    targets = [_argmax(row) for row in reference]  # the reference's own greedy tokens
    return ComparisonStats(
        elements=elements,
        positions=len(reference),
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        mismatch_fraction=mismatches / elements,
        cosine=dot / math.sqrt(norm_r * norm_c) if norm_r and norm_c else 0.0,
        top1_agreement=top1 / len(reference),
        topk_overlap=overlap / len(reference),
        perplexity_ref=perplexity(reference, targets),
        perplexity_candidate=perplexity(candidate, targets),
    )


def check(stats: ComparisonStats, rule: ToleranceRule) -> list[Violation]:
    found: list[Violation] = []

    def at_most(metric: str, actual: float, limit: float) -> None:
        if actual > limit:
            found.append(
                Violation(metric=metric, actual=actual, limit=limit, rule=f"{metric} <= {limit:g}")
            )

    def at_least(metric: str, actual: float, limit: float) -> None:
        if actual < limit:
            found.append(
                Violation(metric=metric, actual=actual, limit=limit, rule=f"{metric} >= {limit:g}")
            )

    at_most("mismatch_fraction", stats.mismatch_fraction, rule.max_mismatch_fraction)
    at_least("cosine", stats.cosine, rule.min_cosine)
    at_least("top1_agreement", stats.top1_agreement, rule.min_top1_agreement)
    at_least("topk_overlap", stats.topk_overlap, rule.min_topk_overlap)
    limit = rule.max_perplexity_delta_pct
    if limit is not None and abs(stats.perplexity_delta_pct) > limit:
        # Two-sided: drifting "better" than the reference is still drifting (e.g. a
        # scale error makes the model overconfident and lowers perplexity).
        found.append(
            Violation(
                metric="perplexity_delta_pct",
                actual=stats.perplexity_delta_pct,
                limit=limit,
                rule=f"|perplexity_delta_pct| <= {limit:g}",
            )
        )
    return found


def compare_logits(
    reference: Matrix,
    candidate: Matrix,
    rule: ToleranceRule,
    *,
    model: str = "",
    dtype: str = "",
    policy_version: str = "",
    policy_fingerprint: str = "",
) -> ComparisonResult:
    stats = compute_stats(reference, candidate, rule)
    return ComparisonResult(
        model=model,
        dtype=dtype,
        stats=stats,
        rule=rule,
        violations=check(stats, rule),
        policy_version=policy_version,
        policy_fingerprint=policy_fingerprint,
    )


def merge(results: Sequence[ComparisonResult]) -> ComparisonResult:
    """Combine per-prompt results into one, keeping the worst value of each metric."""
    if not results:
        raise ValueError("nothing to merge")
    first = results[0]
    positions = sum(r.stats.positions for r in results)

    def weighted(name: str) -> float:
        total = sum(float(getattr(r.stats, name)) * r.stats.positions for r in results)
        return total / positions

    def geo_ppl(name: str) -> float:  # perplexity over all positions
        logs = sum(math.log(getattr(r.stats, name)) * r.stats.positions for r in results)
        return math.exp(logs / positions)

    stats = ComparisonStats(
        elements=sum(r.stats.elements for r in results),
        positions=positions,
        max_abs_err=max(r.stats.max_abs_err for r in results),
        max_rel_err=max(r.stats.max_rel_err for r in results),
        mismatch_fraction=sum(r.stats.mismatch_fraction * r.stats.elements for r in results)
        / sum(r.stats.elements for r in results),
        cosine=min(r.stats.cosine for r in results),
        top1_agreement=weighted("top1_agreement"),
        topk_overlap=weighted("topk_overlap"),
        perplexity_ref=geo_ppl("perplexity_ref"),
        perplexity_candidate=geo_ppl("perplexity_candidate"),
    )
    return first.model_copy(update={"stats": stats, "violations": check(stats, first.rule)})
