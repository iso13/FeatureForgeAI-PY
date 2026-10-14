"""Release gates: what blocks a release, stated in measurable terms.

``featureforge gate --policy release`` reads the last run's metrics, the run
history and the escape log, and answers PASS or BLOCK with every check, the
measured value and the limit. CI uses the exit code; people read the table.

    [tool.featureforge.gates.release]
    min_pass_rate = 1.0
    required_tiers = ["unit", "sim", "single_card", "multi_card", "rack"]
    max_new_failures = 0
    max_flaky = 0
    max_perf_regressions = 0
    max_escapes_without_test = 0
    required_components = ["firmware", "compiler", "runtime", "serving"]

    # A known failure may be waived, with a ticket and an expiry date.
    [[tool.featureforge.gates.release.quarantine]]
    match = "Every card in the node"
    ticket = "HW-231"
    expires = 2026-11-15
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from featureforge.core.escapes import EscapeLog
from featureforge.core.history import RunHistory
from featureforge.core.metrics import RunMetrics, ScenarioResult
from featureforge.core.tiers import Tier


class Quarantine(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo must not silently drop a check

    match: str  # substring of the scenario label or node id
    ticket: str
    expires: date

    def covers(self, result: ScenarioResult) -> bool:
        return self.match in result.label or self.match in result.nodeid


class GatePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo must not silently drop a check

    min_pass_rate: float = 1.0
    required_tiers: list[Tier] = Field(default_factory=list)  # each must run, none skipped
    max_new_failures: int | None = 0  # failing now, passed in the last known-good run
    max_flaky: int | None = None
    max_perf_regressions: int | None = 0
    max_escapes_without_test: int | None = None
    required_components: list[str] = Field(default_factory=list)  # must be in the manifest
    quarantine: list[Quarantine] = Field(default_factory=list)


DEFAULT_GATES: dict[str, GatePolicy] = {
    "commit": GatePolicy(required_tiers=[Tier.UNIT, Tier.SIM]),
    "nightly": GatePolicy(
        required_tiers=[Tier.UNIT, Tier.SIM, Tier.SINGLE_CARD, Tier.MULTI_CARD], max_flaky=2
    ),
    "release": GatePolicy(
        required_tiers=list(Tier),
        max_flaky=0,
        max_escapes_without_test=0,
        required_components=["firmware", "compiler", "runtime", "serving"],
    ),
}


class Check(BaseModel):
    name: str
    passed: bool
    measured: str
    limit: str
    detail: str = ""


class GateResult(BaseModel):
    policy: str
    passed: bool
    checks: list[Check]
    run_id: str = ""
    commit: str = ""

    @property
    def decision(self) -> str:
        return "PASS" if self.passed else "BLOCK"

    def render(self) -> str:
        head = f"Release gate '{self.policy}': {self.decision}"
        if self.commit:
            head += f"  (run {self.run_id}, commit {self.commit})"
        lines = [head]
        for c in self.checks:
            mark = "ok " if c.passed else "✗  "
            lines.append(f"  {mark}{c.name:<26} {c.measured:<22} limit {c.limit}")
            if c.detail and not c.passed:
                lines.append(f"       {c.detail}")
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json") | {"decision": self.decision}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def _names(results: list[ScenarioResult], limit: int = 5) -> str:
    shown = ", ".join(r.label for r in results[:limit])
    return shown + (f" (+{len(results) - limit} more)" if len(results) > limit else "")


def evaluate(
    name: str,
    policy: GatePolicy,
    metrics: RunMetrics,
    history: RunHistory | None = None,
    escapes: EscapeLog | None = None,
    today: date | None = None,
) -> GateResult:
    today = today or date.today()
    checks: list[Check] = []
    failing = [r for r in metrics.scenarios if r.outcome in ("failed", "error")]

    # Quarantine: waived failures don't count, expired waivers do and are flagged.
    waived: list[ScenarioResult] = []
    expired: list[str] = []
    for result in failing:
        entry = next((q for q in policy.quarantine if q.covers(result)), None)
        if entry is None:
            continue
        if entry.expires >= today:
            waived.append(result)
        else:
            expired.append(f"{entry.ticket} expired {entry.expires} ({result.label})")
    counted = [r for r in failing if r not in waived]

    executed = [r for r in metrics.scenarios if r.outcome != "skipped" and r not in waived]
    passed = sum(r.outcome == "passed" for r in executed)
    rate = passed / len(executed) if executed else 0.0
    checks.append(
        Check(
            name="pass rate",
            passed=bool(executed) and rate >= policy.min_pass_rate,
            measured=f"{rate:.1%} of {len(executed)}"
            + (f", {len(waived)} waived" if waived else ""),
            limit=f">= {policy.min_pass_rate:.0%}",
            detail=f"failing: {_names(counted)}" if counted else "nothing executed",
        )
    )
    if policy.quarantine:
        checks.append(
            Check(
                name="quarantine",
                passed=not expired,
                measured=f"{len(waived)} waived, {len(expired)} expired",
                limit="no expired waivers",
                detail="; ".join(expired),
            )
        )

    by_tier = metrics.by_tier
    for tier in policy.required_tiers:
        counts = by_tier.get(tier.value, {})
        ran = sum(v for k, v in counts.items() if k != "skipped")
        skipped = counts.get("skipped", 0)
        checks.append(
            Check(
                name=f"tier {tier.value}",
                passed=ran > 0 and skipped == 0,
                measured=f"{ran} ran, {skipped} skipped",
                limit=">= 1 ran, 0 skipped",
                detail="no scenarios ran in this tier" if ran == 0 else "scenarios were skipped",
            )
        )

    if policy.max_new_failures is not None and history is not None:
        new = [
            label
            for label in history.newly_failing(metrics)
            if label not in {r.label for r in waived}
        ]
        checks.append(
            Check(
                name="new failures vs last good",
                passed=len(new) <= policy.max_new_failures,
                measured=str(len(new)),
                limit=f"<= {policy.max_new_failures}",
                detail=", ".join(new[:5]),
            )
        )

    if policy.max_flaky is not None:
        flaky = metrics.flaky
        checks.append(
            Check(
                name="flaky (passed on rerun)",
                passed=len(flaky) <= policy.max_flaky,
                measured=str(len(flaky)),
                limit=f"<= {policy.max_flaky}",
                detail=_names(flaky),
            )
        )

    if policy.max_perf_regressions is not None:
        regressions = metrics.regressions
        checks.append(
            Check(
                name="perf regressions",
                passed=len(regressions) <= policy.max_perf_regressions,
                measured=str(len(regressions)),
                limit=f"<= {policy.max_perf_regressions}",
                detail="; ".join(
                    f"{m.name} {m.delta_pct:+.1f}% in {r.label}" for r, m in regressions[:5]
                ),
            )
        )

    if policy.max_escapes_without_test is not None:
        open_escapes = escapes.without_test if escapes is not None else []
        checks.append(
            Check(
                name="escapes without a test",
                passed=len(open_escapes) <= policy.max_escapes_without_test,
                measured=str(len(open_escapes)),
                limit=f"<= {policy.max_escapes_without_test}",
                detail=", ".join(f"{e.id} ({e.title})" for e in open_escapes[:5]),
            )
        )

    if policy.required_components:
        recorded = metrics.manifest.components if metrics.manifest else {}
        missing = [c for c in policy.required_components if c not in recorded]
        checks.append(
            Check(
                name="build manifest",
                passed=not missing,
                measured=f"{len(policy.required_components) - len(missing)}"
                f"/{len(policy.required_components)} recorded",
                limit="all components versioned",
                detail="missing: " + ", ".join(missing) if missing else "",
            )
        )

    return GateResult(
        policy=name,
        passed=all(c.passed for c in checks),
        checks=checks,
        run_id=metrics.run_id,
        commit=metrics.manifest.git_commit if metrics.manifest else "",
    )
