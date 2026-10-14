"""Run metrics — "quality measured, not assumed".

Every run writes ``reports/run-metrics.json``: outcomes, pass rate, duration,
and failures grouped by the root-cause layer from the Step Intelligence Layer.
Archiving that file per CI run gives trend data (pass rate, layer mix, slowest
scenarios) without any extra service. The TS engine only had a Prometheus
duration histogram (``src/core/support/metrics.ts``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from featureforge.core.intelligence.models import utc_now


class Measurement(BaseModel):
    """A number a scenario produced (latency, accuracy …), kept for trends and baselines."""

    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = False


class ScenarioResult(BaseModel):
    nodeid: str
    feature: str
    scenario: str
    example: str = ""  # Examples row id for outline scenarios
    tags: list[str] = Field(default_factory=list)
    outcome: str  # passed | failed | skipped | error
    tier: str | None = None
    duration_s: float = 0.0
    failed_step: str | None = None
    root_cause: str | None = None
    brief_id: str | None = None
    missing_step: str | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    stack_layer: str | None = None  # FIRMWARE | COMPILER | RUNTIME | SERVING | HARDWARE …
    stack_confidence: float | None = None
    stack_method: str | None = None
    stack_evidence: list[str] = Field(default_factory=list)
    attempts: int = 1
    flaky: bool = False  # failed, then passed on rerun

    @property
    def label(self) -> str:
        return f"{self.scenario} [{self.example}]" if self.example else self.scenario


class RunMetrics(BaseModel):
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None
    environment: str = ""
    scenarios: list[ScenarioResult] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counter = Counter(result.outcome for result in self.scenarios)
        return {key: counter.get(key, 0) for key in ("passed", "failed", "error", "skipped")}

    @property
    def executed(self) -> int:
        c = self.counts
        return c["passed"] + c["failed"] + c["error"]

    @property
    def pass_rate(self) -> float | None:
        return None if self.executed == 0 else round(self.counts["passed"] / self.executed, 4)

    @property
    def failures_by_layer(self) -> dict[str, int]:
        """Application layers (UI, API …); stack failures are in failures_by_stack_layer."""
        counter = Counter(
            result.root_cause or ("MISSING_STEP" if result.missing_step else "UNCLASSIFIED")
            for result in self.scenarios
            if result.outcome in ("failed", "error")
            and not (result.stack_layer and result.root_cause in (None, "UNKNOWN"))
        )
        return dict(counter.most_common())

    @property
    def failures_by_stack_layer(self) -> dict[str, int]:
        counter = Counter(
            result.stack_layer
            for result in self.scenarios
            if result.stack_layer and (result.outcome in ("failed", "error") or result.flaky)
        )
        return dict(counter.most_common())

    @property
    def flaky(self) -> list[ScenarioResult]:
        return [result for result in self.scenarios if result.flaky]

    @property
    def by_tier(self) -> dict[str, dict[str, int]]:
        """Outcome counts per tier, cheapest tier first."""
        from featureforge.core.tiers import Tier

        order = {t.value: t.rank for t in Tier}
        table: dict[str, Counter[str]] = {}
        for result in self.scenarios:
            table.setdefault(result.tier or "untiered", Counter())[result.outcome] += 1
        return {
            tier: dict(counts)
            for tier, counts in sorted(table.items(), key=lambda kv: order.get(kv[0], 99))
        }

    def slowest(self, limit: int = 5) -> list[ScenarioResult]:
        return sorted(self.scenarios, key=lambda r: r.duration_s, reverse=True)[:limit]

    def to_json(self) -> str:
        payload = self.model_dump(mode="json")
        payload["summary"] = {
            "counts": self.counts,
            "executed": self.executed,
            "pass_rate": self.pass_rate,
            "total_duration_s": round(sum(r.duration_s for r in self.scenarios), 3),
            "failures_by_layer": self.failures_by_layer,
            "by_tier": self.by_tier,
            "failures_by_stack_layer": self.failures_by_stack_layer,
            "flaky": len(self.flaky),
        }
        return json.dumps(payload, indent=2)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> RunMetrics:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("summary", None)
        return cls.model_validate(data)

    def render(self) -> str:
        c = self.counts
        rate = "n/a" if self.pass_rate is None else f"{self.pass_rate:.1%}"
        lines = [
            f"Scenarios: {len(self.scenarios)}  passed {c['passed']}  failed {c['failed']}  "
            f"error {c['error']}  skipped {c['skipped']}  pass rate {rate}",
        ]
        if any(r.tier for r in self.scenarios):
            tiers = "; ".join(
                f"{tier} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                for tier, counts in self.by_tier.items()
            )
            lines.append(f"By tier: {tiers}")
        if self.failures_by_stack_layer:
            layers = ", ".join(f"{k}={v}" for k, v in self.failures_by_stack_layer.items())
            lines.append(f"Failures by stack layer: {layers}")
        if self.failures_by_layer:
            layers = ", ".join(f"{k}={v}" for k, v in self.failures_by_layer.items())
            lines.append(f"Failures by root-cause layer: {layers}")
        if self.flaky:
            lines.append(f"Flaky (passed on rerun): {', '.join(r.label for r in self.flaky)}")
        return "\n".join(lines)
