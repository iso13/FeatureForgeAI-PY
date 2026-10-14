"""Performance regression checks: latency and throughput are correctness criteria.

A measurement is compared with up to two baselines:

* **pinned**: a reviewed number checked into the repo (e.g. the last release),
* **last-known-good**: the same measurement from the newest green run.

It regresses if it is worse than *either* by more than its budget. Checking the
pinned value stops slow creep: against last-known-good alone, a 3% slip per
week never trips a 10% budget, because each green week becomes the baseline.
"""

from __future__ import annotations

import math
import statistics
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from featureforge.core.metrics import Measurement


@dataclass(frozen=True)
class LatencySummary:
    count: int
    mean: float
    p50: float
    p90: float
    p99: float

    @classmethod
    def of(cls, samples: Sequence[float]) -> LatencySummary:
        if not samples:
            raise ValueError("no latency samples")
        ordered = sorted(samples)
        return cls(
            count=len(ordered),
            mean=statistics.fmean(ordered),
            p50=percentile(ordered, 50),
            p90=percentile(ordered, 90),
            p99=percentile(ordered, 99),
        )


def percentile(ordered: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of already-sorted samples (no interpolation surprises)."""
    rank = max(math.ceil(pct / 100 * len(ordered)), 1)
    return float(ordered[rank - 1])


@dataclass(frozen=True)
class Baseline:
    value: float
    source: str  # "pinned" | "last-known-good"
    detail: str = ""  # e.g. the run or file it came from


def delta_pct(value: float, baseline: float, higher_is_better: bool) -> float:
    """Signed % change where positive always means *worse*."""
    if baseline == 0:
        return 0.0 if value == 0 else math.inf
    change = (value - baseline) / abs(baseline) * 100
    return -change if higher_is_better else change


def judge(
    measurement: Measurement, baselines: Sequence[Baseline], budget_pct: float
) -> Measurement:
    """Return ``measurement`` annotated with its worst comparison against ``baselines``."""
    if not baselines:
        return measurement.model_copy(update={"budget_pct": budget_pct})
    worst = max(
        baselines, key=lambda b: delta_pct(measurement.value, b.value, measurement.higher_is_better)
    )
    change = delta_pct(measurement.value, worst.value, measurement.higher_is_better)
    return measurement.model_copy(
        update={
            "baseline": worst.value,
            "baseline_source": worst.source,
            "delta_pct": round(change, 3),
            "budget_pct": budget_pct,
            "regressed": change > budget_pct,
        }
    )


class PinnedBaselines:
    """Reviewed baseline numbers from a TOML file::

    ["toy-llm/int8"]
    ttft_p50_ms = 16.4
    throughput_tok_s = 290.0
    """

    def __init__(self, values: dict[str, dict[str, float]], source: str = "") -> None:
        self.values = values
        self.source = source

    @classmethod
    def load(cls, path: Path) -> PinnedBaselines:
        if not path.is_file():
            return cls({}, str(path))
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        values = {
            str(key): {str(k): float(v) for k, v in table.items()}
            for key, table in raw.items()
            if isinstance(table, dict)
        }
        return cls(values, str(path))

    def get(self, key: str, name: str) -> float | None:
        return self.values.get(key, {}).get(name)
