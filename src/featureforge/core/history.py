"""Run history: every run's metrics, kept with its build manifest.

TS equivalent: ``src/tools/reporting/archiveCucumberReports.ts``, which copied
each report with git metadata. Here the archive is queryable:

* **last known-good**: the newest run with nothing failed, the baseline for
  "what changed?" and for performance comparisons,
* **flakiness**: a test whose outcome flips on the *same* build is flaky; the
  same flip across builds is a regression,
* **trends**: pass rate and failures by stack layer over time.

Runs are JSON files under ``history_dir`` (default ``.featureforge/history``).
In CI, persist that folder between pipelines (cache or artifact).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from featureforge.core.metrics import Measurement, RunMetrics


@dataclass(frozen=True)
class FlakyTest:
    nodeid: str
    label: str
    runs: int
    flips: int  # outcome changes between consecutive runs of the same build
    passed_on_rerun: int

    @property
    def score(self) -> float:
        return (self.flips + self.passed_on_rerun) / max(self.runs, 1)


class RunHistory:
    def __init__(self, directory: Path, keep: int = 200) -> None:
        self.directory = directory
        self.keep = keep

    # ── storage ─────────────────────────────────────────────────────────────
    def archive(self, metrics: RunMetrics) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = re.sub(r"[^0-9T]", "", metrics.started_at)[:21]  # to the microsecond
        commit = metrics.manifest.git_commit[:8] if metrics.manifest else ""
        path = self.directory / f"run-{stamp}-{commit or 'nogit'}-{metrics.run_id}.json"
        metrics.write(path)
        self._prune()
        return path

    def _prune(self) -> None:
        files = sorted(self.directory.glob("run-*.json"))
        for stale in files[: max(len(files) - self.keep, 0)]:
            stale.unlink(missing_ok=True)

    def runs(self) -> list[RunMetrics]:
        """Oldest first. Unreadable files are skipped, not fatal."""
        if not self.directory.is_dir():
            return []
        loaded = []
        for path in sorted(self.directory.glob("run-*.json")):
            try:
                loaded.append(RunMetrics.read(path))
            except (ValueError, json.JSONDecodeError):
                continue
        return sorted(loaded, key=lambda r: r.started_at)

    # ── queries ─────────────────────────────────────────────────────────────
    def last_known_good(
        self, before: str | None = None, scope: str | None = None
    ) -> RunMetrics | None:
        """Newest run that executed something and had no failures or errors.

        With ``scope``, only runs that selected the same tests count: a green run of
        one feature folder says nothing about the rest of the suite.
        """
        for run in reversed(self.runs()):
            if before is not None and run.started_at >= before:
                continue
            if scope is not None and run.scope != scope:
                continue
            counts = run.counts
            if run.executed and counts["failed"] == 0 and counts["error"] == 0:
                return run
        return None

    def baseline_measurement(
        self, key: str, name: str, before: str | None = None
    ) -> tuple[Measurement, RunMetrics] | None:
        """The ``key``/``name`` measurement from the last known-good run that has it."""
        for run in reversed(self.runs()):
            if before is not None and run.started_at >= before:
                continue
            if run.counts["failed"] or run.counts["error"] or not run.executed:
                continue
            for scenario in run.scenarios:
                for measurement in scenario.measurements:
                    if measurement.key == key and measurement.name == name:
                        return measurement, run
        return None

    def flaky_tests(self, window: int = 20) -> list[FlakyTest]:
        """Tests whose outcome changed between runs of the same build, or passed on rerun."""
        recent = self.runs()[-window:]
        outcomes: dict[str, list[tuple[str, str]]] = defaultdict(list)
        labels: dict[str, str] = {}
        reruns: dict[str, int] = defaultdict(int)
        for run in recent:
            build = run.manifest.fingerprint if run.manifest else run.run_id
            for scenario in run.scenarios:
                if scenario.outcome == "skipped":
                    continue
                outcomes[scenario.nodeid].append((build, scenario.outcome))
                labels[scenario.nodeid] = scenario.label
                reruns[scenario.nodeid] += scenario.flaky
        found = []
        for nodeid, seen in outcomes.items():
            flips = sum(1 for (b1, o1), (b2, o2) in pairwise(seen) if b1 == b2 and o1 != o2)
            if flips or reruns[nodeid]:
                found.append(FlakyTest(nodeid, labels[nodeid], len(seen), flips, reruns[nodeid]))
        return sorted(found, key=lambda f: f.score, reverse=True)

    def newly_failing(self, current: RunMetrics) -> list[str]:
        """Labels of tests failing now that passed in the last known-good run of the same scope."""
        good = self.last_known_good(before=current.started_at, scope=current.scope)
        if good is None:
            return []
        passed_before = {s.nodeid for s in good.scenarios if s.outcome == "passed"}
        return [
            s.label
            for s in current.scenarios
            if s.outcome in ("failed", "error") and s.nodeid in passed_before
        ]
