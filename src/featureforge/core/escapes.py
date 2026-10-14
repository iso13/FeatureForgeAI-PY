"""Escape log: defects found after a release, and the test that now catches them.

Every escape is a missing test. Recording which layer it was in and which tier
*should* have caught it shows where coverage is thin; the release gate can
refuse to ship while an escape still has no regression test.

    # quality/escapes.toml
    [[escape]]
    id = "ESC-014"
    title = "int8 logits drift on prompts over 2k tokens after compiler 0.9.2"
    found = "2026-09-30"
    severity = "high"              # critical | high | medium | low
    layer = "COMPILER"             # stack layer it lived in
    should_have_caught = "sim"     # tier that should have caught it
    regression_test = "features/compiler/LongPrompt.feature"   # "" until written
"""

from __future__ import annotations

import tomllib
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from featureforge.core.stack import StackLayer
from featureforge.core.tiers import Tier

Severity = Literal["critical", "high", "medium", "low"]


class Escape(BaseModel):
    id: str
    title: str
    found: date
    severity: Severity = "medium"
    layer: StackLayer = StackLayer.UNKNOWN
    should_have_caught: Tier | None = None
    regression_test: str = ""

    def has_test(self, root: Path) -> bool:
        return bool(self.regression_test) and (root / self.regression_test).exists()


class EscapeLog(BaseModel):
    escapes: list[Escape] = Field(default_factory=list)
    root: Path = Path(".")

    @classmethod
    def load(cls, path: Path, root: Path | None = None) -> EscapeLog:
        """``root``: where ``regression_test`` paths are relative to (the project root)."""
        base = root or path.parent
        if not path.is_file():
            return cls(root=base)
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls(escapes=raw.get("escape", []), root=base)

    @property
    def without_test(self) -> list[Escape]:
        return [e for e in self.escapes if not e.has_test(self.root)]

    def by_layer(self) -> dict[str, int]:
        return dict(Counter(e.layer.value for e in self.escapes).most_common())

    def by_tier(self) -> dict[str, int]:
        counts = Counter(
            e.should_have_caught.value if e.should_have_caught else "unassigned"
            for e in self.escapes
        )
        return dict(counts.most_common())

    def render(self) -> str:
        if not self.escapes:
            return "No escapes recorded."
        covered = len(self.escapes) - len(self.without_test)
        lines = [
            f"Escapes: {len(self.escapes)}  with a regression test: {covered} "
            f"({covered / len(self.escapes):.0%})",
            "By layer: " + ", ".join(f"{k}={v}" for k, v in self.by_layer().items()),
            "Should have been caught in tier: "
            + ", ".join(f"{k}={v}" for k, v in self.by_tier().items()),
        ]
        for escape in sorted(self.escapes, key=lambda e: e.found, reverse=True):
            status = "covered" if escape.has_test(self.root) else "NO TEST"
            lines.append(
                f"  {escape.id:<9} {escape.found}  {escape.severity:<8} {escape.layer.value:<14} "
                f"{status:<8} {escape.title}"
            )
        return "\n".join(lines)
