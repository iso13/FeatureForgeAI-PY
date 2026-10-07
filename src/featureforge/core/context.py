"""Per-scenario state.

TS equivalent: ``CustomWorld`` in ``src/core/support/world.ts``. That interface had
grown to ~60 optional fields (RAG, agents, HIPAA, fairness, EMR …), most typed
``any``, all on one object. Here the core context only holds what every scenario
needs; each pack keeps its own typed state class and asks for it with
:meth:`ScenarioContext.state`::

    @dataclass
    class FairnessState:
        model: Model | None = None
        output: float | None = None

    ctx.state(FairnessState).output = 0.5
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from featureforge.core.settings import Settings

if TYPE_CHECKING:
    from featureforge.core.intelligence.models import DeveloperBrief
    from featureforge.core.network import NetworkCapture
    from featureforge.core.telemetry import SpanLike

T = TypeVar("T")

_EXTENSIONS = {
    "text/plain": "txt",
    "text/html": "html",
    "application/json": "json",
    "image/png": "png",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "unnamed"


@dataclass(frozen=True)
class Attachment:
    name: str
    mime: str
    path: Path


@dataclass
class ScenarioContext:
    settings: Settings
    feature_name: str = ""
    scenario_name: str = ""
    tags: frozenset[str] = frozenset()
    variant: str = ""  # Examples row id for outline scenarios, e.g. "male"
    current_step: str = ""
    step_index: int = 0

    page: Any = None  # playwright.sync_api.Page when a step requested one
    network: NetworkCapture | None = None
    brief: DeveloperBrief | None = None
    scenario_span: SpanLike | None = None
    step_span: SpanLike | None = None

    attachments: list[Attachment] = field(default_factory=list)
    _states: dict[type[Any], Any] = field(default_factory=dict, repr=False)

    # ── typed pack state ────────────────────────────────────────────────────
    def state(self, cls: type[T]) -> T:
        """Return this scenario's instance of ``cls``, creating it on first use."""
        existing = self._states.get(cls)
        if existing is None:
            existing = cls()
            self._states[cls] = existing
        return existing

    def has_tag(self, tag: str) -> bool:
        return tag.lstrip("@") in self.tags

    # ── attachments (TS: this.attach(body, mime)) ───────────────────────────
    @property
    def slug(self) -> str:
        base = f"{slugify(self.feature_name)}__{slugify(self.scenario_name)}"
        return f"{base}__{slugify(self.variant)}" if self.variant else base

    @property
    def attachments_dir(self) -> Path:
        return self.settings.reports_dir / "attachments" / self.slug

    def attach(self, body: str | bytes, mime: str = "text/plain", name: str = "") -> Attachment:
        directory = self.attachments_dir
        directory.mkdir(parents=True, exist_ok=True)
        index = len(self.attachments) + 1
        stem = slugify(name) if name else f"attachment_{index}"
        path = directory / f"{index:02d}_{stem}.{_EXTENSIONS.get(mime, 'bin')}"
        if isinstance(body, bytes):
            path.write_bytes(body)
        else:
            path.write_text(body, encoding="utf-8")
        attachment = Attachment(name=name or stem, mime=mime, path=path)
        self.attachments.append(attachment)
        return attachment
