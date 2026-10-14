"""Test tiers: what runs where, and how often.

Accelerators are scarce, so "run everything on every commit" isn't a strategy.
Each test belongs to one tier, ordered by cost:

    unit → sim → single_card → multi_card → rack

* Tag a scenario ``@tier-single_card`` (or mark a plain test
  ``@pytest.mark.tier_single_card``). Untagged scenarios get ``default``;
  untagged plain tests are ``unit``.
* ``available`` lists what this runner has. A selected test whose tier isn't
  available is *skipped with the reason*, never silently passed or failed.
* ``schedules`` name the tiers each trigger runs: ``--ff-schedule nightly``.

    [tool.featureforge.tiers]
    default = "sim"
    available = ["unit", "sim"]            # or FF_TIERS_AVAILABLE=unit,sim,single_card

    [tool.featureforge.tiers.schedules]
    commit = ["unit", "sim"]
    nightly = ["unit", "sim", "single_card", "multi_card"]
    release = ["unit", "sim", "single_card", "multi_card", "rack"]
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Tier(StrEnum):
    UNIT = "unit"
    SIM = "sim"
    SINGLE_CARD = "single_card"
    MULTI_CARD = "multi_card"
    RACK = "rack"

    @property
    def rank(self) -> int:
        return list(Tier).index(self)

    @property
    def marker(self) -> str:
        return f"tier_{self.value}"


def _all() -> list[Tier]:
    return list(Tier)


DEFAULT_SCHEDULES: dict[str, list[Tier]] = {
    "commit": [Tier.UNIT, Tier.SIM],
    "nightly": [Tier.UNIT, Tier.SIM, Tier.SINGLE_CARD, Tier.MULTI_CARD],
    "release": _all(),
}


class TierSettings(BaseModel):
    default: Tier = Tier.SIM
    available: list[Tier] = Field(default_factory=lambda: [Tier.UNIT, Tier.SIM])
    schedules: dict[str, list[Tier]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_SCHEDULES.items()}
    )

    @field_validator("available", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    def schedule(self, name: str) -> list[Tier]:
        try:
            return self.schedules[name]
        except KeyError:
            known = ", ".join(sorted(self.schedules))
            raise KeyError(f"Unknown schedule '{name}'. Known: {known}") from None


def parse_tiers(values: Iterable[str]) -> list[Tier]:
    """``["sim,single_card", "rack"]`` → tiers; accepts ``single-card`` too."""
    tiers: list[Tier] = []
    for value in values:
        for part in value.split(","):
            name = part.strip().replace("-", "_")
            if not name:
                continue
            try:
                tiers.append(Tier(name))
            except ValueError:
                known = ", ".join(t.value for t in Tier)
                raise ValueError(f"Unknown tier '{part.strip()}'. Known: {known}") from None
    return list(dict.fromkeys(tiers))


def tier_from_tags(tags: Iterable[str]) -> Tier | None:
    """The tier named by ``@tier-<name>`` / ``tier_<name>`` tags; the costliest if several."""
    found = []
    for tag in tags:
        name = tag.lstrip("@").replace("-", "_")
        if name.startswith("tier_"):
            try:
                found.append(Tier(name.removeprefix("tier_")))
            except ValueError:
                known = ", ".join(t.value for t in Tier)
                raise ValueError(f"Unknown tier tag '{tag}'. Known tiers: {known}") from None
    return max(found, key=lambda t: t.rank) if found else None
