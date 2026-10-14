"""Steps for features/Sample.feature. A template: copy it, then delete it."""

from __future__ import annotations

from dataclasses import dataclass, field

from pytest_bdd import given, parsers, then, when

from featureforge.core.context import ScenarioContext
from steps.helpers import Counter


@dataclass
class SampleState:
    counter: Counter = field(default_factory=Counter)


@given(parsers.parse("a counter starting at {start:d}"))
def counter_starts_at(ctx: ScenarioContext, start: int) -> None:
    ctx.state(SampleState).counter = Counter(start)


@when(parsers.parse("I add {amount:d} to the counter"))
def add_to_counter(ctx: ScenarioContext, amount: int) -> None:
    ctx.state(SampleState).counter.add(amount)


@then(parsers.parse("the counter should read {expected:d}"))
def counter_reads(ctx: ScenarioContext, expected: int) -> None:
    actual = ctx.state(SampleState).counter.value
    assert actual == expected, f"Expected counter {expected}, got {actual}"
