"""Steps every pack can use. TS equivalent: ``src/core/steps/env.steps.ts``."""

from __future__ import annotations

from typing import Any

from pytest_bdd import given, parsers, then

from featureforge.core.settings import Settings


@given("I navigate to homepage")
def navigate_home(page: Any, ff_settings: Settings) -> None:
    page.goto(ff_settings.base_url)
    page.wait_for_load_state("domcontentloaded")


@then(parsers.parse('the page title should contain "{text}"'))
def title_contains(page: Any, text: str) -> None:
    assert text in page.title(), f'Expected title to contain "{text}", got "{page.title()}"'
