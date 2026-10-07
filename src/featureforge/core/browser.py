"""Playwright lifecycle as pytest fixtures.

TS equivalents: ``launchBrowser`` in world.ts, the browser parts of the
``Before``/``After`` hooks in hooks.ts, and ``pageFixture.ts``.

The TS engine launched a browser for every scenario unless it was tagged
``@no-browser``. Here the browser is lazy: it starts only when a step function
asks for the ``page`` fixture, one browser per session, one context per scenario.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from featureforge.core.context import ScenarioContext
from featureforge.core.network import NetworkCapture
from featureforge.core.settings import Settings


@pytest.fixture(scope="session")
def ff_playwright() -> Iterator[Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover - depends on optional extra
        pytest.skip("Playwright is not installed: pip install 'featureforge-ai[ui]'")
    with sync_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="session")
def ff_browser(ff_playwright: Any, ff_settings: Settings) -> Iterator[Any]:
    launcher = getattr(ff_playwright, ff_settings.browser)
    browser = launcher.launch(headless=ff_settings.headless, slow_mo=ff_settings.slow_mo)
    if ff_settings.verbose:
        print(f"Browser launched ({ff_settings.browser}, headless={ff_settings.headless})")
    yield browser
    browser.close()


@pytest.fixture
def browser_context(ff_browser: Any, ff_settings: Settings) -> Iterator[Any]:
    context = ff_browser.new_context(
        base_url=ff_settings.base_url,
        viewport={"width": ff_settings.viewport_width, "height": ff_settings.viewport_height},
    )
    yield context
    context.close()


@pytest.fixture
def page(browser_context: Any, ctx: ScenarioContext) -> Iterator[Any]:
    """The Playwright page for this scenario; registers network capture on the context."""
    new_page = browser_context.new_page()
    ctx.page = new_page
    ctx.network = NetworkCapture(new_page)
    yield new_page
    ctx.page = None
