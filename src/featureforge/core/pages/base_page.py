"""Page object base. TS equivalent: ``src/core/pages/BasePage.ts``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BasePage(ABC):
    def __init__(self, page: Any) -> None:
        self.page = page

    @property
    @abstractmethod
    def path(self) -> str:
        """Path relative to the environment base URL."""

    def open(self) -> None:
        self.page.goto(self.path)

    def wait_for_load(self) -> None:
        self.page.wait_for_load_state("networkidle")

    def is_visible(self, locator: Any) -> bool:
        return bool(locator.is_visible())

    @property
    def title(self) -> str:
        return str(self.page.title())

    def capture_screenshot(self, name: str, directory: Path = Path("reports/screenshots")) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{name}.png"
        self.page.screenshot(path=str(target), full_page=True)
        return target
