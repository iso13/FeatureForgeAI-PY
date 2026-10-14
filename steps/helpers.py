"""Shared helpers for project steps: import them as ``from steps.helpers import …``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Counter:
    value: int = 0

    def add(self, amount: int) -> int:
        self.value += amount
        return self.value
