import os

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _isolated_featureforge_env(monkeypatch):
    """Tests describe a plain runner; FF_* settings from the outer shell (e.g. a hardware
    runner's FF_TIERS_AVAILABLE) must not leak into them or their child sessions."""
    for name in list(os.environ):
        if name.startswith("FF_"):
            monkeypatch.delenv(name)
