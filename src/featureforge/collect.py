"""Where scenarios and step definitions come from.

Three sources, in this order:

1. every enabled pack's ``features/`` directory (shipped with the pack),
2. the project's own ``features/`` folder (``features_dir`` setting), with
   subfolders by domain, e.g. ``features/compiler/Lowering.feature``,
3. step definitions in the project's ``steps/`` folder (``steps_dir``), loaded
   automatically as pytest plugins, so a new steps file needs no registration.
   Helpers in that folder import normally: ``from steps.helpers import Board``.

TS equivalent: ``paths`` and ``import`` in cucumber.mjs.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from featureforge.core.settings import Settings, find_pyproject


def project_settings(start: Path | None = None) -> Settings:
    """Settings for the project that contains ``start`` (default: the cwd)."""
    return Settings.load(find_pyproject(start or Path.cwd()))


def feature_paths(settings: Settings | None = None) -> list[str]:
    """Absolute feature directories for ``pytest_bdd.scenarios(*feature_paths())``."""
    from featureforge.packs import get_pack

    settings = settings or project_settings()
    paths = [get_pack(name).features_dir for name in settings.packs]
    project = settings.features_dir
    if project.is_dir() and any(project.rglob("*.feature")):
        paths.append(project)
    return [str(path.resolve()) for path in paths]


def step_modules(steps_dir: Path) -> list[str]:
    """Dotted module names for every ``*.py`` file under ``steps_dir``."""
    if not steps_dir.is_dir():
        return []
    names = []
    for path in sorted(steps_dir.rglob("*.py")):
        relative = path.relative_to(steps_dir.parent).with_suffix("")
        parts = relative.parts
        if parts[-1] == "__init__" or "__pycache__" in parts:
            continue
        names.append(".".join(parts))
    return names


def load_project_steps(plugin_manager: Any, steps_dir: Path) -> list[str]:
    """Import the project's step modules as pytest plugins; return their names."""
    modules = step_modules(steps_dir)
    if not modules:
        return []
    root = str(steps_dir.parent.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    _check_package_origin(steps_dir)
    for module in modules:
        plugin_manager.import_plugin(module)
    return modules


def _check_package_origin(steps_dir: Path) -> None:
    """Fail clearly if another package with the same name was imported first."""
    existing = sys.modules.get(steps_dir.name)
    if existing is None:
        return
    locations = [Path(p).resolve() for p in getattr(existing, "__path__", [])]
    if steps_dir.resolve() not in locations:
        found = ", ".join(map(str, locations)) or "an unknown location"
        raise ImportError(
            f"A different '{steps_dir.name}' package is already imported (from {found}). "
            f"Rename the project's steps folder or set steps_dir in [tool.featureforge]."
        )


def scenario_feature_file(item: Any) -> Path | None:
    """The ``.feature`` file a collected pytest-bdd item came from, if any."""
    scenario = getattr(getattr(item, "function", None), "__scenario__", None)
    filename = getattr(getattr(scenario, "feature", None), "filename", None)
    return Path(filename).resolve() if filename else None


def is_selected(feature_file: Path, selections: Iterable[Path]) -> bool:
    """True when ``feature_file`` is one of ``selections`` or sits under one of them."""
    for selected in selections:
        if feature_file == selected or selected in feature_file.parents:
            return True
    return False
