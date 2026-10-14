"""Pack registry: the plugin layer of the platform.

A pack is a Python package that ships ``.feature`` files plus a pytest plugin
module holding its step definitions and fixtures. Built-in packs live under
``featureforge.packs``; customer or domain packs (``implementations/{customer}``
in the TS restructure plan) are separate distributions registered through the
``featureforge.packs`` entry-point group, so private code never has to live in
the framework repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path


@dataclass(frozen=True)
class Pack:
    name: str
    description: str
    features_dir: Path
    plugin: str  # dotted module path of the pytest plugin with the step definitions


BUILTIN_PACKS = {
    "ai": "featureforge.packs.ai:PACK",
    "inference": "featureforge.packs.inference:PACK",
}


def _load(reference: str) -> Pack:
    module_name, _, attribute = reference.partition(":")
    pack = getattr(import_module(module_name), attribute or "PACK")
    if not isinstance(pack, Pack):
        raise TypeError(f"{reference} does not point to a featureforge Pack")
    return pack


@cache
def discover_packs() -> dict[str, Pack]:
    references = dict(BUILTIN_PACKS)
    for ep in entry_points(group="featureforge.packs"):
        references[ep.name] = ep.value
    return {name: _load(reference) for name, reference in sorted(references.items())}


def get_pack(name: str) -> Pack:
    packs = discover_packs()
    try:
        return packs[name]
    except KeyError:
        raise KeyError(f"Unknown pack '{name}'. Installed: {', '.join(packs)}") from None
