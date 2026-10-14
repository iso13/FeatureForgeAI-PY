"""Build manifest: exactly what was under test.

Each run records the version of every stack component (firmware, compiler,
runtime, serving …), the hardware it ran on and the git commit. Comparing it
with the last known-good run's manifest answers "what changed?", which is the
fastest route to *which layer* broke.

Sources, later wins:

1. ``[tool.featureforge.manifest]`` in pyproject.toml (static defaults)
2. a JSON file: ``manifest_file`` setting or ``FF_MANIFEST`` (written by the build)
3. ``FF_COMPONENT_<NAME>=<version>`` environment variables, e.g.
   ``FF_COMPONENT_COMPILER=0.9.2``; ``FF_HARDWARE=<board id>``
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BuildManifest(BaseModel):
    components: dict[str, str] = Field(default_factory=dict)
    hardware: str = ""
    git_commit: str = ""
    git_branch: str = ""

    @property
    def fingerprint(self) -> str:
        """Identifies the software under test (components + commit), not the board."""
        payload = json.dumps({"c": self.components, "g": self.git_commit}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def changes_since(self, other: BuildManifest) -> dict[str, tuple[str, str]]:
        """Component → (other's version, this version) for everything that differs."""
        names = sorted(set(self.components) | set(other.components))
        changes = {
            name: (other.components.get(name, "∅"), self.components.get(name, "∅"))
            for name in names
            if other.components.get(name) != self.components.get(name)
        }
        if self.hardware and other.hardware and self.hardware != other.hardware:
            changes["board"] = (other.hardware, self.hardware)
        return changes

    @classmethod
    def collect(
        cls,
        static: Mapping[str, Any] | None = None,
        manifest_file: Path | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> BuildManifest:
        env = environ or {}
        data: dict[str, Any] = {"components": {}}
        for source in (dict(static or {}), _read_json(manifest_file)):
            data["components"].update({k: str(v) for k, v in source.get("components", {}).items()})
            for key in ("hardware", "git_commit", "git_branch"):
                if source.get(key):
                    data[key] = str(source[key])
        for key, value in env.items():
            if key.startswith("FF_COMPONENT_") and value:
                data["components"][key.removeprefix("FF_COMPONENT_").lower()] = value
        if env.get("FF_HARDWARE"):
            data["hardware"] = env["FF_HARDWARE"]
        if not data.get("git_commit"):
            data["git_commit"] = _git("rev-parse", "--short=12", "HEAD", cwd=cwd)
            data["git_branch"] = data.get("git_branch") or _git(
                "rev-parse", "--abbrev-ref", "HEAD", cwd=cwd
            )
        return cls.model_validate(data)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""
