"""Typed configuration.

Resolution order (later wins):
  1. defaults on :class:`Settings`
  2. ``[tool.featureforge]`` in pyproject.toml
  3. environment variables (same names as the TypeScript engine where one existed)
  4. explicit overrides (CLI flags / pytest options)

TS equivalents: ``src/core/support/environments.ts`` plus the scattered
``process.env.*`` reads in world.ts, hooks.ts and tracer.ts.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from featureforge.core.tiers import TierSettings

LLMProvider = Literal["none", "openai", "ollama"]

_TRUE = {"1", "true", "yes", "on"}

DEFAULT_ENVIRONMENTS: dict[str, str] = {
    "qa": "https://qa.test.com/",
    "dev": "https://dev.test.com/",
    "staging": "https://staging.test.com/",
}


class LintSettings(BaseModel):
    max_scenarios_per_file: int = 10
    max_feature_name: int = 80
    max_scenario_name: int = 80
    max_step_length: int = 120
    max_background_steps: int = 8
    max_scenario_steps: int = 15
    restricted_tags: list[str] = Field(default_factory=lambda: ["@focus", "@skip"])


class Settings(BaseModel):
    env: str = "qa"
    environments: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_ENVIRONMENTS))

    # Browser
    headless: bool = True
    slow_mo: int = 0
    viewport_width: int = 1280
    viewport_height: int = 800
    browser: Literal["chromium", "firefox", "webkit"] = "chromium"

    # Observability
    enable_tracing: bool = False
    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    verbose: bool = False

    # AI
    llm_provider: LLMProvider = "none"
    llm_model: str = ""
    openai_api_key: str = Field(default="", repr=False)
    ollama_host: str = "http://localhost:11434"

    # Framework
    packs: list[str] = Field(default_factory=lambda: ["ai"])
    reports_dir: Path = Path("reports")
    features_dir: Path = Path("features")  # the project's own feature files
    steps_dir: Path = Path("steps")  # the project's own step definitions
    tolerance_policy: Path | None = None  # inference pack; None = the pack's tolerances.toml
    device_backend: str | None = None  # "module:factory" building a card backend
    tiers: TierSettings = Field(default_factory=TierSettings)

    # Run history & build manifest
    history_dir: Path = Path(".featureforge/history")
    history_keep: int = 200  # runs kept on disk
    manifest: dict[str, Any] = Field(default_factory=dict)  # static [tool.featureforge.manifest]
    manifest_file: Path | None = None  # JSON written by the build (FF_MANIFEST)
    perf_baselines: Path | None = None  # pinned perf numbers; None = the pack's baselines.toml

    # Release gates & escapes
    gates: dict[str, dict[str, Any]] = Field(default_factory=dict)  # [tool.featureforge.gates.*]
    escapes_file: Path = Path("quality/escapes.toml")
    project_root: Path = Path(".")
    lint: LintSettings = Field(default_factory=LintSettings)

    @field_validator("env")
    @classmethod
    def _lower_env(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def base_url(self) -> str:
        try:
            return self.environments[self.env]
        except KeyError:
            known = ", ".join(sorted(self.environments))
            raise ValueError(f"Unknown environment '{self.env}'. Known: {known}") from None

    def relative_to(self, root: Path) -> Settings:
        """Resolve every relative path setting against ``root`` (the project root), so the
        CLI and the pytest plugin agree no matter which folder they're started from."""
        base = root.resolve()
        update: dict[str, Any] = {"project_root": base}
        for key in PATH_FIELDS:
            value = getattr(self, key)
            if value is not None and not value.is_absolute():
                update[key] = (base / value).resolve()
        return self.model_copy(update=update)

    def gate_policy(self, name: str) -> Any:
        """The named release gate: built-in defaults overlaid with [tool.featureforge.gates]."""
        from featureforge.core.gate import DEFAULT_GATES, GatePolicy

        if name not in DEFAULT_GATES and name not in self.gates:
            known = ", ".join(sorted({*DEFAULT_GATES, *self.gates}))
            raise KeyError(f"Unknown gate '{name}'. Known: {known}")
        base = DEFAULT_GATES.get(name, GatePolicy()).model_dump()
        return GatePolicy.model_validate({**base, **self.gates.get(name, {})})

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {"openai": "gpt-4o", "ollama": "llama3.1", "none": ""}[self.llm_provider]

    # ── loading ─────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        pyproject: Path | None = None,
        environ: Mapping[str, str] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Settings:
        data: dict[str, Any] = {}
        if pyproject is not None and pyproject.is_file():
            data.update(_read_pyproject(pyproject))
        env = os.environ if environ is None else environ
        data.update(_from_environ(env))
        if env.get("FF_TIERS_AVAILABLE"):
            data["tiers"] = {**data.get("tiers", {}), "available": env["FF_TIERS_AVAILABLE"]}
        data.update({k: v for k, v in (overrides or {}).items() if v is not None})
        settings = cls.model_validate(data)
        if pyproject is not None and pyproject.is_file():
            settings = settings.relative_to(pyproject.parent)
        if (
            settings.llm_provider == "none"
            and settings.openai_api_key
            and ("llm_provider" not in data)
        ):
            # Same behaviour as the TS engine: a key in the environment turns OpenAI on.
            settings = settings.model_copy(update={"llm_provider": "openai"})
        return settings


PATH_FIELDS = (
    "reports_dir",
    "features_dir",
    "steps_dir",
    "history_dir",
    "escapes_file",
    "tolerance_policy",
    "manifest_file",
    "perf_baselines",
)


def find_pyproject(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def _read_pyproject(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    section = raw.get("tool", {}).get("featureforge", {})
    if not isinstance(section, dict):
        return {}
    return dict(section)  # paths are resolved against the project root in Settings.load


def _from_environ(environ: Mapping[str, str]) -> dict[str, Any]:
    def flag(name: str) -> bool | None:
        value = environ.get(name)
        return None if value is None else value.strip().lower() in _TRUE

    mapping: dict[str, Any] = {
        "env": environ.get("ENV"),
        "headless": flag("HEADLESS"),
        "slow_mo": environ.get("SLOW_MO"),
        "browser": environ.get("FF_BROWSER"),
        "enable_tracing": flag("ENABLE_TRACING"),
        "otlp_endpoint": environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
        "verbose": flag("VERBOSE"),
        "llm_provider": environ.get("FF_LLM_PROVIDER"),
        "llm_model": environ.get("FF_LLM_MODEL"),
        "openai_api_key": environ.get("OPENAI_API_KEY"),
        "ollama_host": environ.get("OLLAMA_HOST"),
        "reports_dir": environ.get("FF_REPORTS_DIR"),
        "packs": _split(environ.get("FF_PACKS")),
        "tolerance_policy": environ.get("FF_TOLERANCE_POLICY"),
        "device_backend": environ.get("FF_DEVICE_BACKEND"),
        "manifest_file": environ.get("FF_MANIFEST"),
        "history_dir": environ.get("FF_HISTORY_DIR"),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _split(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]
