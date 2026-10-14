"""Tolerance policy: what "close enough" means, written down and versioned.

Quantized inference is not bit-identical to a reference, so correctness is a
set of numeric limits, not ``==``. The limits live in a TOML file reviewed like
code. Each run records the policy's version and fingerprint next to its
results, so a loosened tolerance is visible in history instead of silently
turning a red run green.

Lookup order for ``rule_for(model, dtype)``, most specific first:
``"model/dtype"`` → ``"*/dtype"`` → ``"model/*"`` → ``[defaults]``.
Each rule only lists the limits it changes; the rest come from the defaults.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_POLICY_PATH = Path(__file__).parent / "tolerances.toml"


class ToleranceRule(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo must not silently drop a check

    """Limits for one model/precision pair. Element checks follow ``numpy.allclose``."""

    rtol: float = Field(default=1e-5, ge=0, description="relative tolerance per element")
    atol: float = Field(default=1e-6, ge=0, description="absolute tolerance per element")
    max_mismatch_fraction: float = Field(
        default=0.0, ge=0, le=1, description="share of elements allowed outside atol + rtol*|ref|"
    )
    min_cosine: float = Field(default=0.9999, le=1, description="cosine similarity of all logits")
    min_top1_agreement: float = Field(
        default=1.0, ge=0, le=1, description="argmax matches per position"
    )
    topk: int = Field(default=5, ge=1)
    min_topk_overlap: float = Field(
        default=1.0, ge=0, le=1, description="mean |topk ∩ topk_ref| / k"
    )
    max_perplexity_delta_pct: float | None = Field(
        default=None, description="|% change| in perplexity of the reference's own tokens"
    )
    rationale: str = ""


class TolerancePolicy(BaseModel):
    version: str
    owner: str = ""
    reference: str = ""  # what the candidate is compared against, e.g. "fp32 on CPU"
    defaults: ToleranceRule = Field(default_factory=ToleranceRule)
    rules: dict[str, dict[str, Any]] = Field(default_factory=dict)
    source: str = ""

    def rule_for(self, model: str, dtype: str) -> ToleranceRule:
        merged = self.defaults.model_dump()
        for key in (f"{model}/*", f"*/{dtype}", f"{model}/{dtype}"):  # least → most specific
            merged.update(self.rules.get(key, {}))
        return ToleranceRule.model_validate(merged)

    def has_rule(self, model: str, dtype: str) -> bool:
        return any(k in self.rules for k in (f"{model}/{dtype}", f"*/{dtype}", f"{model}/*"))

    @property
    def fingerprint(self) -> str:
        """Stable hash of the limits; changes whenever any limit changes."""
        payload = json.dumps(
            {"defaults": self.defaults.model_dump(), "rules": self.rules}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: Path | None = None) -> TolerancePolicy:
        source = path or DEFAULT_POLICY_PATH
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
        policy = raw.get("policy", {})
        return cls(
            version=str(policy.get("version", "unversioned")),
            owner=str(policy.get("owner", "")),
            reference=str(policy.get("reference", "")),
            defaults=ToleranceRule.model_validate(raw.get("defaults", {})),
            rules={str(k): dict(v) for k, v in raw.get("rules", {}).items()},
            source=str(source),
        )
