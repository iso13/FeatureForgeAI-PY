"""AI quality pack: fairness/bias and RAG validation.

TS equivalents: ``src/features/ai``, ``src/steps/ai`` and ``src/plugins/ai``.
"""

from pathlib import Path

from featureforge.packs import Pack

PACK = Pack(
    name="ai",
    description="Fairness/bias audits and RAG grounding checks for AI systems",
    features_dir=Path(__file__).parent / "features",
    plugin="featureforge.packs.ai.steps",
)
