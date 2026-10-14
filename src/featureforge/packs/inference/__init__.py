"""Inference qualification pack: numerical accuracy, determinism and performance.

Built for accelerator software stacks (firmware → compiler → runtime → serving),
where correctness is numerical rather than boolean and performance is part of
correctness. See ``tolerances.toml`` for what "close enough" means.
"""

from pathlib import Path

from featureforge.packs import Pack

PACK = Pack(
    name="inference",
    description="Numerical accuracy vs a reference, determinism and latency for LLM inference",
    features_dir=Path(__file__).parent / "features",
    plugin="featureforge.packs.inference.steps",
)
