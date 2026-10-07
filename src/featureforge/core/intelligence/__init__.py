"""Step Intelligence Layer: classify a failing step, check each layer, write a brief.

TS equivalent: ``src/core/intelligence/`` (failureClassifier, layerValidator,
developerBrief). The pipeline is the same; the pieces are pure functions over
Pydantic models, so they can be unit-tested without a browser or an LLM.
"""

from featureforge.core.intelligence.brief import (
    generate_developer_brief,
    render_console,
    render_html,
    write_brief,
)
from featureforge.core.intelligence.classifier import classify_failure, reconcile_with_layers
from featureforge.core.intelligence.layer_validator import (
    PageSnapshot,
    snapshot_page,
    validate_all_layers,
)
from featureforge.core.intelligence.models import (
    DeveloperBrief,
    FailureClassification,
    FailureLayer,
    LayerResult,
    LayerValidationReport,
    NetworkRequest,
)

__all__ = [
    "DeveloperBrief",
    "FailureClassification",
    "FailureLayer",
    "LayerResult",
    "LayerValidationReport",
    "NetworkRequest",
    "PageSnapshot",
    "classify_failure",
    "generate_developer_brief",
    "reconcile_with_layers",
    "render_console",
    "render_html",
    "snapshot_page",
    "validate_all_layers",
    "write_brief",
]
