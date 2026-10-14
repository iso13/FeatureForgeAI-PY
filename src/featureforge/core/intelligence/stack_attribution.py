"""Which layer of the inference stack broke: firmware, compiler, runtime, serving,
hardware, the test itself, or nothing (a flake).

A red test on an accelerator stack could be any of these, and telling them apart
fast is most of the value a quality team adds. Evidence is weighed in order of
strength; the first source that decides wins:

1. **Rerun**: failed, then passed on rerun with nothing changed → ``FLAKY``.
2. **Explicit**: the code raised a :class:`~featureforge.core.stack.StackError`
   naming its layer (a backend's ``CompilerError``, a ``DeviceError`` …).
3. **Test code**: Python errors (``NameError``, ``TypeError`` …) raised from step
   or test files → ``TEST``; a typo isn't the compiler's fault.
4. **Change**: exactly one stack component changed since the last known-good
   run (from the build manifest) → that component's layer.
5. **Signals**: words in the error and stack trace (``MLIR``, ``ECC``, ``mailbox``…).

Numerical and performance failures that can't be pinned further stay
``MODEL_NUMERICS`` / ``PERFORMANCE`` with the candidate layers listed, so the
next action (bisect the changed components) is obvious.
"""

from __future__ import annotations

import re
import traceback
from collections.abc import Mapping

from pydantic import BaseModel, Field

from featureforge.core.stack import StackError, StackLayer


class StackAttribution(BaseModel):
    layer: StackLayer
    confidence: float = Field(ge=0.0, le=1.0)
    method: str  # rerun | explicit | change | signals | test-code | unresolved
    evidence: list[str] = Field(default_factory=list)
    candidates: list[StackLayer] = Field(default_factory=list)

    def render(self) -> str:
        head = f"Stack layer: {self.layer.value} ({round(self.confidence * 100)}%, {self.method})"
        lines = [head, *(f"  - {e}" for e in self.evidence)]
        if self.candidates:
            lines.append("  candidates: " + ", ".join(c.value for c in self.candidates))
        return "\n".join(lines)


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


STACK_SIGNALS: dict[StackLayer, re.Pattern[str]] = {
    StackLayer.FIRMWARE: _rx(
        r"\bfirmware\b",
        r"\bfw[ _-]?(?:version|error|fault|hang)",
        r"\bmailbox\b",
        r"\bwatchdog\b",
        r"\bboot ?(?:rom|loader)?\b",
        r"\bmicrocode\b",
    ),
    StackLayer.COMPILER: _rx(
        r"\bcompil(?:er|ation|e)\b",
        r"\blowering\b",
        r"\bMLIR\b",
        r"\bcodegen\b",
        r"\bgraph partition",
        r"\bunsupported op",
        r"\bkernel fusion\b",
        r"\bscheduling pass\b",
    ),
    StackLayer.RUNTIME: _rx(
        r"\bruntime\b",
        r"\bkernel launch",
        r"\bDMA\b",
        r"\bout of (?:device )?memory\b",
        r"\bOOM\b",
        r"\ballocat(?:e|ion|or)\b",
        r"\bstream sync",
        r"\bcommand queue\b",
        r"\bdriver\b",
    ),
    StackLayer.SERVING: _rx(
        r"\bserving\b",
        r"\bendpoint\b",
        r"/v1/(?:completions|chat)",
        r"\bHTTP (?:5\d\d)\b",
        r"\b50[234]\b",
        r"\brequest queue\b",
        r"\bbatch(?:ing|er) ",
        r"\bvLLM\b",
        r"\btokenizer\b",
    ),
    StackLayer.HARDWARE: _rx(
        r"\bECC\b",
        r"\bPCIe\b",
        r"\blink (?:down|training|error)",
        r"\bthermal\b",
        r"\bthrottl",
        r"\bdevice (?:lost|not found|unresponsive)",
        r"\buncorrectable\b",
        r"\bbad (?:board|card)\b",
        r"\bcard\(?s?\)? \d+ disagree",
    ),
}

# Build-manifest component names → layer they belong to.
COMPONENT_LAYERS: dict[str, StackLayer] = {
    "firmware": StackLayer.FIRMWARE,
    "compiler": StackLayer.COMPILER,
    "runtime": StackLayer.RUNTIME,
    "driver": StackLayer.RUNTIME,
    "kernels": StackLayer.RUNTIME,
    "serving": StackLayer.SERVING,
    "board": StackLayer.HARDWARE,
    "card": StackLayer.HARDWARE,
    "silicon": StackLayer.HARDWARE,
}

_UNPINNED = (StackLayer.MODEL_NUMERICS, StackLayer.PERFORMANCE, StackLayer.UNKNOWN)
_TEST_ERRORS = (NameError, AttributeError, TypeError, ImportError, KeyError, IndexError)
_TEST_PATH = re.compile(r"(?:^|[/\\])(?:steps|tests)[/\\]|_steps\.py|(?:^|[/\\])test_[^/\\]*\.py")


def layer_of_component(name: str) -> StackLayer | None:
    key = name.lower().replace("-", "_")
    for prefix, layer in COMPONENT_LAYERS.items():
        if key == prefix or key.startswith(prefix + "_"):
            return layer
    return None


def detect_stack_layers(text: str) -> list[StackLayer]:
    return [layer for layer, pattern in STACK_SIGNALS.items() if pattern.search(text)]


def _raised_in_test_code(exception: BaseException) -> bool:
    frames = traceback.extract_tb(exception.__traceback__)
    return bool(frames) and bool(_TEST_PATH.search(frames[-1].filename))


def attribute_failure(
    exception: BaseException | None,
    message: str = "",
    stack: str = "",
    *,
    changes: Mapping[str, tuple[str, str]] | None = None,
    passed_on_rerun: bool = False,
    first_failure: StackAttribution | None = None,
) -> StackAttribution:
    """Attribute one failure. ``changes`` maps manifest component → (good, now)."""
    if passed_on_rerun:
        evidence = ["failed, then passed on rerun with the same build and inputs"]
        if first_failure is not None:
            evidence.append(f"first attempt looked like {first_failure.layer.value}")
            evidence += first_failure.evidence[:2]
        return StackAttribution(
            layer=StackLayer.FLAKY, confidence=0.9, method="rerun", evidence=evidence
        )

    text = f"{message}\n{stack}"
    explicit = exception.layer if isinstance(exception, StackError) else None
    if explicit is not None and explicit not in _UNPINNED:
        return StackAttribution(
            layer=explicit,
            confidence=0.95,
            method="explicit",
            evidence=[f"{type(exception).__name__}: {_first_line(str(exception))}"],
        )

    if isinstance(exception, _TEST_ERRORS) and not explicit and _raised_in_test_code(exception):
        return StackAttribution(
            layer=StackLayer.TEST,
            confidence=0.7,
            method="test-code",
            evidence=[f"{type(exception).__name__} raised in step/test code"],
        )

    changed = _changed_layers(changes or {})
    if len(changed) == 1:
        ((layer, components),) = changed.items()
        return StackAttribution(
            layer=layer,
            confidence=0.8,
            method="change",
            evidence=[f"only {', '.join(components)} changed since the last known-good run"],
        )

    signalled = detect_stack_layers(text)
    if len(signalled) == 1:
        return StackAttribution(
            layer=signalled[0],
            confidence=0.75,
            method="signals",
            evidence=[f"{signalled[0].value.lower()} signals in the error: {_first_line(message)}"],
        )

    candidates = list(dict.fromkeys([*changed, *signalled]))
    evidence = [_first_line(message)] if message else []
    if len(changed) > 1:
        evidence.append(
            "several components changed since the last known-good run: "
            + ", ".join(c for cs in changed.values() for c in cs)
            + " - bisect them one at a time"
        )
    if explicit in (StackLayer.MODEL_NUMERICS, StackLayer.PERFORMANCE):
        return StackAttribution(
            layer=explicit,
            confidence=0.6,
            method="unresolved",
            evidence=evidence,
            candidates=candidates,
        )
    return StackAttribution(
        layer=StackLayer.UNKNOWN,
        confidence=0.3 if candidates else 0.1,
        method="unresolved",
        evidence=evidence,
        candidates=candidates,
    )


def _changed_layers(changes: Mapping[str, tuple[str, str]]) -> dict[StackLayer, list[str]]:
    layers: dict[StackLayer, list[str]] = {}
    for component, (before, after) in changes.items():
        layer = layer_of_component(component)
        if layer is not None and before != after:
            layers.setdefault(layer, []).append(f"{component} ({before} → {after})")
    return layers


def _first_line(message: str) -> str:
    line = message.strip().splitlines()[0] if message.strip() else ""
    return line if len(line) <= 200 else line[:197] + "..."
