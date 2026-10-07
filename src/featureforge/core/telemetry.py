"""OpenTelemetry tracing that degrades to a no-op.

TS equivalents: ``src/core/support/tracer.ts`` and
``src/plugins/telemetry/traceHelper.ts`` (``withSpan``).

Differences from the TS engine:
* nothing starts at import time — the plugin calls :func:`init_telemetry` once;
* if the SDK extra isn't installed, or tracing is off, every call is a no-op,
  so step code can use :func:`span` unconditionally.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol

from featureforge.core.settings import Settings

TRACER_NAME = "featureforge"

AttrValue = str | bool | int | float


class SpanLike(Protocol):
    def set_attribute(self, key: str, value: AttrValue) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...
    def end(self) -> None: ...


class _NoopSpan:
    def set_attribute(self, key: str, value: AttrValue) -> None:
        return None

    def record_exception(self, exception: BaseException) -> None:
        return None

    def end(self) -> None:
        return None


_provider: Any = None
_enabled = False


def init_telemetry(settings: Settings) -> bool:
    """Configure an OTLP exporter. Returns True when tracing is live."""
    global _provider, _enabled
    if not settings.enable_tracing or _enabled:
        return _enabled
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        if settings.verbose:
            print("Tracing requested but the 'telemetry' extra is not installed; continuing.")
        return False

    resource = Resource.create(
        {
            "service.name": "featureforge-py",
            "service.version": "0.1.0",
            "deployment.environment.name": settings.env,
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
    )
    trace.set_tracer_provider(provider)
    _provider, _enabled = provider, True
    return True


def shutdown_telemetry() -> None:
    global _provider, _enabled
    if _provider is not None:
        _provider.shutdown()
    _provider, _enabled = None, False


def _clean(attributes: Mapping[str, object]) -> dict[str, AttrValue]:
    cleaned: dict[str, AttrValue] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        cleaned[key] = value if isinstance(value, str | bool | int | float) else str(value)
    return cleaned


def start_span(name: str, attributes: Mapping[str, object] | None = None) -> SpanLike:
    """Start a span that the caller ends (used across pytest-bdd hooks)."""
    if not _enabled:
        return _NoopSpan()
    from opentelemetry import trace

    tracer = trace.get_tracer(TRACER_NAME)
    started: SpanLike = tracer.start_span(name, attributes=_clean(attributes or {}))
    return started


@contextmanager
def span(name: str, **attributes: object) -> Iterator[SpanLike]:
    """Context-managed span; the Python form of ``withSpan`` in the TS engine."""
    if not _enabled:
        yield _NoopSpan()
        return
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    tracer = trace.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span(name, attributes=_clean(attributes)) as current:
        try:
            yield current
        except BaseException as exc:
            current.record_exception(exc)
            current.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
