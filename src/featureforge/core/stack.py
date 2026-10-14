"""Errors that name the layer of the inference stack they came from.

A model's answer passes through firmware, the compiler, the runtime and the
serving stack. Backends and step code raise these instead of bare exceptions,
so a red test says *where* it broke, not just *that* it broke. Attribution of
failures that don't raise one of these lives in
``featureforge.core.intelligence.stack_attribution``.
"""

from __future__ import annotations

from enum import StrEnum


class StackLayer(StrEnum):
    FIRMWARE = "FIRMWARE"
    COMPILER = "COMPILER"
    RUNTIME = "RUNTIME"
    SERVING = "SERVING"
    HARDWARE = "HARDWARE"  # a bad board, link or device, not software
    MODEL_NUMERICS = "MODEL_NUMERICS"  # outputs outside tolerance, cause not yet isolated
    PERFORMANCE = "PERFORMANCE"  # latency/throughput regression, cause not yet isolated
    TEST = "TEST"  # the test or its infrastructure is wrong
    FLAKY = "FLAKY"  # failed, then passed on rerun with nothing changed
    UNKNOWN = "UNKNOWN"


class StackError(Exception):
    """Base class: a failure attributed to one layer of the stack."""

    layer: StackLayer = StackLayer.UNKNOWN


class FirmwareError(StackError):
    layer = StackLayer.FIRMWARE


class CompilerError(StackError):
    layer = StackLayer.COMPILER


class RuntimeFault(StackError):
    layer = StackLayer.RUNTIME


class ServingError(StackError):
    layer = StackLayer.SERVING


class DeviceError(StackError):
    layer = StackLayer.HARDWARE


class NumericalMismatch(StackError, AssertionError):
    """Outputs outside the tolerance policy. Still an AssertionError for pytest."""

    layer = StackLayer.MODEL_NUMERICS


class PerformanceRegression(StackError, AssertionError):
    layer = StackLayer.PERFORMANCE
