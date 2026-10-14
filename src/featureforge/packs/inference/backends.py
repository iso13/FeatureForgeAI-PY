"""Inference backends the pack can qualify.

* :class:`ReferenceBackend`: a deterministic fp32 "golden" model. Logits are a
  pure function of (model, prompt, position), so every run is reproducible.
* :class:`SimulatedAccelerator`: the reference's logits pushed through real
  precision arithmetic (bf16 rounding, fp8 e4m3, int8 per-row symmetric
  quantization), plus a latency model and fault injection (scale drift, noise,
  slowdown). It stands in for a card in the ``sim`` tier and in demos.
* :class:`OpenAICompatibleBackend`: a real serving endpoint (vLLM, TGI, a
  vendor runtime). Used for generation latency and token checks; full logits
  are not exposed over that API.

Hardware backends implement the same :class:`Backend` protocol and raise the
``featureforge.core.stack`` errors, so failures are attributed to the right
layer of the stack.
"""

from __future__ import annotations

import json
import math
import random
import struct
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

VOCAB_SIZE = 64
LOGIT_SCALE = 4.0
PRECISIONS = ("fp32", "bf16", "fp8_e4m3", "int8")

# Simulated per-precision serving costs (ms): time to first token, time per output token.
_LATENCY_MS: dict[str, tuple[float, float]] = {
    "fp32": (42.0, 9.0),
    "bf16": (24.0, 5.2),
    "fp8_e4m3": (15.0, 3.1),
    "int8": (16.0, 3.3),
}


@dataclass(frozen=True)
class Generation:
    tokens: list[int]  # token ids; for streamed HTTP backends, one entry per chunk
    ttft_ms: float  # time to first token
    total_ms: float
    text: str = ""

    @property
    def tpot_ms(self) -> float:
        """Mean time per output token after the first."""
        rest = len(self.tokens) - 1
        return (self.total_ms - self.ttft_ms) / rest if rest > 0 else 0.0


class Backend(Protocol):
    name: str
    model: str
    dtype: str

    def logits(self, prompt: str, positions: int) -> list[list[float]]: ...

    def generate(self, prompt: str, max_tokens: int) -> Generation: ...


def _stable_seed(*parts: object) -> int:
    """Seed that is identical across processes (``hash`` is salted per process)."""
    text = "\x1f".join(str(p) for p in parts)
    value = 2166136261
    for byte in text.encode():
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


@dataclass
class ReferenceBackend:
    model: str = "toy-llm"
    dtype: str = "fp32"
    name: str = "reference"

    def logits(self, prompt: str, positions: int) -> list[list[float]]:
        rows = []
        for position in range(positions):
            rng = random.Random(_stable_seed(self.model, prompt, position))
            rows.append([rng.gauss(0.0, 1.0) * LOGIT_SCALE for _ in range(VOCAB_SIZE)])
        return rows

    def generate(self, prompt: str, max_tokens: int) -> Generation:
        tokens = [_argmax(row) for row in self.logits(prompt, max_tokens)]
        ttft, tpot = _LATENCY_MS["fp32"]
        return Generation(tokens, ttft, ttft + tpot * (max_tokens - 1))


# ─── precision arithmetic ───────────────────────────────────────────────────


def to_bf16(x: float) -> float:
    """Round an fp32 value to bfloat16 (round-to-nearest-even on the top 16 bits)."""
    (bits,) = struct.unpack("<I", struct.pack("<f", x))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    (value,) = struct.unpack("<f", struct.pack("<I", bits))
    return float(value)


def to_fp8_e4m3(x: float) -> float:
    """Round to fp8 e4m3 (3 mantissa bits, max ±448, subnormals below 2**-6)."""
    if x == 0 or math.isnan(x):
        return x
    sign = -1.0 if x < 0 else 1.0
    magnitude = min(abs(x), 448.0)
    exponent = max(math.floor(math.log2(magnitude)), -6)
    step = 2.0 ** (exponent - 3)
    return sign * min(round(magnitude / step) * step, 448.0)


def quantize_int8_row(row: list[float]) -> list[float]:
    """Symmetric per-row int8: scale = max|x| / 127, then dequantize."""
    peak = max(abs(v) for v in row) or 1.0
    scale = peak / 127.0
    return [max(-127, min(127, round(v / scale))) * scale for v in row]


def apply_precision(rows: list[list[float]], dtype: str) -> list[list[float]]:
    if dtype == "fp32":
        return [list(r) for r in rows]
    if dtype == "bf16":
        return [[to_bf16(v) for v in r] for r in rows]
    if dtype == "fp8_e4m3":
        return [[to_fp8_e4m3(v) for v in r] for r in rows]
    if dtype == "int8":
        return [quantize_int8_row(r) for r in rows]
    raise ValueError(f"unknown precision '{dtype}'. Known: {', '.join(PRECISIONS)}")


def _argmax(row: list[float]) -> int:
    return max(range(len(row)), key=row.__getitem__)


# ─── simulated accelerator ──────────────────────────────────────────────────


@dataclass
class Faults:
    """Deliberate defects, used to prove the checks catch them."""

    scale_drift: float = 0.0  # multiply every logit by (1 + scale_drift)
    noise_sigma: float = 0.0  # add N(0, sigma) noise to every logit
    slowdown: float = 0.0  # multiply latency by (1 + slowdown)
    nondeterministic: bool = False  # noise differs on every call


@dataclass
class SimulatedAccelerator:
    model: str = "toy-llm"
    dtype: str = "int8"
    name: str = "sim-accelerator"
    jitter: float = 0.02  # ±2% latency noise, seeded
    faults: Faults = field(default_factory=Faults)
    reference: ReferenceBackend = field(default_factory=ReferenceBackend)
    _calls: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in PRECISIONS:
            raise ValueError(f"unknown precision '{self.dtype}'. Known: {', '.join(PRECISIONS)}")
        self.reference = ReferenceBackend(model=self.model)

    def logits(self, prompt: str, positions: int) -> list[list[float]]:
        self._calls += 1
        rows = self.reference.logits(prompt, positions)
        f = self.faults
        if f.scale_drift:
            rows = [[v * (1.0 + f.scale_drift) for v in r] for r in rows]
        if f.noise_sigma:
            salt = self._calls if f.nondeterministic else 0
            rng = random.Random(_stable_seed("noise", self.model, prompt, salt))
            rows = [[v + rng.gauss(0.0, f.noise_sigma) for v in r] for r in rows]
        return apply_precision(rows, self.dtype)

    def generate(self, prompt: str, max_tokens: int) -> Generation:
        tokens = [_argmax(row) for row in self.logits(prompt, max_tokens)]
        ttft, tpot = _LATENCY_MS[self.dtype]
        rng = random.Random(_stable_seed("latency", self.model, self.dtype, prompt, self._calls))
        factor = (1.0 + self.faults.slowdown) * (1.0 + rng.uniform(-self.jitter, self.jitter))
        prompt_cost = 0.05 * len(prompt.split())  # prefill grows with prompt length
        first = (ttft + prompt_cost) * factor
        return Generation(tokens, first, first + tpot * factor * (max_tokens - 1))


# ─── card backends ──────────────────────────────────────────────────────────


class CardFactory(Protocol):
    """Builds a backend bound to one accelerator card. Set ``device_backend`` to
    ``"package.module:factory"`` to qualify real hardware."""

    def __call__(self, model: str, dtype: str, card: int) -> Backend: ...


def simulated_card(model: str, dtype: str, card: int) -> Backend:
    """Card factory backed by the simulator, for trying the hardware tiers anywhere.

    ``FF_SIM_FAULTY_CARD=<n>`` makes card ``n`` return noisy logits, like a bad board.
    """
    import os

    accelerator = SimulatedAccelerator(model=model, dtype=dtype, name=f"sim-card-{card}")
    if os.environ.get("FF_SIM_FAULTY_CARD", "") == str(card):
        accelerator.faults = Faults(noise_sigma=0.5)
    return accelerator


def load_card_factory(reference: str) -> CardFactory:
    from importlib import import_module

    module_name, _, attribute = reference.partition(":")
    factory = getattr(import_module(module_name), attribute or "card_factory")
    if not callable(factory):
        raise TypeError(f"device_backend '{reference}' is not callable")
    return factory  # type: ignore[no-any-return]


# ─── real serving endpoint ──────────────────────────────────────────────────


@dataclass
class OpenAICompatibleBackend:
    """Any server exposing ``POST /v1/completions`` (vLLM, TGI, vendor runtimes)."""

    base_url: str
    model: str
    dtype: str = "unknown"
    api_key: str = ""
    timeout_s: float = 60.0
    name: str = "openai-compatible"

    def logits(self, prompt: str, positions: int) -> list[list[float]]:
        raise NotImplementedError(
            "Full logits are not exposed over the completions API; compare logits with an "
            "in-process or device backend, and use generate() for serving checks."
        )

    def generate(self, prompt: str, max_tokens: int) -> Generation:
        from featureforge.core.stack import ServingError

        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/completions", data=body, headers=headers
        )
        started = time.perf_counter()
        first: float | None = None
        pieces = 0
        text: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    if first is None:
                        first = time.perf_counter()
                    pieces += 1
                    choices = json.loads(line.removeprefix("data:")).get("choices") or [{}]
                    text.append(str(choices[0].get("text", "")))
        except OSError as err:
            raise ServingError(f"completions request to {self.base_url} failed: {err}") from err
        end = time.perf_counter()
        if first is None:
            raise ServingError(f"{self.base_url} returned no tokens")
        return Generation(
            list(range(pieces)), (first - started) * 1000, (end - started) * 1000, "".join(text)
        )
