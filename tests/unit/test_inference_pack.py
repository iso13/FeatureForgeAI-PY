import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from featureforge.core.stack import ServingError, StackLayer
from featureforge.packs.inference.backends import (
    Faults,
    Generation,
    OpenAICompatibleBackend,
    ReferenceBackend,
    SimulatedAccelerator,
    apply_precision,
    quantize_int8_row,
    to_bf16,
    to_fp8_e4m3,
)
from featureforge.packs.inference.compare import compare_logits, merge
from featureforge.packs.inference.steps import first_divergence, load_prompts
from featureforge.packs.inference.tolerance import TolerancePolicy, ToleranceRule

# ─── precision arithmetic ───────────────────────────────────────────────────


def test_bf16_rounds_to_nearest_even():
    assert to_bf16(1.0) == 1.0
    assert to_bf16(1.0 + 2**-8) == 1.0  # exactly halfway → even (1.0)
    assert to_bf16(1.0 + 3 * 2**-8) == 1.0 + 2**-6  # halfway → even (…0100)
    assert to_bf16(1.0 + 2**-7) == 1.0 + 2**-7  # representable


def test_fp8_e4m3_keeps_three_mantissa_bits_and_saturates():
    assert to_fp8_e4m3(0.3) == 0.3125  # step 2**-5 between 0.25 and 0.5
    assert to_fp8_e4m3(-1.0) == -1.0
    assert to_fp8_e4m3(1000.0) == 448.0
    assert to_fp8_e4m3(0.0) == 0.0


def test_int8_error_is_bounded_by_half_a_step():
    row = [3.7, -12.4, 0.01, 5.5, -0.3]
    step = 12.4 / 127
    quantized = quantize_int8_row(row)
    assert max(abs(q - x) for q, x in zip(quantized, row, strict=True)) <= step / 2 + 1e-12
    assert quantized[1] == pytest.approx(-12.4)


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="unknown precision"):
        apply_precision([[1.0]], "fp4")
    with pytest.raises(ValueError, match="unknown precision"):
        SimulatedAccelerator(dtype="fp4")


def test_reference_logits_are_reproducible():
    a = ReferenceBackend().logits("hello", 4)
    assert a == ReferenceBackend().logits("hello", 4)
    assert a != ReferenceBackend(model="other").logits("hello", 4)


# ─── tolerance policy ───────────────────────────────────────────────────────

POLICY = """
[policy]
version = "test.1"

[defaults]
rtol = 1e-5
atol = 1e-6
min_cosine = 0.99999

[rules."*/int8"]
atol = 0.065
min_top1_agreement = 0.98

[rules."llama/int8"]
atol = 0.1

[rules."llama/*"]
min_cosine = 0.999
"""


@pytest.fixture
def policy(tmp_path):
    path = tmp_path / "tolerances.toml"
    path.write_text(POLICY)
    return TolerancePolicy.load(path)


def test_rule_lookup_goes_from_specific_to_general(policy):
    assert policy.rule_for("toy", "fp32").atol == 1e-6  # defaults
    assert policy.rule_for("toy", "int8").atol == 0.065  # */int8
    llama = policy.rule_for("llama", "int8")
    assert llama.atol == 0.1  # llama/int8 wins
    assert llama.min_top1_agreement == 0.98  # still inherited from */int8
    assert llama.min_cosine == 0.999  # from llama/*
    assert llama.rtol == 1e-5  # from defaults


def test_fingerprint_changes_when_a_limit_changes(policy):
    loosened = policy.model_copy(update={"rules": {**policy.rules, "*/int8": {"atol": 0.5}}})
    assert loosened.fingerprint != policy.fingerprint
    assert policy.version == "test.1"


def test_shipped_policy_covers_every_precision():
    shipped = TolerancePolicy.load()
    for dtype in ("bf16", "fp8_e4m3", "int8"):
        assert shipped.has_rule("toy-llm", dtype)
    assert not shipped.has_rule("toy-llm", "fp32")  # fp32 uses the exact defaults


# ─── comparison ─────────────────────────────────────────────────────────────


def test_identical_logits_pass_every_limit():
    rows = ReferenceBackend().logits("p", 4)
    result = compare_logits(rows, rows, ToleranceRule(max_perplexity_delta_pct=0.0))
    assert result.passed
    assert result.stats.cosine == pytest.approx(1.0)
    assert result.stats.top1_agreement == 1.0 and result.stats.mismatch_fraction == 0.0


def test_scale_drift_is_flagged_both_ways_on_perplexity():
    reference = ReferenceBackend().logits("p", 8)
    rule = ToleranceRule(
        rtol=0.005,
        atol=0.065,
        max_mismatch_fraction=0.001,
        min_cosine=0.99,
        min_top1_agreement=0.9,
        min_topk_overlap=0.9,
        max_perplexity_delta_pct=1.0,
    )
    sharper = [[v * 1.03 for v in row] for row in reference]
    result = compare_logits(reference, sharper, rule)
    assert result.stats.perplexity_delta_pct < 0  # looks "better"
    assert result.violated("perplexity_delta_pct") and result.violated("mismatch_fraction")
    assert "|perplexity_delta_pct| <= 1" in result.render()


def test_shape_mismatch_is_an_error_not_a_pass():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_logits([[1.0]], [[1.0], [2.0]], ToleranceRule())
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        compare_logits([[1.0, 2.0]], [[1.0]], ToleranceRule())


def test_merge_keeps_the_worst_values():
    reference = ReferenceBackend()
    healthy = SimulatedAccelerator(dtype="int8")
    drifted = SimulatedAccelerator(dtype="int8", faults=Faults(scale_drift=0.02))
    rule = TolerancePolicy.load().rule_for("toy-llm", "int8")
    good = compare_logits(reference.logits("a", 8), healthy.logits("a", 8), rule)
    bad = compare_logits(reference.logits("b", 8), drifted.logits("b", 8), rule)
    merged = merge([good, bad])
    assert merged.stats.max_abs_err == bad.stats.max_abs_err
    assert merged.stats.positions == 16
    assert not merged.passed


def test_every_shipped_precision_passes_the_shipped_policy():
    policy = TolerancePolicy.load()
    reference = ReferenceBackend()
    for dtype in ("fp32", "bf16", "fp8_e4m3", "int8"):
        candidate = SimulatedAccelerator(dtype=dtype)
        rule = policy.rule_for("toy-llm", dtype)
        results = [
            compare_logits(reference.logits(p, 16), candidate.logits(p, 16), rule)
            for p in load_prompts("smoke")
        ]
        assert merge(results).passed, merge(results).render()


# ─── determinism & latency model ────────────────────────────────────────────


def test_first_divergence_points_at_run_and_token():
    runs = [Generation([1, 2, 3], 1, 2), Generation([1, 2, 3], 1, 2), Generation([1, 9, 3], 1, 2)]
    assert first_divergence(runs) == (2, 1)
    assert first_divergence(runs[:2]) is None


def test_slowdown_fault_scales_latency():
    fast = SimulatedAccelerator(dtype="int8", jitter=0.0).generate("p", 8)
    slow = SimulatedAccelerator(dtype="int8", jitter=0.0, faults=Faults(slowdown=0.2)).generate(
        "p", 8
    )
    assert slow.ttft_ms == pytest.approx(fast.ttft_ms * 1.2)
    assert slow.tpot_ms == pytest.approx(fast.tpot_ms * 1.2)


def test_load_prompts_names_known_sets():
    assert len(load_prompts("smoke")) == 8
    with pytest.raises(FileNotFoundError, match="Known: smoke"):
        load_prompts("nope")


# ─── OpenAI-compatible serving backend ──────────────────────────────────────


class _StreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in ["Can", "berra"][: body["max_tokens"]]:
            chunk = {"choices": [{"text": word}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def test_openai_compatible_backend_streams_and_times_tokens():
    server = HTTPServer(("127.0.0.1", 0), _StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleBackend(f"http://127.0.0.1:{server.server_port}", "m")
        generation = backend.generate("capital of Australia?", 2)
    finally:
        server.shutdown()
    assert generation.text == "Canberra"
    assert len(generation.tokens) == 2
    assert 0 <= generation.ttft_ms <= generation.total_ms


def test_openai_compatible_backend_failure_is_a_serving_error():
    backend = OpenAICompatibleBackend("http://127.0.0.1:9", "m", timeout_s=2)
    with pytest.raises(ServingError) as raised:
        backend.generate("p", 1)
    assert raised.value.layer is StackLayer.SERVING
    with pytest.raises(NotImplementedError):
        backend.logits("p", 1)
