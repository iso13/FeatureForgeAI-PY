@inference @no-browser @tier-sim
Feature: Latency and throughput hold against the baseline
  We sell latency and efficiency, so a double-digit slowdown is a bug of the
  same severity as a wrong answer. Each number is compared with the pinned
  release baseline and the last known-good run; worse than either by more than
  its budget fails the scenario.

  Scenario Outline: <dtype> serving performance stays within budget
    Given the accelerator runs "toy-llm" in "<dtype>"
    When I send 32 requests of 64 tokens from the "smoke" prompt set
    Then "ttft_p50_ms" should not regress more than 10% against the baseline
    And "ttft_p99_ms" should not regress more than 15% against the baseline
    And "tpot_p50_ms" should not regress more than 10% against the baseline
    And "throughput_tok_s" should not drop more than 5% against the baseline

    Examples:
      | dtype    |
      | bf16     |
      | fp8_e4m3 |
      | int8     |

  Scenario: A 15% slowdown in int8 is flagged as a regression
    Given the accelerator runs "toy-llm" in "int8"
    And a 15% slowdown is injected
    When I send 32 requests of 64 tokens from the "smoke" prompt set
    Then "ttft_p50_ms" should be flagged as a regression of more than 10%
    And "throughput_tok_s" should be flagged as a regression of more than 5%
