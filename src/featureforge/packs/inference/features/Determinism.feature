@inference @no-browser @tier-sim
Feature: Deterministic output on a fixed build
  The same build, prompt and inputs must produce the same tokens. Without that,
  "it was right yesterday" can't be told apart from noise.

  Scenario: Repeated greedy generations are identical
    Given the accelerator runs "toy-llm" in "int8"
    When I generate 16 tokens for the same prompt 3 times
    Then every run should produce identical tokens

  Scenario: The determinism check catches run-to-run variation
    Given the accelerator runs "toy-llm" in "int8"
    And the accelerator output varies from call to call
    When I generate 16 tokens for the same prompt 3 times
    Then the runs should not all be identical
