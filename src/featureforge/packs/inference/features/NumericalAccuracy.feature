@inference @no-browser @tier-sim
Feature: Numerical accuracy against the fp32 reference
  Quantized inference is not bit-identical to the reference. Each precision is
  held to the limits in the tolerance policy, which is versioned and reviewed
  like code, and each run records which policy version it was judged against.

  Background:
    Given the "toy-llm" fp32 reference model
    And the tolerance policy is loaded

  Scenario Outline: <dtype> logits stay within the tolerance policy
    Given the accelerator runs "toy-llm" in "<dtype>"
    When I compare logits on the "smoke" prompt set
    Then every limit in the tolerance policy should hold

    Examples:
      | dtype    |
      | fp32     |
      | bf16     |
      | fp8_e4m3 |
      | int8     |

  Scenario: A 2% scale drift in int8 is caught even though perplexity improves
    Given the accelerator runs "toy-llm" in "int8"
    And a 2% scale drift is injected
    When I compare logits on the "smoke" prompt set
    Then the comparison should flag "mismatch_fraction"
    And the comparison should flag "perplexity_delta_pct"
