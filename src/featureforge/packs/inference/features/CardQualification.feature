@inference @no-browser
Feature: Accuracy and consistency on accelerator cards
  The sim-tier checks again, on cards. These need hardware, so they run on the
  nightly and release schedules rather than on every commit, and skip with a
  reason on runners without cards.

  Background:
    Given the "toy-llm" fp32 reference model
    And the tolerance policy is loaded

  @tier-single_card
  Scenario: int8 on a card stays within the tolerance policy
    Given card 0 runs "toy-llm" in "int8"
    When I compare logits on the "smoke" prompt set
    Then every limit in the tolerance policy should hold

  @tier-multi_card
  Scenario: Every card in the node produces the same tokens
    Given cards 0 to 3 run "toy-llm" in "int8"
    When each card generates 16 tokens for the same prompt
    Then every card should produce identical tokens
