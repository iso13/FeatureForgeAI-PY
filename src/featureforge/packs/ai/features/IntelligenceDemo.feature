@demo @no-browser @ai
Feature: Step Intelligence Layer demo
  Intentionally failing scenario that shows the Developer Brief.
  Excluded by default; run it with: featureforge run --tags "@demo"

  Scenario: A biased decision is caught and classified
    Given a model is loaded
    And I input a sample with demographic group "white"
    When I get the model's decision
    Then the decision outcome should be within a fairness threshold of 0.02
