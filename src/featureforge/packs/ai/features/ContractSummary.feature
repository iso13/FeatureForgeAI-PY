@contractSummary @no-browser @ai @rag
Feature: Summarize contracts from retrieved clauses
  As a contract analyst
  I want AI summaries built only from retrieved clauses
  So that I can trust what the summary says

  Background:
    Given the RAG system has indexed contract documents containing clauses for confidentiality and payment terms

  Scenario: Summary is grounded in the retrieved clauses
    When I request a summary of a contract
    Then the system should retrieve relevant context for each clause
    And the system should generate a summary including confidentiality and payment terms
    And the summary should match the retrieved content

  Scenario: Missing clauses are flagged rather than invented
    When I request a summary of a contract missing indemnity clauses
    Then the system should log a warning about the missing context
    And the generated summary should note the missing indemnity information

  Scenario: Newly added clauses appear in the next summary
    Given a new indemnity clause is added to the contract
    When I re-request the summary
    Then the new clause should be included in the updated AI summary
