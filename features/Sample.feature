@sample @no-browser
Feature: Sample project feature
  A template for the project's own features. Feature files in this folder (and
  its subfolders) run after the packs' features; their steps live in steps/.
  Copy this file as a starting point, then delete it.

  Scenario: A step uses a shared helper from the steps folder
    Given a counter starting at 2
    When I add 3 to the counter
    Then the counter should read 5
