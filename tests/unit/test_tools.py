import ast
from pathlib import Path

from featureforge.packs import get_pack
from featureforge.tools.gherkin import parse_feature
from featureforge.tools.lint import lint_feature
from featureforge.tools.scaffold import (
    StepPattern,
    collect_step_patterns,
    missing_steps,
    parse_pattern_to_regex,
    render_stubs,
    scaffold,
    step_signature,
)

OUTLINE = (
    """@pricing
Feature: Price quotes

  Background:
    Given a rating engine

  Scenario Outline: Quote by state
    Given a "<state>" homeowner with 2 claims
    When I request a quote
    Then the premium should be <premium>
    """
    + '"""'
    + """
    doc string with Given inside is ignored
    """
    + '"""'
    + """

    Examples:
      | state | premium | unused |
      | NE    | 100     | x      |
      | CA    | 140     | y      |
"""
)


def test_parser_reads_structure():
    feature = parse_feature(OUTLINE)
    assert feature.name == "Price quotes"
    assert feature.tags == ["@pricing"]
    assert feature.background and feature.background.steps[0].text == "a rating engine"
    outline = feature.scenarios[0]
    assert outline.is_outline and outline.example_count == 2
    assert [s.keyword for s in outline.steps] == ["Given", "When", "Then"]
    assert outline.examples[0].header == ["state", "premium", "unused"]


def test_lint_flags_outline_problems_and_style():
    issues = {(i.rule, i.message) for i in lint_feature(parse_feature(OUTLINE + "   "))}
    rules = {rule for rule, _ in issues}
    assert "no-unused-variables" in rules
    assert "no-background-only-scenario" in rules
    assert "no-trailing-spaces" in rules
    assert "new-line-at-eof" in rules


def test_lint_rules_on_bad_feature():
    text = (
        "@focus @focus\nFeature:\n\n  Scenario: a\n    Given x\n    Given y\n"
        "  Scenario: a\n    When z\n\n\n  Scenario Outline: o\n    Given <missing>\n"
    )
    rules = {i.rule for i in lint_feature(parse_feature(text))}
    assert {
        "no-unnamed-features",
        "no-duplicate-tags",
        "no-restricted-tags",
        "use-and",
        "no-dupe-scenario-names",
        "no-multiple-empty-lines",
        "no-scenario-outlines-without-examples",
    } <= rules


def test_shipped_features_are_lint_clean():
    for path in sorted(get_pack("ai").features_dir.glob("*.feature")):
        feature = parse_feature(path.read_text(), path)
        assert lint_feature(feature) == [], path.name


def test_parse_pattern_regex():
    regex = parse_pattern_to_regex('a "{group}" with {count:d} claims at {rate:g}')
    pattern = StepPattern("given", "parse", 'a "{group}" with {count:d} claims at {rate:g}')
    assert pattern.matches('a "NE" with 2 claims at 0.5')
    assert not pattern.matches('a "NE" with two claims at 0.5')
    assert regex.startswith("a\\ ")


def test_step_signature_names_parameters():
    pattern, params = step_signature('a "<state>" homeowner with 2 claims')
    assert pattern == 'a "{state}" homeowner with {number2:g} claims'
    assert params == [("state", "str"), ("number2", "float")]
    pattern, params = step_signature("the premium should be <premium>")
    assert pattern == "the premium should be {premium}"


def test_pack_steps_cover_pack_features():
    pack = get_pack("ai")
    steps = Path(__file__).parents[2] / "src/featureforge/packs/ai/steps.py"
    missing, _ = scaffold([pack.features_dir], [steps])
    # Only the @wip salary scenario is intentionally unimplemented.
    assert {m.text for m in missing} == {
        "a dataset with gender and job title attributes",
        "the model predicts salaries for each entry",
        "the salary distribution should not significantly differ by gender",
    }


def test_scaffold_outline_and_render_valid_python(tmp_path):
    steps_file = tmp_path / "steps.py"
    steps_file.write_text(
        "from pytest_bdd import given, when, parsers\n"
        "@given('a rating engine')\ndef a(): ...\n"
        "@when(parsers.re(r'I request an? quote'))\ndef b(): ...\n"
    )
    patterns = collect_step_patterns([steps_file])
    assert {p.kind for p in patterns} == {"literal", "re"}
    missing = missing_steps([parse_feature(OUTLINE)], patterns)
    assert [m.text for m in missing] == [
        'a "<state>" homeowner with 2 claims',
        "the premium should be <premium>",
    ]
    assert missing[0].sample == 'a "NE" homeowner with 2 claims'
    source = render_stubs(missing)
    ast.parse(source)
    assert "@given(parsers.parse('a \"{state}\" homeowner with {number2:g} claims'))" in source
    assert "def the_premium_should_be(ctx: ScenarioContext, premium: str) -> None:" in source
