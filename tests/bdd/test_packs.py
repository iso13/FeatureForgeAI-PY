"""Collect every scenario: the enabled packs' features, then the project's ``features/``.

TS equivalent: ``paths: ['src/features/**/*.feature']`` in cucumber.mjs.
"""

from pytest_bdd import scenarios

from featureforge.collect import feature_paths

scenarios(*feature_paths())
