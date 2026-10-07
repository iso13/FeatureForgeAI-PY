"""Collect every scenario from the enabled packs.

TS equivalent: ``paths: ['src/features/**/*.feature']`` in cucumber.mjs.
"""

from pytest_bdd import scenarios

from featureforge.packs import get_pack

scenarios(str(get_pack("ai").features_dir))
