"""Unit tests for the Mondrian strata functions."""

from __future__ import annotations

from care.conformal.strata import (
    by_column,
    by_domain,
    by_source,
    marginal_strata,
)
from care.core import RepairCandidate


def test_marginal_is_single_stratum():
    assert marginal_strata(RepairCandidate(target_ref="r1::age")) == "default"


def test_by_column_uses_column_and_groups_metadata():
    f = by_column()
    assert f(RepairCandidate(target_ref="r1::age")) == "col:age"
    assert f("r2::city") == "col:city"
    assert f("meta:ds:license") == "col:__meta__"


def test_by_domain_maps_columns_to_domains():
    f = by_domain({"age": "pii", "salary": "pii", "city": "geo"})
    assert f("r1::age") == "domain:pii"
    assert f("r1::city") == "domain:geo"
    assert f("r1::unmapped") == "domain:other"


def test_by_source_uses_map_then_evidence_fallback():
    f = by_source({"r1::age": "census"})
    assert f("r1::age") == "src:census"
    # ref not in map -> fall back to the repair's evidence source
    r = RepairCandidate(target_ref="r9::x", evidence=["scraper"])
    assert f(r) == "src:scraper"
    # neither -> default
    assert f(RepairCandidate(target_ref="r9::x")) == "src:unknown"
