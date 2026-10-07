"""
Manuscript-aligned structural score for AllomEq-RAG.

Reconstructed from the manuscript specification.

Maximum score = 1000:
    taxonomy/species       250
    geography              250
    diameter/range         300
    in-range bonus          50
    component              100
    predictor completeness  50
                           ----
                           1000

IMPORTANT:
This module performs scoring for candidate discovery/ranking only.
It does NOT replace the final five-constraint applicability gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


MAX_SCORE = 1000.0

SPECIES_MAX = 250.0
GEOGRAPHY_MAX = 250.0
RANGE_BASE_MAX = 300.0
RANGE_BONUS_MAX = 50.0
COMPONENT_MAX = 100.0
PREDICTOR_MAX = 50.0


@dataclass(frozen=True)
class StructuralScore:
    species: float
    geography: float
    range_base: float
    range_bonus: float
    component: float
    predictors: float
    total: float


def _unit_interval(
    value: Optional[float],
    name: str,
) -> float:
    """
    Convert a known similarity/compatibility value to [0,1].

    Unknown values are represented by None and contribute zero
    to discovery scoring. They must NOT be interpreted as final
    applicability failure.
    """
    if value is None:
        return 0.0

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be in [0,1] or None; got {value}"
        )

    return value


def score_structural(
    *,
    species_similarity: Optional[float],
    geography_similarity: Optional[float],
    range_similarity: Optional[float],
    range_in_documented_bounds: Optional[bool],
    component_similarity: Optional[float],
    predictor_completeness: Optional[float],
) -> StructuralScore:

    species_u = _unit_interval(
        species_similarity,
        "species_similarity",
    )

    geography_u = _unit_interval(
        geography_similarity,
        "geography_similarity",
    )

    range_u = _unit_interval(
        range_similarity,
        "range_similarity",
    )

    component_u = _unit_interval(
        component_similarity,
        "component_similarity",
    )

    predictor_u = _unit_interval(
        predictor_completeness,
        "predictor_completeness",
    )

    species = SPECIES_MAX * species_u
    geography = GEOGRAPHY_MAX * geography_u
    range_base = RANGE_BASE_MAX * range_u

    # Bonus is awarded only when documented range evidence
    # explicitly establishes that the query is within bounds.
    range_bonus = (
        RANGE_BONUS_MAX
        if range_in_documented_bounds is True
        else 0.0
    )

    component = COMPONENT_MAX * component_u
    predictors = PREDICTOR_MAX * predictor_u

    total = (
        species
        + geography
        + range_base
        + range_bonus
        + component
        + predictors
    )

    if not 0.0 <= total <= MAX_SCORE:
        raise AssertionError(
            f"structural total outside [0,1000]: {total}"
        )

    return StructuralScore(
        species=species,
        geography=geography,
        range_base=range_base,
        range_bonus=range_bonus,
        component=component,
        predictors=predictors,
        total=total,
    )


if __name__ == "__main__":

    perfect = score_structural(
        species_similarity=1.0,
        geography_similarity=1.0,
        range_similarity=1.0,
        range_in_documented_bounds=True,
        component_similarity=1.0,
        predictor_completeness=1.0,
    )

    assert perfect.total == 1000.0

    unknown = score_structural(
        species_similarity=None,
        geography_similarity=None,
        range_similarity=None,
        range_in_documented_bounds=None,
        component_similarity=None,
        predictor_completeness=None,
    )

    assert unknown.total == 0.0

    partial = score_structural(
        species_similarity=0.8,
        geography_similarity=0.5,
        range_similarity=1.0,
        range_in_documented_bounds=True,
        component_similarity=1.0,
        predictor_completeness=1.0,
    )

    expected = (
        250 * 0.8
        + 250 * 0.5
        + 300
        + 50
        + 100
        + 50
    )

    assert partial.total == expected
    assert partial.total == 825.0

    print("PASS: manuscript structural score")
    print("PERFECT =", perfect.total)
    print("UNKNOWN =", unknown.total)
    print("PARTIAL =", partial.total)
