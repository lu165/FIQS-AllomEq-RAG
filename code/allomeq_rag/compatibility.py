"""
Manuscript-aligned reconstructed compatibility layer for AllomEq-RAG.

IMPORTANT
---------
This module is NOT claimed to be the historical implementation that
generated earlier experimental results.

Historical auditing showed that:
1. FEDB does not natively contain complete genus/family/climate-zone fields.
2. The historical taxonomy cache covers only a subset of FEDB tree types.
3. Historical geographic resources provide strong location coverage, but
   the location-generalization cache does not contain a complete
   deterministic numeric implementation of the manuscript scoring scheme.

Therefore this module implements the scoring semantics described in the
manuscript while refusing to infer unavailable taxonomy/geography metadata.

Generalized matching is used for candidate discovery. It must not be
confused with the final explicit applicability gate.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


SPECIES_DISCOVERY_THRESHOLD = 0.70
GEOGRAPHY_DISCOVERY_THRESHOLD = 0.50


class TaxonomicRelation(str, Enum):
    EXACT_SPECIES = "exact_species"
    SAME_GENUS = "same_genus"
    SAME_FAMILY = "same_family"
    UNRELATED = "unrelated"
    UNKNOWN = "unknown"


class GeographicRelation(str, Enum):
    EXACT_LOCATION = "exact_location"
    ADJACENT_GEOGRAPHIC_ZONE = "adjacent_geographic_zone"
    SAME_BROADER_CLIMATIC_ZONE = "same_broader_climatic_zone"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CompatibilityScore:
    relation: str
    score: Optional[float]
    metadata_known: bool
    discovery_pass: Optional[bool]
    reason: str


# ============================================================
# TAXONOMIC GENERALIZED MATCHING
# ============================================================

def taxonomic_score(
    relation: TaxonomicRelation,
) -> CompatibilityScore:
    """
    Manuscript generalized taxonomic similarity:

        exact species = 1.0
        same genus    = 0.8
        same family   = 0.5

    Species discovery threshold = 0.7.

    UNKNOWN never receives an invented score.
    """

    if relation == TaxonomicRelation.EXACT_SPECIES:
        score = 1.0

    elif relation == TaxonomicRelation.SAME_GENUS:
        score = 0.8

    elif relation == TaxonomicRelation.SAME_FAMILY:
        score = 0.5

    elif relation == TaxonomicRelation.UNRELATED:
        score = 0.0

    elif relation == TaxonomicRelation.UNKNOWN:
        return CompatibilityScore(
            relation=relation.value,
            score=None,
            metadata_known=False,
            discovery_pass=None,
            reason=(
                "Taxonomic metadata are insufficient; "
                "no similarity score was inferred."
            ),
        )

    else:
        raise ValueError(
            f"Unsupported taxonomic relation: {relation}"
        )

    return CompatibilityScore(
        relation=relation.value,
        score=score,
        metadata_known=True,
        discovery_pass=(
            score >= SPECIES_DISCOVERY_THRESHOLD
        ),
        reason=(
            f"Taxonomic generalized similarity={score:.1f}; "
            f"candidate-discovery threshold="
            f"{SPECIES_DISCOVERY_THRESHOLD:.1f}."
        ),
    )


# ============================================================
# GEOGRAPHIC GENERALIZED MATCHING
# ============================================================

def geographic_score(
    relation: GeographicRelation,
) -> CompatibilityScore:
    """
    Manuscript generalized geographic similarity.

    Exact location is represented as 1.0.

    The manuscript-defined generalized relationships are:

        adjacent geographic zone   = 0.7
        same broader climatic zone = 0.5

    Geography discovery threshold = 0.5.

    UNKNOWN never receives an invented score.
    """

    if relation == GeographicRelation.EXACT_LOCATION:
        score = 1.0

    elif (
        relation
        == GeographicRelation.ADJACENT_GEOGRAPHIC_ZONE
    ):
        score = 0.7

    elif (
        relation
        == GeographicRelation.SAME_BROADER_CLIMATIC_ZONE
    ):
        score = 0.5

    elif relation == GeographicRelation.INCOMPATIBLE:
        score = 0.0

    elif relation == GeographicRelation.UNKNOWN:
        return CompatibilityScore(
            relation=relation.value,
            score=None,
            metadata_known=False,
            discovery_pass=None,
            reason=(
                "Geographic metadata are insufficient; "
                "no similarity score was inferred."
            ),
        )

    else:
        raise ValueError(
            f"Unsupported geographic relation: {relation}"
        )

    return CompatibilityScore(
        relation=relation.value,
        score=score,
        metadata_known=True,
        discovery_pass=(
            score >= GEOGRAPHY_DISCOVERY_THRESHOLD
        ),
        reason=(
            f"Geographic generalized similarity={score:.1f}; "
            f"candidate-discovery threshold="
            f"{GEOGRAPHY_DISCOVERY_THRESHOLD:.1f}."
        ),
    )


# ============================================================
# DISCOVERY HELPERS
# ============================================================

def passes_species_discovery(
    relation: TaxonomicRelation,
) -> Optional[bool]:
    return taxonomic_score(
        relation
    ).discovery_pass


def passes_geography_discovery(
    relation: GeographicRelation,
) -> Optional[bool]:
    return geographic_score(
        relation
    ).discovery_pass
