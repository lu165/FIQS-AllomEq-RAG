import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "code"),
)

from allomeq_rag.compatibility import (
    SPECIES_DISCOVERY_THRESHOLD,
    GEOGRAPHY_DISCOVERY_THRESHOLD,
    TaxonomicRelation,
    GeographicRelation,
    taxonomic_score,
    geographic_score,
)


assert SPECIES_DISCOVERY_THRESHOLD == 0.70
assert GEOGRAPHY_DISCOVERY_THRESHOLD == 0.50


# ============================================================
# TAXONOMY
# ============================================================

r = taxonomic_score(
    TaxonomicRelation.EXACT_SPECIES
)
assert r.score == 1.0
assert r.discovery_pass is True


r = taxonomic_score(
    TaxonomicRelation.SAME_GENUS
)
assert r.score == 0.8
assert r.discovery_pass is True


r = taxonomic_score(
    TaxonomicRelation.SAME_FAMILY
)
assert r.score == 0.5
assert r.discovery_pass is False


r = taxonomic_score(
    TaxonomicRelation.UNRELATED
)
assert r.score == 0.0
assert r.discovery_pass is False


r = taxonomic_score(
    TaxonomicRelation.UNKNOWN
)
assert r.score is None
assert r.metadata_known is False
assert r.discovery_pass is None


# ============================================================
# GEOGRAPHY
# ============================================================

r = geographic_score(
    GeographicRelation.EXACT_LOCATION
)
assert r.score == 1.0
assert r.discovery_pass is True


r = geographic_score(
    GeographicRelation.ADJACENT_GEOGRAPHIC_ZONE
)
assert r.score == 0.7
assert r.discovery_pass is True


# Boundary is intentionally inclusive.
r = geographic_score(
    GeographicRelation.SAME_BROADER_CLIMATIC_ZONE
)
assert r.score == 0.5
assert r.discovery_pass is True


r = geographic_score(
    GeographicRelation.INCOMPATIBLE
)
assert r.score == 0.0
assert r.discovery_pass is False


r = geographic_score(
    GeographicRelation.UNKNOWN
)
assert r.score is None
assert r.metadata_known is False
assert r.discovery_pass is None


print(
    "PASS: manuscript-aligned compatibility semantics"
)
