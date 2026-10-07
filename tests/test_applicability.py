import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from allomeq_rag.applicability import (
    check_component,
    check_required_predictors,
    parse_documented_range,
    check_measurement_range,
    evaluate_basic_applicability,
)


# ------------------------------------------------------------
# Component
# ------------------------------------------------------------

assert check_component("树干", "干").passed
assert check_component("叶", "叶").passed
assert not check_component("枝", "叶").passed


# ------------------------------------------------------------
# Required predictors
# ------------------------------------------------------------

assert check_required_predictors(
    ["D", "H"],
    ["D", "H"],
).passed

assert check_required_predictors(
    ["DBH", "H"],
    ["D", "H"],
).passed

assert not check_required_predictors(
    ["D"],
    ["D", "H"],
).passed


# ------------------------------------------------------------
# Measurement range
# ------------------------------------------------------------

assert parse_documented_range("5-30 cm") == (5.0, 30.0)
assert parse_documented_range("胸径5～30cm") == (5.0, 30.0)
assert parse_documented_range("DBH 10–40 cm") == (10.0, 40.0)

assert check_measurement_range(
    20,
    "5-30 cm",
).passed

assert not check_measurement_range(
    40,
    "5-30 cm",
).passed


# ------------------------------------------------------------
# Combined basic gate
# ------------------------------------------------------------

result = evaluate_basic_applicability(
    query_component="树干",
    equation_component="干",
    supplied_variables=["D", "H"],
    equation_variables=["D", "H"],
    dbh=20,
    size_class="5-30 cm",
)

assert result["applicable"] is True


result = evaluate_basic_applicability(
    query_component="树干",
    equation_component="干",
    supplied_variables=["D"],
    equation_variables=["D", "H"],
    dbh=20,
    size_class="5-30 cm",
)

assert result["applicable"] is False

print("PASS: deterministic basic applicability tests")


# ------------------------------------------------------------
# Complete five-constraint applicability gate
# ------------------------------------------------------------

from allomeq_rag.applicability import (
    check_species_taxonomy,
    check_geography,
    evaluate_applicability,
    SPECIES_COMPATIBILITY_THRESHOLD,
    GEOGRAPHY_COMPATIBILITY_THRESHOLD,
)


assert SPECIES_COMPATIBILITY_THRESHOLD == 0.70
assert GEOGRAPHY_COMPATIBILITY_THRESHOLD == 0.50


# Boundary values must pass.
assert check_species_taxonomy(0.70).passed
assert check_geography(0.50).passed


# Values below the frozen thresholds must fail.
assert not check_species_taxonomy(0.699999).passed
assert not check_geography(0.499999).passed


# Missing compatibility evidence must fail closed.
assert not check_species_taxonomy(None).passed
assert not check_geography(None).passed


# Complete positive case.
full_result = evaluate_applicability(
    species_compatibility_score=0.90,
    geography_compatibility_score=0.80,
    query_component="树干",
    equation_component="干",
    supplied_variables=["D", "H"],
    equation_variables=["D", "H"],
    dbh=20,
    size_class="5-30 cm",
)

assert full_result["applicable"] is True
assert full_result["failed_constraints"] == []
assert len(full_result["checks"]) == 5


# Species failure.
full_result = evaluate_applicability(
    species_compatibility_score=0.69,
    geography_compatibility_score=0.80,
    query_component="树干",
    equation_component="干",
    supplied_variables=["D", "H"],
    equation_variables=["D", "H"],
    dbh=20,
    size_class="5-30 cm",
)

assert full_result["applicable"] is False
assert "species_taxonomy" in full_result["failed_constraints"]


# Geography failure.
full_result = evaluate_applicability(
    species_compatibility_score=0.90,
    geography_compatibility_score=0.49,
    query_component="树干",
    equation_component="干",
    supplied_variables=["D", "H"],
    equation_variables=["D", "H"],
    dbh=20,
    size_class="5-30 cm",
)

assert full_result["applicable"] is False
assert "geography" in full_result["failed_constraints"]


# Required-predictor failure.
full_result = evaluate_applicability(
    species_compatibility_score=0.90,
    geography_compatibility_score=0.80,
    query_component="树干",
    equation_component="干",
    supplied_variables=["D"],
    equation_variables=["D", "H"],
    dbh=20,
    size_class="5-30 cm",
)

assert full_result["applicable"] is False
assert "required_predictors" in full_result["failed_constraints"]


# Documented-range failure.
full_result = evaluate_applicability(
    species_compatibility_score=0.90,
    geography_compatibility_score=0.80,
    query_component="树干",
    equation_component="干",
    supplied_variables=["D", "H"],
    equation_variables=["D", "H"],
    dbh=40,
    size_class="5-30 cm",
)

assert full_result["applicable"] is False
assert "measurement_range" in full_result["failed_constraints"]


print("PASS: complete five-constraint applicability gate")
