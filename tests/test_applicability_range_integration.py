import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "code"),
)

from allomeq_rag import applicability_v3 as app


# ============================================================
# A. IN RANGE
# ============================================================

r = app.check_measurement_range(
    dbh=10,
    size_class="胸径范围：4.5-24.63cm",
)

assert r.passed is True
assert "within documented" in r.reason


# ============================================================
# B. OUT OF RANGE
# ============================================================

r = app.check_measurement_range(
    dbh=40,
    size_class="胸径范围：4.5-24.63cm",
)

assert r.passed is False
assert "outside documented" in r.reason


# ============================================================
# C. NOT DOCUMENTED
# ============================================================

r = app.check_measurement_range(
    dbh=10,
    size_class="未指定",
)

assert r.passed is True
assert r.reason.startswith("NOT_DOCUMENTED:")


# ============================================================
# D. QUERY DBH MISSING
# ============================================================

r = app.check_measurement_range(
    dbh=None,
    size_class="胸径范围：4.5-24.63cm",
)

assert r.passed is True
assert r.reason.startswith("NOT_DOCUMENTED:")


# ============================================================
# E. COMPLETE FIVE-CHECK POSITIVE CASE
# ============================================================

r = app.evaluate_applicability(
    species_compatibility_score=0.70,
    geography_compatibility_score=0.50,
    query_component="树干",
    equation_component="干",
    supplied_variables=["D", "H"],
    equation_variables=["D", "H"],
    dbh=20,
    size_class="胸径范围：4.5-24.63cm",
)

assert r["applicable"] is True
assert len(r["checks"]) == 5


# ============================================================
# F. RANGE FAILURE MUST FAIL COMPLETE GATE
# ============================================================

r = app.evaluate_applicability(
    species_compatibility_score=0.70,
    geography_compatibility_score=0.50,
    query_component="树干",
    equation_component="干",
    supplied_variables=["D", "H"],
    equation_variables=["D", "H"],
    dbh=40,
    size_class="胸径范围：4.5-24.63cm",
)

assert r["applicable"] is False
assert "measurement_range" in r["failed_constraints"]


# ============================================================
# G. UNDOCUMENTED RANGE MUST NOT BE FALSELY LABELED
#    AS AN EXPLICIT RANGE MISMATCH
# ============================================================

r = app.evaluate_applicability(
    species_compatibility_score=0.70,
    geography_compatibility_score=0.50,
    query_component="树干",
    equation_component="干",
    supplied_variables=["D", "H"],
    equation_variables=["D", "H"],
    dbh=20,
    size_class="未指定",
)

range_check = [
    x
    for x in r["checks"]
    if x["name"] == "measurement_range"
][0]

assert range_check["passed"] is True
assert range_check["reason"].startswith(
    "NOT_DOCUMENTED:"
)

print(
    "PASS: applicability v3 range-adapter integration"
)
