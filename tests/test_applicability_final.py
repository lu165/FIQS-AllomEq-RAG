import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "code"),
)

from allomeq_rag.applicability_final import (
    GateStatus,
    ConstraintEvidence,
    aggregate_constraints,
    evaluate_applicability,
)


def C(name, status):
    return ConstraintEvidence(
        name=name,
        status=status,
        reason="test",
    )


NAMES = [
    "species",
    "geography",
    "component",
    "predictors",
    "measurement_range",
]


# ============================================================
# Aggregation truth table
# ============================================================

r = aggregate_constraints(
    *[
        C(name, GateStatus.PASS)
        for name in NAMES
    ]
)

assert r.status == GateStatus.PASS
assert r.action == "candidate_may_be_recommended"


r = aggregate_constraints(
    C("species", GateStatus.FAIL),
    C("geography", GateStatus.PASS),
    C("component", GateStatus.PASS),
    C("predictors", GateStatus.PASS),
    C("measurement_range", GateStatus.PASS),
)

assert r.status == GateStatus.FAIL
assert "species" in r.failed_constraints


r = aggregate_constraints(
    C("species", GateStatus.PASS),
    C("geography", GateStatus.UNRESOLVED),
    C("component", GateStatus.PASS),
    C("predictors", GateStatus.PASS),
    C("measurement_range", GateStatus.PASS),
)

assert r.status == GateStatus.UNRESOLVED
assert "geography" in r.unresolved_constraints


# FAIL must dominate UNRESOLVED.
r = aggregate_constraints(
    C("species", GateStatus.FAIL),
    C("geography", GateStatus.UNRESOLVED),
    C("component", GateStatus.PASS),
    C("predictors", GateStatus.PASS),
    C("measurement_range", GateStatus.PASS),
)

assert r.status == GateStatus.FAIL


# ============================================================
# End-to-end: fully documented PASS
# ============================================================

r = evaluate_applicability(
    species_compatible=True,
    geography_compatible=True,
    query_component="树干",
    equation_component="树干",
    fedb_variables=["D", "H"],
    size_class="胸径范围：5-30cm",
    dbh=20,
    calc_params={
        "D": 20,
        "H": 15,
    },
    question="胸径20 cm，树高15 m。",
)

assert r.status == GateStatus.PASS


# ============================================================
# End-to-end: explicit component FAIL
# ============================================================

r = evaluate_applicability(
    species_compatible=True,
    geography_compatible=True,
    query_component="树干",
    equation_component="树根",
    fedb_variables=["D", "H"],
    size_class="胸径范围：5-30cm",
    dbh=20,
    calc_params={
        "D": 20,
        "H": 15,
    },
)

assert r.status == GateStatus.FAIL
assert "component" in r.failed_constraints


# ============================================================
# End-to-end: explicit range FAIL
# ============================================================

r = evaluate_applicability(
    species_compatible=True,
    geography_compatible=True,
    query_component="树干",
    equation_component="树干",
    fedb_variables=["D", "H"],
    size_class="胸径范围：5-30cm",
    dbh=50,
    calc_params={
        "D": 50,
        "H": 15,
    },
)

assert r.status == GateStatus.FAIL
assert "measurement_range" in r.failed_constraints


# ============================================================
# Missing range documentation must NOT silently pass
# ============================================================

r = evaluate_applicability(
    species_compatible=True,
    geography_compatible=True,
    query_component="树干",
    equation_component="树干",
    fedb_variables=["D", "H"],
    size_class="未指定",
    dbh=20,
    calc_params={
        "D": 20,
        "H": 15,
    },
)

assert r.status == GateStatus.UNRESOLVED
assert "measurement_range" in r.unresolved_constraints


# ============================================================
# Missing predictor evidence must NOT automatically fail
# ============================================================

r = evaluate_applicability(
    species_compatible=True,
    geography_compatible=True,
    query_component="树干",
    equation_component="树干",
    fedb_variables=["D", "H"],
    size_class="胸径范围：5-30cm",
    dbh=20,
    calc_params=None,
    question="请选择适用的生物量方程。",
)

assert r.status == GateStatus.UNRESOLVED
assert "predictors" in r.unresolved_constraints


# ============================================================
# Unknown species compatibility must remain unresolved
# ============================================================

r = evaluate_applicability(
    species_compatible=None,
    geography_compatible=True,
    query_component="树干",
    equation_component="树干",
    fedb_variables=["D", "H"],
    size_class="胸径范围：5-30cm",
    dbh=20,
    calc_params={
        "D": 20,
        "H": 15,
    },
)

assert r.status == GateStatus.UNRESOLVED
assert "species" in r.unresolved_constraints


print(
    "PASS: final five-constraint tri-state applicability gate"
)
