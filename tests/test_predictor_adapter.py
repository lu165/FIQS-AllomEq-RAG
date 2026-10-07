import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "code"),
)

from allomeq_rag.predictor_adapter import (
    PredictorStatus,
    required_predictors_from_fedb,
    supplied_predictors_from_calc_params,
    supplied_predictors_from_question,
    check_required_predictors,
)


# Standard D/H equation.
r = required_predictors_from_fedb(
    ["D", "H"]
)

assert r.canonical_variables == {
    "DBH",
    "HEIGHT",
}
assert not r.unknown_variables


# DBH aliases.
r = required_predictors_from_fedb(
    ["DBH"]
)

assert r.canonical_variables == {
    "DBH"
}


r = required_predictors_from_fedb(
    ["D_{1.3}", "H"]
)

assert r.canonical_variables == {
    "DBH",
    "HEIGHT",
}


# D0 must not silently collapse into DBH.
r = required_predictors_from_fedb(
    ["D0", "H"]
)

assert r.canonical_variables == {
    "D0",
    "HEIGHT",
}


# Malformed historical field must be UNKNOWN.
r = required_predictors_from_fedb(
    [
        "D_{0}",
        "D_{0)-0.0048*(1.5369+0.7152*D_{0}",
    ]
)

assert "D0" in r.canonical_variables
assert len(r.unknown_variables) == 1


# Do alias from benchmark.
s = supplied_predictors_from_calc_params(
    {
        "D": 20,
        "H": 15,
        "note": "example",
        "user_equation": "ignored",
    }
)

assert s == {
    "DBH",
    "HEIGHT",
}


# Question extraction.
s = supplied_predictors_from_question(
    "某树胸径为20 cm，树高为15 m，求生物量。"
)

assert "DBH" in s
assert "HEIGHT" in s


# Complete requirement.
c = check_required_predictors(
    ["D", "H"],
    calc_params={
        "D": 20,
        "H": 15,
    },
)

assert c.status == PredictorStatus.SATISFIED


# Missing structured params alone must NOT become FAIL.
c = check_required_predictors(
    ["D", "H"],
    calc_params=None,
    question="请选择适用的生物量方程。",
)

assert c.status == PredictorStatus.UNKNOWN


# Question can establish supplied predictors.
c = check_required_predictors(
    ["D", "H"],
    calc_params=None,
    question="胸径20 cm，树高15 m。",
)

assert c.status == PredictorStatus.SATISFIED


# Malformed required variable remains UNKNOWN.
c = check_required_predictors(
    [
        "D_{0}",
        "D_{0)-0.0048*(1.5369+0.7152*D_{0}",
    ],
    calc_params={
        "Do": 10,
    },
)

assert c.status == PredictorStatus.UNKNOWN


print(
    "PASS: conservative predictor adapter"
)
