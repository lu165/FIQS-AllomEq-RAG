"""
Final manuscript-aligned reconstructed applicability gate for AllomEq-RAG.

IMPORTANT
---------
This is a reconstructed reproducibility implementation. It is not claimed
to be the historical implementation that generated earlier results.

The final applicability decision is based on five explicit constraints:

1. taxonomic/species compatibility
2. geographic compatibility
3. biomass-component compatibility
4. required-predictor availability
5. documented measurement-range compatibility

Decision policy
---------------
Any explicit FAIL       -> FAIL
All five explicit PASS  -> PASS
Otherwise               -> UNRESOLVED

Generalized taxonomy/geography similarity used during candidate discovery
must not be substituted for the final explicit applicability decision.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .predictor_adapter import (
    PredictorStatus,
    check_required_predictors,
)

from .range_adapter import (
    RangeStatus,
    check_dbh_range,
)


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ConstraintEvidence:
    name: str
    status: GateStatus
    reason: str


@dataclass(frozen=True)
class ApplicabilityDecision:
    status: GateStatus
    constraints: Tuple[ConstraintEvidence, ...]
    failed_constraints: Tuple[str, ...]
    unresolved_constraints: Tuple[str, ...]
    action: str
    reason: str


# ============================================================
# BASIC EXPLICIT CONSTRAINTS
# ============================================================

def explicit_species_constraint(
    compatible: Optional[bool],
    reason: Optional[str] = None,
) -> ConstraintEvidence:
    """
    Final species/taxonomic applicability evidence.

    compatible=True:
        explicit evidence establishes compatibility.

    compatible=False:
        explicit evidence establishes incompatibility.

    compatible=None:
        evidence is insufficient.

    Candidate-discovery similarity must be resolved upstream and must not
    be passed here as if it were final compatibility evidence.
    """

    if compatible is True:
        return ConstraintEvidence(
            name="species",
            status=GateStatus.PASS,
            reason=reason or (
                "Explicit taxonomic/species compatibility established."
            ),
        )

    if compatible is False:
        return ConstraintEvidence(
            name="species",
            status=GateStatus.FAIL,
            reason=reason or (
                "Explicit taxonomic/species incompatibility established."
            ),
        )

    return ConstraintEvidence(
        name="species",
        status=GateStatus.UNRESOLVED,
        reason=reason or (
            "Taxonomic/species compatibility could not be "
            "established deterministically."
        ),
    )


def explicit_geography_constraint(
    compatible: Optional[bool],
    reason: Optional[str] = None,
) -> ConstraintEvidence:

    if compatible is True:
        return ConstraintEvidence(
            name="geography",
            status=GateStatus.PASS,
            reason=reason or (
                "Explicit geographic compatibility established."
            ),
        )

    if compatible is False:
        return ConstraintEvidence(
            name="geography",
            status=GateStatus.FAIL,
            reason=reason or (
                "Explicit geographic incompatibility established."
            ),
        )

    return ConstraintEvidence(
        name="geography",
        status=GateStatus.UNRESOLVED,
        reason=reason or (
            "Geographic compatibility could not be "
            "established deterministically."
        ),
    )


def component_constraint(
    query_component: object,
    equation_component: object,
) -> ConstraintEvidence:
    """
    Conservative exact normalized component check.

    No semantic synonym is invented here. Synonym normalization, if used,
    must be provided by a separately documented upstream component mapper.
    """

    if query_component is None or equation_component is None:
        return ConstraintEvidence(
            name="component",
            status=GateStatus.UNRESOLVED,
            reason="Biomass-component information is incomplete.",
        )

    q = str(query_component).strip()
    e = str(equation_component).strip()

    if not q or not e:
        return ConstraintEvidence(
            name="component",
            status=GateStatus.UNRESOLVED,
            reason="Biomass-component information is empty.",
        )

    if q == e:
        return ConstraintEvidence(
            name="component",
            status=GateStatus.PASS,
            reason="Biomass components match exactly.",
        )

    return ConstraintEvidence(
        name="component",
        status=GateStatus.FAIL,
        reason=(
            f"Biomass-component mismatch: query={q!r}, "
            f"equation={e!r}."
        ),
    )


# ============================================================
# PREDICTOR CONSTRAINT
# ============================================================

def predictor_constraint(
    fedb_variables: object,
    calc_params: object = None,
    question: object = None,
) -> ConstraintEvidence:

    result = check_required_predictors(
        fedb_variables=fedb_variables,
        calc_params=calc_params,
        question=question,
    )

    if result.status == PredictorStatus.SATISFIED:
        return ConstraintEvidence(
            name="predictors",
            status=GateStatus.PASS,
            reason=result.reason,
        )

    # Current conservative adapter intentionally does not infer FAIL
    # merely from absent structured evidence.
    if result.status == PredictorStatus.MISSING:
        return ConstraintEvidence(
            name="predictors",
            status=GateStatus.FAIL,
            reason=result.reason,
        )

    return ConstraintEvidence(
        name="predictors",
        status=GateStatus.UNRESOLVED,
        reason=result.reason,
    )


# ============================================================
# MEASUREMENT-RANGE CONSTRAINT
# ============================================================

def measurement_range_constraint(
    size_class: object,
    dbh: object,
) -> ConstraintEvidence:
    """
    Uses the exact historical DBH-range parser wrapped by the public
    three-state range adapter.
    """

    result = check_dbh_range(
        dbh=dbh,
        size_class=size_class,
    )

    if result.status == RangeStatus.IN_RANGE:
        return ConstraintEvidence(
            name="measurement_range",
            status=GateStatus.PASS,
            reason=result.reason,
        )

    if result.status == RangeStatus.OUT_OF_RANGE:
        return ConstraintEvidence(
            name="measurement_range",
            status=GateStatus.FAIL,
            reason=result.reason,
        )

    return ConstraintEvidence(
        name="measurement_range",
        status=GateStatus.UNRESOLVED,
        reason=result.reason,
    )


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_constraints(
    *constraints: ConstraintEvidence,
) -> ApplicabilityDecision:

    if len(constraints) != 5:
        raise ValueError(
            "The final applicability gate requires exactly "
            "five constraints."
        )

    names = tuple(c.name for c in constraints)

    expected = {
        "species",
        "geography",
        "component",
        "predictors",
        "measurement_range",
    }

    if set(names) != expected or len(set(names)) != 5:
        raise ValueError(
            "The five constraints must be exactly: "
            "species, geography, component, predictors, "
            "measurement_range."
        )

    failed = tuple(
        c.name
        for c in constraints
        if c.status == GateStatus.FAIL
    )

    unresolved = tuple(
        c.name
        for c in constraints
        if c.status == GateStatus.UNRESOLVED
    )

    if failed:
        return ApplicabilityDecision(
            status=GateStatus.FAIL,
            constraints=tuple(constraints),
            failed_constraints=failed,
            unresolved_constraints=unresolved,
            action="warning_or_refusal",
            reason=(
                "At least one explicit applicability constraint failed."
            ),
        )

    if unresolved:
        return ApplicabilityDecision(
            status=GateStatus.UNRESOLVED,
            constraints=tuple(constraints),
            failed_constraints=tuple(),
            unresolved_constraints=unresolved,
            action="clarification_or_warning",
            reason=(
                "No explicit constraint failed, but at least one "
                "constraint lacks sufficient evidence."
            ),
        )

    return ApplicabilityDecision(
        status=GateStatus.PASS,
        constraints=tuple(constraints),
        failed_constraints=tuple(),
        unresolved_constraints=tuple(),
        action="candidate_may_be_recommended",
        reason=(
            "All five explicit applicability constraints passed."
        ),
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def evaluate_applicability(
    *,
    species_compatible: Optional[bool],
    geography_compatible: Optional[bool],
    query_component: object,
    equation_component: object,
    fedb_variables: object,
    size_class: object,
    dbh: object = None,
    calc_params: object = None,
    question: object = None,
    species_reason: Optional[str] = None,
    geography_reason: Optional[str] = None,
) -> ApplicabilityDecision:

    constraints = (
        explicit_species_constraint(
            species_compatible,
            species_reason,
        ),
        explicit_geography_constraint(
            geography_compatible,
            geography_reason,
        ),
        component_constraint(
            query_component,
            equation_component,
        ),
        predictor_constraint(
            fedb_variables,
            calc_params=calc_params,
            question=question,
        ),
        measurement_range_constraint(
            size_class,
            dbh,
        ),
    )

    return aggregate_constraints(
        *constraints
    )
