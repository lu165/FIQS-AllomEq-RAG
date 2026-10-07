"""
Manuscript-aligned predictor adapter for AllomEq-RAG.

This is a reconstructed reproducibility implementation.

Principles
----------
1. Preserve the original FEDB `variables` values.
2. Canonicalize only mappings that can be interpreted deterministically.
3. Do not treat arbitrary benchmark `_calc_params` keys as predictors.
4. Do not silently repair malformed historical FEDB variable strings.
5. Missing metadata is UNKNOWN, not automatically FAIL.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, FrozenSet
import re


class PredictorStatus(str, Enum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PredictorRequirement:
    raw_variables: tuple[str, ...]
    canonical_variables: FrozenSet[str]
    unknown_variables: tuple[str, ...]


@dataclass(frozen=True)
class PredictorCheck:
    status: PredictorStatus
    required: FrozenSet[str]
    supplied: FrozenSet[str]
    missing: FrozenSet[str]
    unknown_required: tuple[str, ...]
    reason: str


# ============================================================
# Canonical predictor vocabulary
# ============================================================

# Diameter-like variables are deliberately kept distinct when
# their measurement position/type is explicitly different.

EXACT_MAP = {
    "D": "DBH",
    "DBH": "DBH",
    "D_{1.3}": "DBH",

    "H": "HEIGHT",

    "D0": "D0",
    "D_{0}": "D0",

    "D0.2": "D0.2",
    "D0.3": "D0.3",

    "D_{轮}": "DIAMETER_WHEEL",
    "D_{围}": "CIRCUMFERENCE_DIAMETER",

    "V": "VOLUME",
    "V_total": "TOTAL_VOLUME",

    "A": "A",
    "B": "B",
    "B_total": "TOTAL_BIOMASS",

    "W_T": "TOTAL_BIOMASS",
    "W_aboveground": "ABOVEGROUND_BIOMASS",

    "W_干": "STEM_BIOMASS",
    "W_枝": "BRANCH_BIOMASS",
    "W_叶": "LEAF_BIOMASS",

    "Cc": "CC",
    "O": "O",
}


# Benchmark/user-side aliases.
SUPPLIED_ALIAS_MAP = {
    "D": "DBH",
    "DBH": "DBH",
    "dbh": "DBH",
    "胸径": "DBH",

    "H": "HEIGHT",
    "height": "HEIGHT",
    "HEIGHT": "HEIGHT",
    "树高": "HEIGHT",
    "height_m": "HEIGHT",

    "D0": "D0",
    "Do": "D0",
    "D_0": "D0",

    "V": "VOLUME",
    "A": "A",
    "B": "B",

    "W_T": "TOTAL_BIOMASS",
    "W_above": "ABOVEGROUND_BIOMASS",
    "W_aboveground": "ABOVEGROUND_BIOMASS",
}


# Keys known to be metadata, derivation text, diagnostics,
# converted formulas, or other non-predictor material.
NON_PREDICTOR_KEY_PATTERNS = (
    "equation",
    "formula",
    "note",
    "说明",
    "换算",
    "系数",
    "指数",
    "常数",
    "建议",
    "推导",
    "计算",
    "参数说明",
    "error",
    "result",
    "step",
    "intercept",
    "slope",
)


# ============================================================
# FEDB requirement parsing
# ============================================================

def canonicalize_required_variable(
    raw: object,
) -> Optional[str]:
    """
    Return a canonical predictor when the mapping is known.

    Unknown/malformed values return None rather than being guessed.
    """

    if raw is None:
        return None

    s = str(raw).strip()

    if not s:
        return None

    if s in EXACT_MAP:
        return EXACT_MAP[s]

    # Detect obviously malformed equation fragments that leaked
    # into the historical variables field.
    if any(
        token in s
        for token in (
            "*",
            "+",
            "=",
            "(",
            ")",
        )
    ):
        return None

    return None


def required_predictors_from_fedb(
    variables: object,
) -> PredictorRequirement:
    """
    Convert a FEDB `variables` list into conservative canonical
    predictor requirements.
    """

    if not isinstance(variables, (list, tuple)):
        return PredictorRequirement(
            raw_variables=tuple(),
            canonical_variables=frozenset(),
            unknown_variables=(
                "<variables_not_list>",
            ),
        )

    raw_values = tuple(
        str(x).strip()
        for x in variables
    )

    canonical = set()
    unknown = []

    for raw in raw_values:

        mapped = canonicalize_required_variable(
            raw
        )

        if mapped is None:
            unknown.append(raw)
        else:
            canonical.add(mapped)

    return PredictorRequirement(
        raw_variables=raw_values,
        canonical_variables=frozenset(
            canonical
        ),
        unknown_variables=tuple(
            unknown
        ),
    )


# ============================================================
# User / benchmark supplied predictors
# ============================================================

def _looks_non_predictor_key(
    key: str,
) -> bool:

    low = key.lower()

    return any(
        token.lower() in low
        for token in NON_PREDICTOR_KEY_PATTERNS
    )


def supplied_predictors_from_calc_params(
    calc_params: object,
) -> FrozenSet[str]:
    """
    Extract only recognized predictor keys from benchmark
    `_calc_params`.

    Absence of `_calc_params` is not interpreted as absence of
    user predictors.
    """

    if not isinstance(calc_params, dict):
        return frozenset()

    supplied = set()

    for key in calc_params.keys():

        s = str(key).strip()

        if not s:
            continue

        if _looks_non_predictor_key(s):
            continue

        mapped = SUPPLIED_ALIAS_MAP.get(s)

        if mapped is not None:
            supplied.add(mapped)

    return frozenset(supplied)


# ============================================================
# Conservative extraction from natural-language question
# ============================================================

def supplied_predictors_from_question(
    question: object,
) -> FrozenSet[str]:
    """
    Detect explicit measurement mentions in the query.

    This intentionally recognizes only conservative patterns.
    It does not infer missing measurements.
    """

    if question is None:
        return frozenset()

    text = str(question)

    supplied = set()

    # Explicit DBH / diameter measurements.
    dbh_patterns = [
        r"\bDBH\b",
        r"胸径",
        r"\bD\s*[=:：]?\s*\d",
        r"\bD\s*(?:为|是)\s*\d",
    ]

    if any(
        re.search(
            p,
            text,
            flags=re.IGNORECASE,
        )
        for p in dbh_patterns
    ):
        supplied.add("DBH")

    # Explicit height measurements.
    height_patterns = [
        r"树高",
        r"\bheight\b",
        r"\bH\s*[=:：]?\s*\d",
        r"\bH\s*(?:为|是)\s*\d",
    ]

    if any(
        re.search(
            p,
            text,
            flags=re.IGNORECASE,
        )
        for p in height_patterns
    ):
        supplied.add("HEIGHT")

    # Volume.
    if re.search(
        r"材积|\bvolume\b|\bV\s*[=:：]?\s*\d",
        text,
        flags=re.IGNORECASE,
    ):
        supplied.add("VOLUME")

    return frozenset(supplied)


def collect_supplied_predictors(
    calc_params: object = None,
    question: object = None,
) -> FrozenSet[str]:

    return frozenset(
        set(
            supplied_predictors_from_calc_params(
                calc_params
            )
        )
        |
        set(
            supplied_predictors_from_question(
                question
            )
        )
    )


# ============================================================
# Three-state requirement check
# ============================================================

def check_required_predictors(
    fedb_variables: object,
    calc_params: object = None,
    question: object = None,
) -> PredictorCheck:

    requirement = required_predictors_from_fedb(
        fedb_variables
    )

    supplied = collect_supplied_predictors(
        calc_params=calc_params,
        question=question,
    )

    # Unknown required variable means we cannot make a complete
    # deterministic applicability decision.
    if requirement.unknown_variables:

        return PredictorCheck(
            status=PredictorStatus.UNKNOWN,
            required=requirement.canonical_variables,
            supplied=supplied,
            missing=frozenset(),
            unknown_required=(
                requirement.unknown_variables
            ),
            reason=(
                "At least one FEDB required variable could not "
                "be mapped deterministically."
            ),
        )

    required = requirement.canonical_variables

    missing = required - supplied

    if not missing:

        return PredictorCheck(
            status=PredictorStatus.SATISFIED,
            required=required,
            supplied=supplied,
            missing=frozenset(),
            unknown_required=tuple(),
            reason=(
                "All deterministically identified required "
                "predictors are supplied."
            ),
        )

    # Critical policy:
    # Missing from `_calc_params` alone is insufficient evidence
    # that the user omitted the predictor. If neither structured
    # params nor query text establish the variable, the state is
    # UNKNOWN at this adapter layer.
    return PredictorCheck(
        status=PredictorStatus.UNKNOWN,
        required=required,
        supplied=supplied,
        missing=frozenset(missing),
        unknown_required=tuple(),
        reason=(
            "One or more required predictors were not observed "
            "in the available structured/query evidence; "
            "absence is not treated as proven omission."
        ),
    )
