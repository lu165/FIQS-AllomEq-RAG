"""
Deterministic applicability checks for AllomEq-RAG.

This module is part of the manuscript-aligned reconstructed
implementation. It does not overwrite or claim identity with the
historical development scripts.

Implemented here:
1. component compatibility
2. required-predictor availability
3. documented measurement-range checking

Taxonomic/species and geographic compatibility are intentionally
handled separately because their manuscript-aligned rules require
explicit normalization/generalization policies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Set, Tuple


@dataclass
class ConstraintResult:
    name: str
    passed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_component(value: Any) -> str:
    """Normalize common biomass-component aliases."""
    text = _clean(value).lower()

    aliases = {
        "干": "stem",
        "树干": "stem",
        "stem": "stem",

        "枝": "branch",
        "树枝": "branch",
        "枝条": "branch",
        "branch": "branch",

        "叶": "leaf",
        "树叶": "leaf",
        "leaf": "leaf",
        "foliage": "leaf",

        "根": "root",
        "树根": "root",
        "root": "root",

        "皮": "bark",
        "树皮": "bark",
        "bark": "bark",

        "地上": "aboveground",
        "地上部分": "aboveground",
        "地上生物量": "aboveground",
        "aboveground": "aboveground",

        "地下": "belowground",
        "地下部分": "belowground",
        "belowground": "belowground",

        "全株": "whole_tree",
        "整株": "whole_tree",
        "总生物量": "whole_tree",
        "whole tree": "whole_tree",
        "whole_tree": "whole_tree",
    }

    return aliases.get(text, text)


def check_component(
    query_component: Any,
    equation_component: Any,
) -> ConstraintResult:

    q = normalize_component(query_component)
    e = normalize_component(equation_component)

    if not q:
        return ConstraintResult(
            "component",
            False,
            "Query component is missing."
        )

    if not e:
        return ConstraintResult(
            "component",
            False,
            "Equation component metadata is missing."
        )

    if q != e:
        return ConstraintResult(
            "component",
            False,
            f"Component mismatch: query={q}, equation={e}."
        )

    return ConstraintResult(
        "component",
        True,
        f"Component matched: {q}."
    )


def normalize_variables(values: Any) -> Set[str]:
    """Normalize predictor-variable metadata."""
    if values is None:
        return set()

    if isinstance(values, str):
        raw = re.split(r"[,，;/\s]+", values)
    elif isinstance(values, Iterable):
        raw = list(values)
    else:
        raw = [values]

    normalized = set()

    for value in raw:
        v = _clean(value).upper()

        if not v:
            continue

        aliases = {
            "DBH": "D",
            "胸径": "D",
            "TREE_HEIGHT": "H",
            "HEIGHT": "H",
            "树高": "H",
            "VOLUME": "V",
            "材积": "V",
            "AGE": "A",
            "林龄": "A",
        }

        normalized.add(aliases.get(v, v))

    return normalized


def check_required_predictors(
    supplied_variables: Any,
    equation_variables: Any,
) -> ConstraintResult:

    supplied = normalize_variables(supplied_variables)
    required = normalize_variables(equation_variables)

    if not required:
        return ConstraintResult(
            "required_predictors",
            False,
            "Equation predictor metadata is missing."
        )

    missing = required - supplied

    if missing:
        return ConstraintResult(
            "required_predictors",
            False,
            "Missing required predictor(s): "
            + ", ".join(sorted(missing))
        )

    return ConstraintResult(
        "required_predictors",
        True,
        "All required predictors are available."
    )


def parse_documented_range(
    size_class: Any,
) -> Optional[Tuple[float, float]]:
    """
    Parse a documented DBH/diameter range from common size_class text.

    Examples:
        '5-30 cm'
        '5～30cm'
        'DBH 5–30 cm'
        '胸径5~30 cm'
    """
    text = _clean(size_class)

    if not text:
        return None

    normalized = (
        text.replace("～", "-")
            .replace("~", "-")
            .replace("—", "-")
            .replace("–", "-")
    )

    patterns = [
        r"(?:dbh|胸径|直径)?\s*[:：]?\s*"
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*cm",

        r"(?:dbh|胸径|直径)\s*[:：]?\s*"
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        m = re.search(pattern, normalized, flags=re.IGNORECASE)
        if m:
            low = float(m.group(1))
            high = float(m.group(2))

            if low > high:
                low, high = high, low

            return low, high

    return None


def check_measurement_range(
    dbh: Optional[float],
    size_class: Any,
) -> ConstraintResult:

    if dbh is None:
        return ConstraintResult(
            "measurement_range",
            False,
            "Query DBH is missing."
        )

    documented = parse_documented_range(size_class)

    if documented is None:
        return ConstraintResult(
            "measurement_range",
            False,
            "No parseable documented DBH range is available."
        )

    low, high = documented
    value = float(dbh)

    if not low <= value <= high:
        return ConstraintResult(
            "measurement_range",
            False,
            f"DBH {value:g} cm is outside documented "
            f"range [{low:g}, {high:g}] cm."
        )

    return ConstraintResult(
        "measurement_range",
        True,
        f"DBH {value:g} cm is within documented "
        f"range [{low:g}, {high:g}] cm."
    )


def evaluate_basic_applicability(
    *,
    query_component: Any,
    equation_component: Any,
    supplied_variables: Any,
    equation_variables: Any,
    dbh: Optional[float],
    size_class: Any,
) -> Dict[str, Any]:

    checks = [
        check_component(
            query_component,
            equation_component,
        ),
        check_required_predictors(
            supplied_variables,
            equation_variables,
        ),
        check_measurement_range(
            dbh,
            size_class,
        ),
    ]

    return {
        "applicable": all(c.passed for c in checks),
        "checks": [c.to_dict() for c in checks],
    }
