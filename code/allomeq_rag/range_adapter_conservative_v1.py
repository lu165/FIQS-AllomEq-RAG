"""
Measurement-range adapter for the manuscript-aligned AllomEq-RAG
reconstructed implementation.

The DBH range parser preserves the historical parsing rules used in
the development code, while exposing an explicit three-state result:

    IN_RANGE
    OUT_OF_RANGE
    NOT_DOCUMENTED

NOT_DOCUMENTED means that no machine-checkable DBH range could be
established from the available size_class metadata. It must NOT be
silently interpreted as OUT_OF_RANGE.

Historical source:
    historical source: abl_full.py
    parse_d_range()

The historical source file is not modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class RangeStatus(str, Enum):
    IN_RANGE = "IN_RANGE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    NOT_DOCUMENTED = "NOT_DOCUMENTED"


@dataclass(frozen=True)
class RangeCheckResult:
    status: RangeStatus
    dbh: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    raw_size_class: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def parse_dbh_range(
    size_class: Any,
) -> Optional[Tuple[float, float]]:
    """
    Parse DBH range from real FEDB size_class metadata.

    This follows the historical parser family used by the v23/v24
    development implementation. The parser deliberately requires
    DBH/diameter semantics where necessary so that unrelated numbers
    such as stand age or tree height are not treated as DBH ranges.
    """

    if size_class is None:
        return None

    sc = str(size_class).strip()

    if not sc:
        return None

    # Normalize common dash variants only for matching.
    normalized = (
        sc.replace("～", "-")
          .replace("~", "-")
          .replace("–", "-")
          .replace("—", "-")
    )

    patterns = [
        # 胸径范围：4.5-24.63cm
        # 胸径5-30cm
        # 胸径为 5-30 cm
        r"胸径[范围：:为\s]*"
        r"(\d+(?:\.\d+)?)\s*-\s*"
        r"(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",

        # DBH range: 5.1-21.6 cm
        # DBH 5-30cm
        r"\bDBH\b[范围range：:=\s]*"
        r"(\d+(?:\.\d+)?)\s*-\s*"
        r"(\d+(?:\.\d+)?)\s*(?:cm)?",

        # 小树(胸径1-10cm)
        # already covered above, retained as explicit historical form
        r"\(\s*胸径[范围：:为\s]*"
        r"(\d+(?:\.\d+)?)\s*-\s*"
        r"(\d+(?:\.\d+)?)\s*(?:cm|厘米)?\s*\)",

        # 直径范围 5-30cm
        r"直径[范围：:为\s]*"
        r"(\d+(?:\.\d+)?)\s*-\s*"
        r"(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",

        # 5 <= DBH <= 30
        # 5 ≤ 胸径 ≤ 30
        r"(\d+(?:\.\d+)?)\s*(?:<=|≤)\s*"
        r"(?:DBH|胸径|直径|D)\s*"
        r"(?:<=|≤)\s*(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        m = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if not m:
            continue

        low = float(m.group(1))
        high = float(m.group(2))

        if low > high:
            low, high = high, low

        return low, high

    return None


def check_dbh_range(
    dbh: Optional[float],
    size_class: Any,
) -> RangeCheckResult:
    """
    Evaluate DBH against documented range metadata.

    Three-state semantics:

    IN_RANGE:
        a documented DBH range exists and DBH is inside it.

    OUT_OF_RANGE:
        a documented DBH range exists and DBH is outside it.

    NOT_DOCUMENTED:
        query DBH is absent, or no machine-checkable DBH range can be
        established from size_class.

    NOT_DOCUMENTED is intentionally distinct from OUT_OF_RANGE.
    """

    raw = "" if size_class is None else str(size_class)

    if dbh is None:
        return RangeCheckResult(
            status=RangeStatus.NOT_DOCUMENTED,
            dbh=None,
            lower=None,
            upper=None,
            raw_size_class=raw,
            reason="Query DBH is not available.",
        )

    documented = parse_dbh_range(size_class)

    if documented is None:
        return RangeCheckResult(
            status=RangeStatus.NOT_DOCUMENTED,
            dbh=float(dbh),
            lower=None,
            upper=None,
            raw_size_class=raw,
            reason=(
                "No machine-checkable documented DBH range "
                "could be established."
            ),
        )

    low, high = documented
    value = float(dbh)

    if value < low or value > high:
        return RangeCheckResult(
            status=RangeStatus.OUT_OF_RANGE,
            dbh=value,
            lower=low,
            upper=high,
            raw_size_class=raw,
            reason=(
                f"DBH {value:g} cm is outside documented "
                f"range [{low:g}, {high:g}] cm."
            ),
        )

    return RangeCheckResult(
        status=RangeStatus.IN_RANGE,
        dbh=value,
        lower=low,
        upper=high,
        raw_size_class=raw,
        reason=(
            f"DBH {value:g} cm is within documented "
            f"range [{low:g}, {high:g}] cm."
        ),
    )
