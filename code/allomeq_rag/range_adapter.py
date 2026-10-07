"""
Measurement-range adapter for the manuscript-aligned reconstructed
AllomEq-RAG implementation.

Important provenance
--------------------
The DBH parsing logic below is preserved from the historical
development implementation (abl_full.py / parse_d_range).

The historical source file itself is NOT modified.

This module adds a three-state wrapper around that parser:

    IN_RANGE
    OUT_OF_RANGE
    NOT_DOCUMENTED

NOT_DOCUMENTED must not be silently interpreted as OUT_OF_RANGE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional


# ============================================================
# HISTORICAL PARSER
#
# This block will be replaced automatically below by the exact
# AST-extracted historical implementation.
# ============================================================

def parse_d_range(sc):
    """v23增强：支持真实数据中的多种size_class格式"""
    if not sc: return None
    sc_str = str(sc)
    for pat in [
        # 格式1：胸径X-Ycm
        r'\u80f8\u5f84[\u8303\u56f4\uff1a:为]*\s*(\d+\.?\d*)\s*[-\u2013~\uff5e]\s*(\d+\.?\d*)\s*(?:cm|\u5398\u7c73)',
        # 格式2：胸径X, Ycm（逗号分隔）
        r'\u80f8\u5f84[\u8303\u56f4\uff1a:为]*\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*(?:cm|\u5398\u7c73)',
        # 格式3：D=X-Ycm
        r'D\s*[=:\uff1a]\s*(\d+\.?\d*)\s*[-\u2013~\uff5e]\s*(\d+\.?\d*)\s*(?:cm|\u5398\u7c73)',
        # 格式4：径级X-Y
        r'\u5f84\u7ea7\s*(\d+\.?\d*)\s*[-\u2013~\uff5e]\s*(\d+\.?\d*)',
        # 格式5：纯数字X-Ycm
        r'(\d+\.?\d*)\s*[-\u2013~\uff5e]\s*(\d+\.?\d*)\s*cm',
        # 格式6：胸径 X Y cm（空格分隔）
        r'\u80f8\u5f84\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*(?:cm|\u5398\u7c73)',
    ]:
        m = re.search(pat, sc_str, re.IGNORECASE)
        if m:
            try:
                lo, hi = float(m.group(1)), float(m.group(2))
                # 自动纠正顺序颠倒
                if lo > hi: lo, hi = hi, lo
                if lo < hi: return lo, hi
            except:
                continue
    return None


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
        result = asdict(self)
        result["status"] = self.status.value
        return result


def check_dbh_range(
    dbh: Optional[float],
    size_class: Any,
) -> RangeCheckResult:
    """
    Apply the historical DBH-range parser and expose explicit
    three-state semantics.
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

    documented = parse_d_range(size_class)

    if documented is None:
        return RangeCheckResult(
            status=RangeStatus.NOT_DOCUMENTED,
            dbh=float(dbh),
            lower=None,
            upper=None,
            raw_size_class=raw,
            reason=(
                "No machine-checkable documented DBH range "
                "could be established from size_class."
            ),
        )

    low, high = documented
    low = float(low)
    high = float(high)

    if low > high:
        low, high = high, low

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
