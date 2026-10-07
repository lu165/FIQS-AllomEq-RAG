import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from allomeq_rag.range_adapter import (
    RangeStatus,
    parse_dbh_range,
    check_dbh_range,
)


# ------------------------------------------------------------
# Real FEDB formats observed in the audit
# ------------------------------------------------------------

assert parse_dbh_range(
    "林龄5-26年, 胸径范围：4.5-24.63cm"
) == (4.5, 24.63)

assert parse_dbh_range(
    "小树(胸径1-10cm)"
) == (1.0, 10.0)

assert parse_dbh_range(
    "中树(胸径10-20cm)"
) == (10.0, 20.0)

assert parse_dbh_range(
    "胸径范围: 5.1-21.6cm; 树高范围: 5.9-20.9m"
) == (5.1, 21.6)

assert parse_dbh_range(
    "胸径范围：1.75-31.7cm"
) == (1.75, 31.7)


# ------------------------------------------------------------
# Must not confuse unrelated ranges with DBH
# ------------------------------------------------------------

assert parse_dbh_range(
    "林龄5-26年"
) is None

assert parse_dbh_range(
    "树高范围: 5.9-20.9m"
) is None

assert parse_dbh_range(
    "未指定"
) is None

assert parse_dbh_range(
    ""
) is None


# ------------------------------------------------------------
# Three-state behavior
# ------------------------------------------------------------

r = check_dbh_range(
    20,
    "胸径范围：4.5-24.63cm",
)
assert r.status == RangeStatus.IN_RANGE


r = check_dbh_range(
    30,
    "胸径范围：4.5-24.63cm",
)
assert r.status == RangeStatus.OUT_OF_RANGE


r = check_dbh_range(
    20,
    "未指定",
)
assert r.status == RangeStatus.NOT_DOCUMENTED


r = check_dbh_range(
    None,
    "胸径范围：4.5-24.63cm",
)
assert r.status == RangeStatus.NOT_DOCUMENTED


# Boundary values pass.
assert check_dbh_range(
    4.5,
    "胸径范围：4.5-24.63cm",
).status == RangeStatus.IN_RANGE

assert check_dbh_range(
    24.63,
    "胸径范围：4.5-24.63cm",
).status == RangeStatus.IN_RANGE


print("PASS: range adapter unit tests")
