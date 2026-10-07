"""
Verify that the public manuscript-aligned adapter produces exactly
the same parse_d_range() outputs as the historical abl_full.py
implementation for all 7,394 FEDB records.

The historical script is parsed with AST rather than imported.
"""

import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "code"),
)

from allomeq_rag.range_adapter import (
    parse_d_range as public_parser,
    check_dbh_range,
    RangeStatus,
)


HISTORICAL = Path(
    __import__("os").environ.get(
        "FIQS_HISTORICAL_PATH",
        "reproducibility/development_artifacts/"
        "ablation_framework_logic_closed_loop_snapshot.py"
    )
)

FEDB = Path(
    __import__("os").environ.get(
        "FIQS_FEDB_PATH",
        "data_public/fedb_public.json"
    )
)


# ============================================================
# Load exact historical function without importing abl_full.py
# ============================================================

source = HISTORICAL.read_text(encoding="utf-8")
tree = ast.parse(source)

node = None

for item in tree.body:
    if (
        isinstance(item, ast.FunctionDef)
        and item.name == "parse_d_range"
    ):
        node = item
        break

if node is None:
    raise RuntimeError(
        "Historical parse_d_range not found"
    )

module = ast.Module(
    body=[node],
    type_ignores=[],
)

ast.fix_missing_locations(module)

namespace = {
    "re": re,
}

exec(
    compile(
        module,
        str(HISTORICAL),
        "exec",
    ),
    namespace,
)

historical_parser = namespace["parse_d_range"]


# ============================================================
# Compare every FEDB record
# ============================================================

with FEDB.open("r", encoding="utf-8") as f:
    records = json.load(f)["equations"]


same = 0
different = []

historical_parsed = 0
public_parsed = 0


for record in records:

    sc = record.get("size_class")

    old = historical_parser(sc)
    new = public_parser(sc)

    if old is not None:
        historical_parsed += 1

    if new is not None:
        public_parsed += 1

    if old == new:
        same += 1
    else:
        different.append(
            {
                "id": record.get("id"),
                "size_class": sc,
                "historical": old,
                "public": new,
            }
        )


print("TOTAL              =", len(records))
print("IDENTICAL          =", same)
print("DIFFERENT          =", len(different))
print("HISTORICAL_PARSED  =", historical_parsed)
print("PUBLIC_PARSED      =", public_parsed)


if different:
    print("\nFIRST DIFFERENCES:")

    for item in different[:30]:
        print(item)


assert len(records) == 7394
assert len(different) == 0
assert historical_parsed == public_parsed

print(
    "PASS: public parser is exactly equivalent "
    "to historical parser on all FEDB records"
)


# ============================================================
# Three-state wrapper sanity tests
# ============================================================

r = check_dbh_range(
    10,
    "胸径范围：4.5-24.63cm",
)
assert r.status == RangeStatus.IN_RANGE


r = check_dbh_range(
    40,
    "胸径范围：4.5-24.63cm",
)
assert r.status == RangeStatus.OUT_OF_RANGE


r = check_dbh_range(
    10,
    "未指定",
)
assert r.status == RangeStatus.NOT_DOCUMENTED


r = check_dbh_range(
    None,
    "胸径范围：4.5-24.63cm",
)
assert r.status == RangeStatus.NOT_DOCUMENTED


print("PASS: three-state range wrapper")
