import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read-only audit of the real 522-query equation benchmark and
the 7,394-record FEDB.

NO model inference.
NO embeddings.
NO GPU.
NO modification of source data.
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter


FEDB = Path(os.environ.get("FIQS_FEDB_PATH","data_public/fedb_public.json"))

BENCHMARK = Path(
    "<PRIVATE_ROOT>/homee/"
    "最新生物量测试集构建_v7_with_neg_原始_20260503_131441.json"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def present(v):
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict, tuple, set)):
        return len(v) > 0
    return True


def parse_dbh_range(text):
    """
    Audit parser only.

    This deliberately recognizes several common historical
    size_class forms without modifying applicability.py.
    """
    if not present(text):
        return None

    s = str(text).strip()

    s = (
        s.replace("～", "-")
         .replace("~", "-")
         .replace("—", "-")
         .replace("–", "-")
    )

    # 5-30 cm / DBH 5-30 / 胸径5-30
    patterns = [
        r"(?:dbh|胸径|直径)?\s*[:：]?\s*"
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",

        # 5 ≤ DBH ≤ 30
        r"(\d+(?:\.\d+)?)\s*(?:<=|≤)\s*"
        r"(?:dbh|胸径|直径|d)\s*(?:<=|≤)\s*"
        r"(\d+(?:\.\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return (min(a, b), max(a, b))

    return None


def classify_expected(case):
    """
    Do NOT assume a label schema.

    Search likely fields and report what actually exists.
    """
    candidates = [
        "applicable",
        "is_applicable",
        "applicability",
        "label",
        "expected_behavior",
        "expected_result",
        "should_reject",
        "negative",
        "gen_type",
        "group",
    ]

    return {
        k: case.get(k)
        for k in candidates
        if k in case
    }


# ============================================================
# LOAD
# ============================================================

with FEDB.open("r", encoding="utf-8") as f:
    fedb_root = json.load(f)

with BENCHMARK.open("r", encoding="utf-8") as f:
    bench_root = json.load(f)


if isinstance(fedb_root, dict):
    fedb = fedb_root.get("equations", [])
else:
    fedb = fedb_root


if isinstance(bench_root, dict):
    benchmark = bench_root.get("test_cases", [])
else:
    benchmark = bench_root


print("=" * 78)
print("REAL DATA SCHEMA / COVERAGE AUDIT")
print("=" * 78)

print("\n[FILES]")
print("FEDB      :", FEDB)
print("Benchmark :", BENCHMARK)
print("FEDB SHA256      :", sha256(FEDB))
print("Benchmark SHA256 :", sha256(BENCHMARK))

print("\n[COUNTS]")
print("FEDB records      :", len(fedb))
print("Benchmark records :", len(benchmark))


# ============================================================
# FEDB KEYS
# ============================================================

fedb_keys = Counter()

for r in fedb:
    fedb_keys.update(r.keys())

print("\n[FEDB FIELD OCCURRENCE]")

for key, n in sorted(
    fedb_keys.items(),
    key=lambda x: (-x[1], x[0])
):
    print(
        f"{key:30s} "
        f"{n:5d}/{len(fedb)} "
        f"({100*n/len(fedb):6.2f}%)"
    )


# ============================================================
# FIVE-CONSTRAINT FEDB FIELDS
# ============================================================

fields = [
    "tree_type",
    "component",
    "variables",
    "location",
    "size_class",
]

print("\n[FEDB FIVE-CONSTRAINT FIELD COVERAGE]")

for field in fields:
    n = sum(present(r.get(field)) for r in fedb)
    print(
        f"{field:20s}: "
        f"{n:5d}/{len(fedb)} "
        f"({100*n/len(fedb):6.2f}%)"
    )


# ============================================================
# VARIABLES
# ============================================================

variable_forms = Counter()

for r in fedb:
    v = r.get("variables")

    if isinstance(v, list):
        key = "LIST:" + ",".join(map(str, v))
    elif isinstance(v, str):
        key = "STRING:" + v
    elif v is None:
        key = "<MISSING>"
    else:
        key = type(v).__name__ + ":" + str(v)

    variable_forms[key] += 1


print("\n[FEDB VARIABLES: TOP 30 FORMS]")

for value, n in variable_forms.most_common(30):
    print(f"{n:5d}  {value[:160]}")


# ============================================================
# SIZE CLASS
# ============================================================

size_present = 0
size_parseable = 0
size_examples_parseable = []
size_examples_unparseable = []

for r in fedb:
    sc = r.get("size_class")

    if not present(sc):
        continue

    size_present += 1
    parsed = parse_dbh_range(sc)

    if parsed is not None:
        size_parseable += 1
        if len(size_examples_parseable) < 20:
            size_examples_parseable.append((str(sc), parsed))
    else:
        if len(size_examples_unparseable) < 40:
            size_examples_unparseable.append(str(sc))


print("\n[FEDB SIZE_CLASS AUDIT]")
print("Present   :", size_present)

if size_present:
    print(
        "Parseable :",
        size_parseable,
        f"({100*size_parseable/size_present:.2f}% of present)"
    )

print("\nParseable examples:")
for raw, parsed in size_examples_parseable:
    print("  ", repr(raw), "=>", parsed)

print("\nUnparseable examples:")
for raw in size_examples_unparseable:
    print("  ", repr(raw))


# ============================================================
# BENCHMARK KEYS
# ============================================================

bench_keys = Counter()

for r in benchmark:
    bench_keys.update(r.keys())

print("\n[BENCHMARK FIELD OCCURRENCE]")

for key, n in sorted(
    bench_keys.items(),
    key=lambda x: (-x[1], x[0])
):
    print(
        f"{key:30s} "
        f"{n:4d}/{len(benchmark)} "
        f"({100*n/len(benchmark):6.2f}%)"
    )


# ============================================================
# BENCHMARK IMPORTANT FIELDS
# ============================================================

important = [
    "test_id",
    "question",
    "tree_type",
    "component",
    "location",
    "equation",
    "expected_result",
    "unit",
    "_calc_params",
    "gen_type",
    "group",
]

print("\n[BENCHMARK IMPORTANT FIELD COVERAGE]")

for field in important:
    n = sum(present(r.get(field)) for r in benchmark)
    print(
        f"{field:20s}: "
        f"{n:4d}/{len(benchmark)} "
        f"({100*n/len(benchmark):6.2f}%)"
    )


# ============================================================
# LABEL / NEGATIVE STRUCTURE
# ============================================================

print("\n[BENCHMARK POSSIBLE LABEL FIELDS]")

possible_label_fields = [
    "applicable",
    "is_applicable",
    "applicability",
    "label",
    "expected_behavior",
    "expected_result",
    "should_reject",
    "negative",
    "gen_type",
    "group",
    "source",
]

for field in possible_label_fields:
    if any(field in r for r in benchmark):

        values = Counter()

        for r in benchmark:
            value = r.get(field)

            if isinstance(value, (dict, list)):
                value = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                )

            values[str(value)] += 1

        print(f"\nFIELD = {field}")

        for value, n in values.most_common(30):
            print(f"  {n:4d}  {value[:180]}")


# ============================================================
# CALC PARAM STRUCTURE
# ============================================================

calc_keys = Counter()
calc_present = 0

for r in benchmark:
    cp = r.get("_calc_params")

    if isinstance(cp, dict):
        calc_present += 1
        calc_keys.update(cp.keys())


print("\n[BENCHMARK _calc_params]")
print("dict records:", calc_present)

for key, n in calc_keys.most_common():
    print(
        f"{key:20s}: "
        f"{n:4d}/{calc_present if calc_present else 1}"
    )


# ============================================================
# EQUATION FIELD TYPES
# ============================================================

equation_types = Counter(
    type(r.get("equation")).__name__
    for r in benchmark
)

print("\n[BENCHMARK equation FIELD TYPES]")
for k, n in equation_types.items():
    print(k, n)


# ============================================================
# FIRST 5 RECORD STRUCTURES
# ============================================================

print("\n[FIRST FIVE BENCHMARK RECORD SUMMARIES]")

for i, r in enumerate(benchmark[:5], 1):

    print(f"\n--- RECORD {i} ---")

    print("keys:", sorted(r.keys()))

    for k in [
        "test_id",
        "source",
        "group",
        "gen_type",
        "tree_type",
        "component",
        "location",
        "question",
        "equation",
        "expected_result",
        "_calc_params",
    ]:
        if k in r:
            value = repr(r.get(k))
            print(f"{k}: {value[:500]}")


# ============================================================
# UNIQUE IDS
# ============================================================

ids = [
    str(r.get("test_id"))
    for r in benchmark
    if present(r.get("test_id"))
]

print("\n[TEST ID CHECK]")
print("IDs present :", len(ids))
print("Unique IDs  :", len(set(ids)))
print("Duplicates  :", len(ids) - len(set(ids)))


# ============================================================
# FINAL ASSERTIONS
# ============================================================

print("\n[ASSERTIONS]")

if len(fedb) == 7394:
    print("PASS: FEDB count = 7394")
else:
    print("WARNING: FEDB count is not 7394")


if len(benchmark) == 522:
    print("PASS: benchmark count = 522")
else:
    print("WARNING: benchmark count is not 522")


print("\nAUDIT COMPLETE -- NO SOURCE DATA MODIFIED")
