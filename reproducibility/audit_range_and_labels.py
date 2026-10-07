import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import Counter, defaultdict


FEDB = Path(os.environ.get("FIQS_FEDB_PATH","data_public/fedb_public.json"))

BENCHMARK = Path(
    "<PRIVATE_ROOT>/homee/"
    "最新生物量测试集构建_v7_with_neg_原始_20260503_131441.json"
)


with FEDB.open("r", encoding="utf-8") as f:
    root = json.load(f)

fedb = root["equations"]


with BENCHMARK.open("r", encoding="utf-8") as f:
    root = json.load(f)

cases = root["test_cases"]


def nonempty(x):
    if x is None:
        return False
    if isinstance(x, str):
        return bool(x.strip())
    return True


# ============================================================
# 1. EXACT SIZE_CLASS DISTRIBUTION
# ============================================================

counter = Counter(
    str(r.get("size_class")).strip()
    for r in fedb
    if nonempty(r.get("size_class"))
)

print("=" * 80)
print("SIZE_CLASS + BENCHMARK LABEL AUDIT")
print("=" * 80)

print("\n===== SIZE_CLASS SUMMARY =====")
print("FEDB total:", len(fedb))
print("Non-empty size_class:", sum(counter.values()))
print("Unique size_class strings:", len(counter))


print("\n===== TOP 150 EXACT SIZE_CLASS VALUES =====")

for value, n in counter.most_common(150):
    print(f"{n:5d}\t{repr(value)}")


# ============================================================
# 2. CLASSIFY SIZE_CLASS FORMATS
# ============================================================

def classify(text):
    s = str(text).strip()

    if re.search(
        r"\d+(?:\.\d+)?\s*[-~～—–]\s*"
        r"\d+(?:\.\d+)?",
        s
    ):
        return "numeric_range"

    if re.search(
        r"(?:<=|>=|<|>|≤|≥)\s*\d|"
        r"\d\s*(?:<=|>=|<|>|≤|≥)",
        s
    ):
        return "inequality"

    if re.search(
        r"\d+(?:\.\d+)?\s*(?:cm|厘米|mm|毫米|m|米)",
        s,
        re.I
    ):
        return "single_numeric_with_unit"

    if re.search(r"\d", s):
        return "contains_number_other"

    return "text_only"


classes = Counter()

examples = defaultdict(list)

for value, n in counter.items():

    c = classify(value)

    classes[c] += n

    if len(examples[c]) < 60:
        examples[c].append((value, n))


print("\n===== SIZE_CLASS FORMAT COUNTS =====")

for c, n in classes.most_common():

    print(
        f"{c:28s}"
        f"{n:6d}/{sum(counter.values())} "
        f"({100*n/sum(counter.values()):6.2f}%)"
    )


for c in classes:

    print(f"\n===== EXAMPLES: {c} =====")

    for value, n in examples[c]:
        print(f"{n:5d}\t{repr(value)}")


# ============================================================
# 3. LOOK FOR RANGE INFORMATION OUTSIDE size_class
# ============================================================

possible_range_fields = Counter()

keywords = (
    "dbh",
    "diam",
    "diameter",
    "size",
    "range",
    "胸径",
    "径级",
    "直径",
    "age",
    "height",
)


for r in fedb:
    for k, v in r.items():

        kl = str(k).lower()

        if any(x in kl for x in keywords):
            possible_range_fields[k] += 1


print("\n===== POSSIBLE RANGE-RELATED FEDB FIELDS =====")

for k, n in possible_range_fields.most_common():
    print(f"{k:30s} {n:5d}")


# ============================================================
# 4. SHOW FULL RECORDS FOR REPRESENTATIVE SIZE_CLASS TYPES
# ============================================================

shown = set()

print("\n===== REPRESENTATIVE FEDB RECORDS =====")

for r in fedb:

    sc = r.get("size_class")

    if not nonempty(sc):
        continue

    c = classify(sc)

    if c in shown:
        continue

    shown.add(c)

    print(f"\n--- TYPE: {c} ---")

    for key in [
        "id",
        "tree_type",
        "component",
        "equation",
        "parameters",
        "variables",
        "location",
        "size_class",
        "source",
    ]:
        print(f"{key}: {repr(r.get(key))[:1000]}")

    if len(shown) == len(classes):
        break


# ============================================================
# 5. BENCHMARK LABEL STRUCTURE
# ============================================================

print("\n\n" + "=" * 80)
print("BENCHMARK LABEL STRUCTURE")
print("=" * 80)


for field in [
    "group",
    "gen_type",
    "source",
    "expected_result",
]:

    values = Counter()

    for r in cases:

        value = r.get(field)

        if isinstance(value, (dict, list)):
            value = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )

        values[str(value)] += 1

    print(f"\n===== FIELD: {field} =====")

    for value, n in values.most_common(100):
        print(f"{n:4d}\t{value[:500]}")


# ============================================================
# 6. MISSING TREE/COMPONENT/LOCATION CASES
# ============================================================

missing_core = []

for r in cases:

    missing = [
        k
        for k in ["tree_type", "component", "location"]
        if not nonempty(r.get(k))
    ]

    if missing:
        missing_core.append((r, missing))


print("\n===== CASES WITH MISSING CORE FIELDS =====")
print("Count:", len(missing_core))


for r, missing in missing_core:

    print("\n---")

    print("test_id:", r.get("test_id"))
    print("missing:", missing)
    print("group:", repr(r.get("group")))
    print("gen_type:", repr(r.get("gen_type")))
    print("source:", repr(r.get("source")))
    print("question:", repr(r.get("question"))[:800])
    print("expected_result:", repr(r.get("expected_result"))[:800])
    print("_calc_params:", repr(r.get("_calc_params"))[:800])


# ============================================================
# 7. CROSS-TAB GROUP x GEN_TYPE
# ============================================================

cross = Counter(
    (
        str(r.get("group")),
        str(r.get("gen_type")),
    )
    for r in cases
)

print("\n===== GROUP x GEN_TYPE =====")

for (group, gen_type), n in sorted(
    cross.items(),
    key=lambda x: (-x[1], x[0])
):
    print(
        f"{n:4d}\t"
        f"group={group!r}\t"
        f"gen_type={gen_type!r}"
    )


# ============================================================
# 8. NEGATIVE-LIKE TEXT SEARCH
# ============================================================

negative_words = [
    "拒绝",
    "不适用",
    "无法",
    "不能",
    "缺少",
    "不足",
    "超出",
    "无适用",
    "无匹配",
    "reject",
    "inapplicable",
]


negative_like = []

for r in cases:

    blob = " ".join(
        str(r.get(k, ""))
        for k in [
            "group",
            "gen_type",
            "source",
            "question",
            "expected_result",
        ]
    ).lower()

    if any(w.lower() in blob for w in negative_words):
        negative_like.append(r)


print("\n===== NEGATIVE-LIKE TEXT COUNT =====")
print(len(negative_like))


print("\n===== FIRST 40 NEGATIVE-LIKE CASES =====")

for r in negative_like[:40]:

    print(
        r.get("test_id"),
        "| group=", repr(r.get("group")),
        "| gen_type=", repr(r.get("gen_type")),
        "| expected=", repr(r.get("expected_result"))[:250],
    )


print("\nAUDIT COMPLETE -- READ ONLY")
