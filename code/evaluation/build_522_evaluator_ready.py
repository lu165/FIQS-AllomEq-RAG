import os
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

BENCH = (
    Path(
        os.environ.get(
            "FIQS_DATA_ROOT",
            "./data_public/"
        )
    )
    / "最新生物量测试集构建_v7_with_neg_原始_20260503_131441.json"
)

REF = (
    ROOT / "reproducibility/"
    "applicable_reference_manifest.jsonl"
)

OUT = (
    ROOT / "reproducibility/"
    "benchmark_522_evaluator_ready.jsonl"
)

# ------------------------------------------------------------
# Load benchmark
# ------------------------------------------------------------

obj = json.loads(
    BENCH.read_text(encoding="utf-8")
)

cases = obj["test_cases"]

assert len(cases) == 522

# ------------------------------------------------------------
# Load 434 applicable reference sets
# ------------------------------------------------------------

ref_rows = [
    json.loads(line)
    for line in REF.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

assert len(ref_rows) == 434

ref_map = {
    row["test_id"]: row
    for row in ref_rows
}

assert len(ref_map) == 434

# ------------------------------------------------------------
# Build evaluator-ready 522 records
# ------------------------------------------------------------

rows = []
counts = Counter()

for i, case in enumerate(cases):

    tid = (
        case.get("test_id")
        or case.get("id")
        or f"INDEX_{i}"
    )

    is_inapplicable = (
        case.get("requires_rejection") is True
    )

    if is_inapplicable:

        # Formal negative:
        # no equation reference set is required.
        row = {
            "test_id": tid,
            "formal_label": "INAPPLICABLE",
            "requires_rejection": True,
            "acceptable_equations": [],
            "reference_primary": None,
            "reference_provenance":
                "formal_rejection_label",
            "question": case.get("question"),
            "rejection_reason":
                case.get("rejection_reason"),
        }

        counts["INAPPLICABLE"] += 1

    else:

        if tid not in ref_map:
            raise RuntimeError(
                f"Applicable case missing "
                f"reference set: {tid}"
            )

        ref = ref_map[tid]

        acceptable = ref["reference_set"]

        if not acceptable:
            raise RuntimeError(
                f"Empty reference set: {tid}"
            )

        row = {
            "test_id": tid,
            "formal_label": "APPLICABLE",
            "requires_rejection": False,
            "acceptable_equations":
                acceptable,
            "reference_primary":
                ref["reference_primary"],
            "reference_provenance":
                ref["reference_provenance"],
            "question": case.get("question"),
            "rejection_reason": None,
        }

        counts["APPLICABLE"] += 1

    rows.append(row)

# ------------------------------------------------------------
# Global validation
# ------------------------------------------------------------

ids = [
    row["test_id"]
    for row in rows
]

assert len(rows) == 522
assert len(set(ids)) == 522

assert counts["APPLICABLE"] == 434
assert counts["INAPPLICABLE"] == 88

assert sum(
    bool(row["acceptable_equations"])
    for row in rows
    if row["formal_label"] == "APPLICABLE"
) == 434

assert sum(
    bool(row["acceptable_equations"])
    for row in rows
    if row["formal_label"] == "INAPPLICABLE"
) == 0

# Every applicable benchmark ID must occur
# exactly once in the 434 reference manifest.
used_refs = {
    row["test_id"]
    for row in rows
    if row["formal_label"] == "APPLICABLE"
}

assert used_refs == set(ref_map)

# ------------------------------------------------------------
# Write
# ------------------------------------------------------------

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT.open(
    "w",
    encoding="utf-8",
) as f:

    for row in rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

print("=" * 70)
print("522 EVALUATOR-READY MANIFEST")
print("=" * 70)

print("TOTAL =", len(rows))
print(
    "APPLICABLE =",
    counts["APPLICABLE"]
)
print(
    "INAPPLICABLE =",
    counts["INAPPLICABLE"]
)

print(
    "APPLICABLE_WITH_REFERENCE =",
    sum(
        bool(x["acceptable_equations"])
        for x in rows
        if x["formal_label"]
        == "APPLICABLE"
    )
)

print(
    "INAPPLICABLE_WITH_REFERENCE =",
    sum(
        bool(x["acceptable_equations"])
        for x in rows
        if x["formal_label"]
        == "INAPPLICABLE"
    )
)

print(
    "LEGACY_REFERENCE_RECOVERY =",
    sum(
        x["reference_provenance"]
        == "legacy_reference_recovery"
        for x in rows
    )
)

print(
    "HISTORICAL_PROGRAMMATIC_GOLD =",
    sum(
        x["reference_provenance"]
        == "historical_programmatic_gold"
        for x in rows
    )
)

print(
    "FORMAL_REJECTION_LABEL =",
    sum(
        x["reference_provenance"]
        == "formal_rejection_label"
        for x in rows
    )
)

print("UNIQUE_IDS =", len(set(ids)))
print("OUTPUT =", OUT)

print()
print("EVALUATOR_READY_522_PASS")
