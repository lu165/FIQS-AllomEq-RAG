import os
import json
import importlib.util
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]

BENCH = (
    Path(
        os.environ.get(
            "FIQS_DATA_ROOT",
            "./data_public/"
        )
    )
    / "最新生物量测试集构建_v7_with_neg_原始_20260503_131441.json"
)

FEDB = Path(
    Path(os.environ.get("FIQS_FEDB_PATH","./data_public/林业方程_标准化.json"))
)

HIST = (
    ROOT / "code/evaluation/"
    "historical_v24_evaluator.py"
)

OUT = (
    ROOT / "reproducibility/"
    "applicable_reference_manifest.jsonl"
)

spec = importlib.util.spec_from_file_location(
    "historical_v24_evaluator",
    HIST,
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

fedb = json.loads(
    FEDB.read_text(encoding="utf-8")
)
mod.db = fedb["equations"]

cases = json.loads(
    BENCH.read_text(encoding="utf-8")
)["test_cases"]

assert len(cases) == 522
assert len(mod.db) == 7394

rows = []
sources = Counter()

for i, case in enumerate(cases):

    # Formal applicability is defined ONLY by this label.
    if case.get("requires_rejection") is True:
        continue

    tid = (
        case.get("test_id")
        or case.get("id")
        or f"INDEX_{i}"
    )

    gold, primary = mod.build_gold_set(case)

    references = sorted(
        x for x in gold if x
    )

    provenance = (
        "historical_programmatic_gold"
    )

    recovered_legacy = False

    # Historical build_gold_set returns empty when the
    # current top-level equation field is blank.
    if not references:

        legacy = str(
            case.get("_legacy_equation") or ""
        ).strip()

        if legacy:
            normalized = mod.normalize_eq(legacy)

            if normalized:
                references = [normalized]
                primary = normalized
                provenance = (
                    "legacy_reference_recovery"
                )
                recovered_legacy = True

    if not references:
        raise RuntimeError(
            f"No reference equation for {tid}"
        )

    row = {
        "test_id": tid,
        "formal_applicable": True,
        "reference_primary": primary,
        "reference_set": references,
        "reference_set_size": len(references),
        "reference_provenance": provenance,
        "legacy_recovered": recovered_legacy,
        "benchmark_equation": case.get(
            "equation"
        ),
        "legacy_equation": case.get(
            "_legacy_equation"
        ),
        "tree_type": case.get("tree_type"),
        "location": case.get("location"),
        "component": case.get("component"),
        "source": case.get("source"),
    }

    rows.append(row)
    sources[provenance] += 1

assert len(rows) == 434

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

print("OUTPUT =", OUT)
print("ROWS =", len(rows))
print("PROVENANCE =", dict(sources))

print(
    "EMPTY_REFERENCE_SET =",
    sum(
        not x["reference_set"]
        for x in rows
    )
)

print(
    "LEGACY_RECOVERED =",
    sum(
        x["legacy_recovered"]
        for x in rows
    )
)

print("LEGACY_IDS =")
for x in rows:
    if x["legacy_recovered"]:
        print(
            x["test_id"],
            "=>",
            x["reference_primary"],
        )

print("BUILD_REFERENCE_MANIFEST_PASS")
