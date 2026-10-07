import json
import importlib.util
from pathlib import Path

from allomeq_rag.structural_score_historical import (
    HistoricalConfig,
    calculate_score_historical,
)

SRC = Path("<PRIVATE_ROOT>/homee/ablation_framework_logic_closed_loop.py")
FEDB = Path("<PRIVATE_ROOT>/homee/林业方程_标准化.json")


spec = importlib.util.spec_from_file_location("historical_ablation", SRC)
hist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hist)


with open(FEDB, "r", encoding="utf-8") as f:
    obj = json.load(f)

records = obj["equations"]


queries = [
    {
        "tree_type": "樟子松",
        "location": "黑龙江省",
        "D": 15.0,
        "H": 12.0,
        "component": "地上",
    },
    {
        "tree_type": "杉木",
        "location": "福建省",
        "D": 25.0,
        "H": None,
        "component": "干",
    },
    {
        "tree_type": "",
        "location": "",
        "D": None,
        "H": None,
        "component": "",
    },
]


hist_cfg = hist.ExperimentConfig(
    "equivalence",
    use_species_constraint=True,
    use_location_constraint=True,
    use_diameter_constraint=True,
    use_component_constraint=True,
    alpha=0.3,
)

new_cfg = HistoricalConfig(
    use_species_constraint=True,
    use_location_constraint=True,
    use_diameter_constraint=True,
    use_component_constraint=True,
    alpha=0.3,
)


tested = 0
mismatches = []

for q_idx, query in enumerate(queries):
    for r_idx, eq in enumerate(records):

        a = hist.calculate_score_with_config(eq, query, hist_cfg)
        b = calculate_score_historical(eq, query, new_cfg)

        tested += 1

        if a != b:
            mismatches.append(
                {
                    "query_index": q_idx,
                    "record_index": r_idx,
                    "historical": a,
                    "extracted": b,
                    "equation_id": eq.get("id"),
                }
            )

            if len(mismatches) >= 20:
                break

    if mismatches:
        break


print("FEDB_RECORDS =", len(records))
print("QUERIES =", len(queries))
print("COMPARISONS =", tested)
print("MISMATCHES =", len(mismatches))

for x in mismatches[:10]:
    print(x)

assert len(records) == 7394
assert not mismatches, mismatches[:5]

print("STRUCTURAL_HISTORICAL_EQUIVALENCE_PASS")
