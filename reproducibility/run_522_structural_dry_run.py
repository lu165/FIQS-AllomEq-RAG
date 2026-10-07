"""
522-case structural dry-run for the reconstructed AllomEq-RAG gate.

IMPORTANT
---------
This script is diagnostic only.

It does NOT:
- call an LLM,
- call DeepSeek,
- call an external API,
- perform final retrieval evaluation,
- compute manuscript Safety/Coverage/F1,
- claim reproduction of historical reported results.

Its purpose is to quantify how much of the frozen 522-case benchmark
can be deterministically evaluated using currently available structured
evidence.

Unknown evidence remains UNKNOWN / UNRESOLVED.
"""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import sys


ROOT = Path("<PRIVATE_ROOT>/小论文修改实验")
HOME = Path("<PRIVATE_ROOT>/homee")

FEDB_PATH = HOME / "林业方程_标准化.json"

BENCH_PATH = (
    HOME /
    "最新生物量测试集构建_v7_with_neg_原始_20260503_131441.json"
)

OUT_DIR = (
    ROOT /
    "github_release/results/structural_dry_run"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

sys.path.insert(
    0,
    str(ROOT / "github_release/code"),
)

from allomeq_rag.applicability_final import (
    GateStatus,
    component_constraint,
    predictor_constraint,
    measurement_range_constraint,
)


# ============================================================
# Helpers
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def nonempty(value):
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(
        value,
        (list, tuple, dict, set),
    ):
        return bool(value)

    return True


def norm_text(value):
    if value is None:
        return ""

    return str(value).strip()


def extract_dbh(calc_params):
    """
    Conservative DBH extraction.

    Only explicit structured DBH-like keys are accepted here.
    No LLM inference and no free-text guessing.
    """

    if not isinstance(calc_params, dict):
        return None, None

    preferred = (
        "D",
        "DBH",
        "D_{1.3}",
        "D_1.3",
    )

    for key in preferred:
        if key not in calc_params:
            continue

        value = calc_params.get(key)

        try:
            return float(value), key
        except (TypeError, ValueError):
            pass

    return None, None


def build_exact_index(records, field):
    index = {}

    for rec in records:
        value = norm_text(
            rec.get(field)
        )

        if not value:
            continue

        index.setdefault(
            value,
            [],
        ).append(rec)

    return index


# ============================================================
# Load immutable source data
# ============================================================

with open(
    FEDB_PATH,
    encoding="utf-8",
) as f:
    fedb_root = json.load(f)

fedb = fedb_root["equations"]

with open(
    BENCH_PATH,
    encoding="utf-8",
) as f:
    bench_root = json.load(f)

cases = bench_root["test_cases"]


assert len(fedb) == 7394
assert len(cases) == 522


# ============================================================
# Exact lookup indexes
#
# These are NOT semantic retrieval.
# They are only used to identify whether a benchmark record
# has a uniquely/evidently corresponding structured candidate.
# ============================================================

equation_index = build_exact_index(
    fedb,
    "equation",
)


# ============================================================
# Counters
# ============================================================

label_counts = Counter()

candidate_match_counts = Counter()

component_counts = Counter()
predictor_counts = Counter()
range_counts = Counter()

diagnostic_gate_counts = Counter()

unresolved_reason_counts = Counter()

rows = []


# ============================================================
# Process 522 cases
# ============================================================

for i, case in enumerate(
    cases,
    start=1,
):
    test_id = (
        case.get("test_id")
        or f"row_{i:04d}"
    )

    requires_rejection = bool(
        case.get(
            "requires_rejection",
            False,
        )
    )

    label = (
        "inapplicable"
        if requires_rejection
        else "applicable"
    )

    label_counts[label] += 1

    benchmark_equation = norm_text(
        case.get("equation")
    )

    exact_candidates = (
        equation_index.get(
            benchmark_equation,
            [],
        )
        if benchmark_equation
        else []
    )

    if not benchmark_equation:
        match_status = (
            "benchmark_equation_missing"
        )

    elif len(exact_candidates) == 0:
        match_status = (
            "no_exact_fedb_equation_match"
        )

    elif len(exact_candidates) == 1:
        match_status = (
            "unique_exact_fedb_equation_match"
        )

    else:
        match_status = (
            "multiple_exact_fedb_equation_matches"
        )

    candidate_match_counts[
        match_status
    ] += 1

    # --------------------------------------------------------
    # Only evaluate candidate-dependent constraints when an
    # exact FEDB equation candidate exists.
    #
    # If several records share the exact equation string, use
    # only candidates whose component also exactly equals the
    # benchmark component. If ambiguity remains, do not guess.
    # --------------------------------------------------------

    candidate = None
    candidate_resolution = ""

    if len(exact_candidates) == 1:
        candidate = exact_candidates[0]
        candidate_resolution = (
            "unique_equation"
        )

    elif len(exact_candidates) > 1:
        qcomp = norm_text(
            case.get("component")
        )

        comp_matches = [
            rec
            for rec in exact_candidates
            if norm_text(
                rec.get("component")
            ) == qcomp
        ]

        if len(comp_matches) == 1:
            candidate = comp_matches[0]
            candidate_resolution = (
                "equation_plus_component"
            )
        else:
            candidate_resolution = (
                "ambiguous_exact_equation"
            )

    else:
        candidate_resolution = (
            "no_candidate"
        )

    calc_params = case.get(
        "_calc_params"
    )

    dbh, dbh_source = extract_dbh(
        calc_params
    )

    # --------------------------------------------------------
    # Candidate-independent metadata presence
    # --------------------------------------------------------

    query_tree = norm_text(
        case.get("tree_type")
    )

    query_location = norm_text(
        case.get("location")
    )

    query_component = norm_text(
        case.get("component")
    )

    # --------------------------------------------------------
    # No resolved candidate -> candidate-dependent checks
    # remain unresolved.
    # --------------------------------------------------------

    if candidate is None:
        component_status = (
            GateStatus.UNRESOLVED
        )

        predictor_status = (
            GateStatus.UNRESOLVED
        )

        range_status = (
            GateStatus.UNRESOLVED
        )

        component_reason = (
            "No unambiguous exact FEDB "
            "candidate was established."
        )

        predictor_reason = (
            component_reason
        )

        range_reason = (
            component_reason
        )

    else:
        c = component_constraint(
            query_component,
            candidate.get(
                "component"
            ),
        )

        p = predictor_constraint(
            candidate.get(
                "variables"
            ),
            calc_params=calc_params,
            question=case.get(
                "question"
            ),
        )

        r = measurement_range_constraint(
            candidate.get(
                "size_class"
            ),
            dbh,
        )

        component_status = c.status
        predictor_status = p.status
        range_status = r.status

        component_reason = c.reason
        predictor_reason = p.reason
        range_reason = r.reason

    component_counts[
        component_status.value
    ] += 1

    predictor_counts[
        predictor_status.value
    ] += 1

    range_counts[
        range_status.value
    ] += 1

    # --------------------------------------------------------
    # Species/geography FINAL compatibility:
    #
    # Intentionally UNKNOWN here.
    #
    # Exact strings are recorded below, but are NOT silently
    # converted into final gate evidence because the final
    # compatibility adapter has not yet been integrated.
    # --------------------------------------------------------

    species_status = (
        GateStatus.UNRESOLVED
    )

    geography_status = (
        GateStatus.UNRESOLVED
    )

    # --------------------------------------------------------
    # Diagnostic overall state
    #
    # This follows the tri-state aggregation policy but is NOT
    # the formal manuscript applicability result.
    # --------------------------------------------------------

    statuses = [
        species_status,
        geography_status,
        component_status,
        predictor_status,
        range_status,
    ]

    if GateStatus.FAIL in statuses:
        diagnostic_status = (
            GateStatus.FAIL
        )

    elif all(
        s == GateStatus.PASS
        for s in statuses
    ):
        diagnostic_status = (
            GateStatus.PASS
        )

    else:
        diagnostic_status = (
            GateStatus.UNRESOLVED
        )

    diagnostic_gate_counts[
        diagnostic_status.value
    ] += 1

    if species_status == GateStatus.UNRESOLVED:
        unresolved_reason_counts[
            "species_final_evidence"
        ] += 1

    if geography_status == GateStatus.UNRESOLVED:
        unresolved_reason_counts[
            "geography_final_evidence"
        ] += 1

    if component_status == GateStatus.UNRESOLVED:
        unresolved_reason_counts[
            "component"
        ] += 1

    if predictor_status == GateStatus.UNRESOLVED:
        unresolved_reason_counts[
            "predictors"
        ] += 1

    if range_status == GateStatus.UNRESOLVED:
        unresolved_reason_counts[
            "measurement_range"
        ] += 1

    candidate_tree = (
        norm_text(
            candidate.get(
                "tree_type"
            )
        )
        if candidate
        else ""
    )

    candidate_location = (
        norm_text(
            candidate.get(
                "location"
            )
        )
        if candidate
        else ""
    )

    rows.append(
        {
            "test_id": test_id,
            "requires_rejection": (
                requires_rejection
            ),
            "benchmark_label": label,

            "query_tree_type": (
                query_tree
            ),
            "query_location": (
                query_location
            ),
            "query_component": (
                query_component
            ),

            "benchmark_equation_present": (
                bool(
                    benchmark_equation
                )
            ),

            "exact_equation_match_count": (
                len(
                    exact_candidates
                )
            ),

            "candidate_match_status": (
                match_status
            ),

            "candidate_resolution": (
                candidate_resolution
            ),

            "candidate_tree_type": (
                candidate_tree
            ),

            "candidate_location": (
                candidate_location
            ),

            "tree_string_exact": (
                bool(
                    candidate
                    and query_tree
                    and candidate_tree
                    and query_tree
                    == candidate_tree
                )
            ),

            "location_string_exact": (
                bool(
                    candidate
                    and query_location
                    and candidate_location
                    and query_location
                    == candidate_location
                )
            ),

            "dbh": dbh,
            "dbh_source": (
                dbh_source or ""
            ),

            "species_status": (
                species_status.value
            ),

            "geography_status": (
                geography_status.value
            ),

            "component_status": (
                component_status.value
            ),

            "predictor_status": (
                predictor_status.value
            ),

            "measurement_range_status": (
                range_status.value
            ),

            "diagnostic_gate_status": (
                diagnostic_status.value
            ),

            "component_reason": (
                component_reason
            ),

            "predictor_reason": (
                predictor_reason
            ),

            "measurement_range_reason": (
                range_reason
            ),
        }
    )


# ============================================================
# Validate frozen benchmark composition
# ============================================================

assert (
    label_counts["applicable"]
    == 434
)

assert (
    label_counts["inapplicable"]
    == 88
)


# ============================================================
# Save case-level CSV
# ============================================================

csv_path = (
    OUT_DIR /
    "structural_dry_run_522.csv"
)

with open(
    csv_path,
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Save JSON summary
# ============================================================

summary = {
    "status": (
        "diagnostic_only_not_manuscript_result"
    ),

    "fedb_records": len(fedb),
    "benchmark_cases": len(cases),

    "benchmark_labels": dict(
        label_counts
    ),

    "candidate_match_status": dict(
        candidate_match_counts
    ),

    "component_status": dict(
        component_counts
    ),

    "predictor_status": dict(
        predictor_counts
    ),

    "measurement_range_status": dict(
        range_counts
    ),

    "diagnostic_gate_status": dict(
        diagnostic_gate_counts
    ),

    "unresolved_constraints": dict(
        unresolved_reason_counts
    ),

    "source_sha256": {
        "fedb": sha256(
            FEDB_PATH
        ),
        "benchmark": sha256(
            BENCH_PATH
        ),
    },

    "notes": [
        (
            "No LLM/API calls were made."
        ),
        (
            "No manuscript Safety/Coverage/F1 "
            "metrics were computed."
        ),
        (
            "Species and geography final "
            "compatibility were intentionally "
            "left UNRESOLVED."
        ),
        (
            "Generalized discovery similarity "
            "was not treated as final "
            "applicability evidence."
        ),
        (
            "Exact benchmark equation strings "
            "were used only for diagnostic "
            "candidate linkage."
        ),
    ],
}

json_path = (
    OUT_DIR /
    "structural_dry_run_522_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Terminal summary
# ============================================================

print(
    "===== 522 STRUCTURAL DRY-RUN ====="
)

print(
    "FEDB_TOTAL =",
    len(fedb),
)

print(
    "BENCHMARK_TOTAL =",
    len(cases),
)

print(
    "APPLICABLE =",
    label_counts["applicable"],
)

print(
    "INAPPLICABLE =",
    label_counts["inapplicable"],
)

print()

print(
    "===== CANDIDATE LINKAGE ====="
)

for k, v in sorted(
    candidate_match_counts.items()
):
    print(
        f"{k} = {v}"
    )

print()

print(
    "===== COMPONENT ====="
)

for k, v in sorted(
    component_counts.items()
):
    print(
        f"{k} = {v}"
    )

print()

print(
    "===== PREDICTORS ====="
)

for k, v in sorted(
    predictor_counts.items()
):
    print(
        f"{k} = {v}"
    )

print()

print(
    "===== MEASUREMENT RANGE ====="
)

for k, v in sorted(
    range_counts.items()
):
    print(
        f"{k} = {v}"
    )

print()

print(
    "===== DIAGNOSTIC FIVE-CONSTRAINT STATE ====="
)

for k, v in sorted(
    diagnostic_gate_counts.items()
):
    print(
        f"{k} = {v}"
    )

print()

print(
    "===== UNRESOLVED COUNTS ====="
)

for k, v in sorted(
    unresolved_reason_counts.items(),
    key=lambda x: (
        -x[1],
        x[0],
    ),
):
    print(
        f"{k} = {v}"
    )

print()

print(
    "CSV =",
    csv_path,
)

print(
    "SUMMARY =",
    json_path,
)

print(
    "PASS: diagnostic structural dry-run completed"
)
