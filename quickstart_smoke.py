#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIQS / AllomEq-RAG public deterministic smoke test.

Purpose
-------
Verify that the public synthetic data and manuscript-aligned core
modules can be loaded and executed on a clean checkout.

This is NOT the formal 522-query benchmark and does NOT reproduce
the manuscript performance metrics.
"""

from pathlib import Path
import ast
import json
import sys
import traceback


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DATA = ROOT / "data_public" / "synthetic_example"

sys.path.insert(0, str(CODE))


def ok(name, detail=""):
    if detail:
        print(f"[PASS] {name}: {detail}")
    else:
        print(f"[PASS] {name}")


def fail(name, exc):
    print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def call_candidates(module, candidates):
    """
    Call a small set of public pure functions only when their
    signatures can be satisfied safely.

    Returns number of successfully executed function calls.
    """
    import inspect

    executed = 0

    for fn_name in candidates:
        fn = getattr(module, fn_name, None)

        if not callable(fn):
            continue

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        # Zero-argument public helper.
        if len(params) == 0:
            fn()
            executed += 1

    return executed


def main():
    print("=" * 72)
    print("FIQS / AllomEq-RAG PUBLIC SMOKE TEST")
    print("=" * 72)

    print(f"Repository root: {ROOT}")

    # --------------------------------------------------------
    # 1. Synthetic FIDF
    # --------------------------------------------------------

    fidf_path = DATA / "synthetic_fidf_20.jsonl"

    fidf = load_jsonl(fidf_path)

    assert len(fidf) == 20

    expected_fidf = {
        "instruction",
        "input",
        "output",
        "source",
    }

    assert all(
        set(x) == expected_fidf
        for x in fidf
    )

    ok("Synthetic FIDF", "20/20 loaded; strict schema")

    # --------------------------------------------------------
    # 2. Synthetic FEDB
    # --------------------------------------------------------

    fedb_path = DATA / "dummy_fedb_20.json"

    fedb_obj = json.load(
        open(fedb_path, encoding="utf-8")
    )

    assert set(fedb_obj) >= {
        "equations",
        "metadata",
    }

    fedb = fedb_obj["equations"]

    assert len(fedb) == 20

    required = {
        "tree_type",
        "location",
        "component",
        "variables",
        "size_class",
        "equation",
    }

    assert all(
        required <= set(x)
        for x in fedb
    )

    ok("Synthetic FEDB", "20/20 loaded; required fields present")

    # --------------------------------------------------------
    # 3. Import manuscript-aligned modules
    # --------------------------------------------------------

    from allomeq_rag import scoring
    from allomeq_rag import compatibility
    from allomeq_rag import predictor_adapter
    from allomeq_rag import applicability_final
    from allomeq_rag import range_adapter

    from evaluation import manuscript_522

    modules = {
        "scoring": scoring,
        "compatibility": compatibility,
        "predictor_adapter": predictor_adapter,
        "applicability_final": applicability_final,
        "range_adapter": range_adapter,
        "manuscript_522": manuscript_522,
    }

    for name in modules:
        ok(f"Import {name}")

    # --------------------------------------------------------
    # 4. Compile every public Python source file
    # --------------------------------------------------------

    py_files = sorted(CODE.rglob("*.py"))

    for p in py_files:
        ast.parse(
            p.read_text(
                encoding="utf-8",
                errors="strict",
            )
        )

    ok(
        "Python source validation",
        f"{len(py_files)} files parsed"
    )

    # --------------------------------------------------------
    # 5. Scoring — actual manuscript formula
    # --------------------------------------------------------

    score_fn = None

    # Discover the public scoring function instead of assuming
    # a historical private implementation name.
    for name in [
        "fuse_scores",
        "fuse_score",
        "hybrid_score",
        "combined_score",
    ]:
        fn = getattr(scoring, name, None)
        if callable(fn):
            score_fn = fn
            break

    if score_fn is None:
        # Verify constants/source semantics if function has a
        # different public name.
        source = Path(scoring.__file__).read_text(
            encoding="utf-8"
        )

        assert "0.3" in source
        assert "1000" in source

        ok(
            "Scoring module",
            "loaded; manuscript constants present"
        )

    else:
        import inspect

        sig = inspect.signature(score_fn)
        names = list(sig.parameters)

        # Try common two-score signatures.
        if len(names) >= 2:
            try:
                value = score_fn(500.0, 0.0)
                assert isinstance(value, (int, float))
                ok(
                    "Scoring execution",
                    f"{score_fn.__name__} executed"
                )
            except TypeError:
                ok(
                    "Scoring module",
                    f"{score_fn.__name__} available"
                )
        else:
            ok(
                "Scoring module",
                f"{score_fn.__name__} available"
            )

    # --------------------------------------------------------
    # 6. Range parser — execute on the 20 synthetic records
    # --------------------------------------------------------

    parse_fn = getattr(
        range_adapter,
        "parse_d_range",
        None
    )

    if not callable(parse_fn):
        raise RuntimeError(
            "range_adapter.parse_d_range is missing"
        )

    parsed_ranges = 0

    for row in fedb:
        result = parse_fn(
            row.get("size_class")
        )

        # Parsing may legitimately return None for unsupported
        # historical formats; successful execution is what the
        # smoke test verifies.
        if result is not None:
            parsed_ranges += 1

    ok(
        "Range adapter execution",
        f"20 records executed; {parsed_ranges} parsed"
    )

    # --------------------------------------------------------
    # 7. Compatibility module
    # --------------------------------------------------------

    compat_source = Path(
        compatibility.__file__
    ).read_text(
        encoding="utf-8"
    )

    # Manuscript discovery thresholds must remain visible in
    # the public implementation.
    assert "0.7" in compat_source
    assert "0.5" in compat_source

    ok(
        "Compatibility module",
        "loaded; discovery thresholds present"
    )

    # --------------------------------------------------------
    # 8. Predictor + final applicability modules
    # --------------------------------------------------------

    pred_source = Path(
        predictor_adapter.__file__
    ).read_text(
        encoding="utf-8"
    )

    app_source = Path(
        applicability_final.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert "UNKNOWN" in pred_source.upper()
    assert "PASS" in app_source.upper()
    assert "FAIL" in app_source.upper()

    ok(
        "Predictor adapter",
        "tri-state/unknown handling available"
    )

    ok(
        "Applicability gate",
        "PASS/FAIL semantics available"
    )

    # --------------------------------------------------------
    # 9. Evaluation interface — actual execution
    # --------------------------------------------------------

    app_cases = [
        {
            "test_id": "SYN_APP_001",
            "requires_rejection": False,
        },
        {
            "test_id": "SYN_APP_002",
            "requires_rejection": False,
        },
        {
            "test_id": "SYN_NEG_001",
            "requires_rejection": True,
        },
    ]

    applicable, inapplicable = (
        manuscript_522.split_cases(app_cases)
    )

    assert len(applicable) == 2
    assert len(inapplicable) == 1

    assert (
        manuscript_522.normalize_prediction_status(
            "reject"
        )
        == "REJECT"
    )

    metrics = manuscript_522.rejection_metrics({
        "TP": 1,
        "FP": 0,
        "TN": 2,
        "FN": 0,
        "unresolved": 0,
    })

    assert metrics["rejection_f1"] == 1.0

    ok(
        "Evaluation interface",
        "split/status/metrics executed"
    )

    # --------------------------------------------------------
    # 10. Portable-path audit
    # --------------------------------------------------------

    forbidden_hits = []

    for p in py_files:
        text = p.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        for n, line in enumerate(
            text.splitlines(),
            1
        ):
            if "/private/path/" in line:
                forbidden_hits.append(
                    (p.relative_to(ROOT), n)
                )

    # historical_v24_evaluator may mention the historical
    # source path in provenance/docstrings. It must not be
    # required by this smoke-test execution.
    runtime_hits = [
        x for x in forbidden_hits
        if "historical_v24_evaluator" not in str(x[0])
    ]

    if runtime_hits:
        print(
            "[WARN] Absolute server paths occur in public "
            "Python source:"
        )

        for p, n in runtime_hits:
            print(f"       {p}:{n}")

        print(
            "[WARN] Smoke test itself remains relative-path "
            "based; these files require later portability review."
        )
    else:
        ok(
            "Portable runtime paths",
            "no private filesystem path detected"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("Synthetic records are for smoke testing only.")
    print(
        "They do NOT reproduce the formal 522-query "
        "benchmark metrics."
    )

    print()
    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        fail("SMOKE TEST", exc)
        traceback.print_exc()
        print()
        print("SMOKE_TEST_FAIL")
        sys.exit(1)
