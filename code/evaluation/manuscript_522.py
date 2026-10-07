# -*- coding: utf-8 -*-
"""
Manuscript-aligned evaluator for the 522-query equation benchmark.

Benchmark semantics
-------------------
Total: 522
Applicable: 434
Inapplicable: 88

Ground-truth applicability is defined by the frozen benchmark field:

    requires_rejection == True  -> inapplicable
    otherwise                   -> applicable

IMPORTANT
---------
This module is distinct from historical_v24_evaluator.py.

Historical v24 bucket labels such as `is_neg`, `bucket_main`,
`bucket_rejection`, etc. are NOT used as the formal manuscript
applicability ground truth.

This evaluator does not fabricate or reconstruct model predictions.
It evaluates externally supplied system outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class BenchmarkLabel:
    test_id: str
    requires_rejection: bool

    @property
    def applicable(self) -> bool:
        return not self.requires_rejection


def formal_label(case: Dict[str, Any]) -> BenchmarkLabel:
    """
    Return the frozen formal applicability label.
    """
    return BenchmarkLabel(
        test_id=str(case.get("test_id", "")),
        requires_rejection=(case.get("requires_rejection") is True),
    )


def split_cases(
    cases: Iterable[Dict[str, Any]]
):
    """
    Split benchmark using ONLY requires_rejection.
    """
    applicable = []
    inapplicable = []

    for case in cases:
        if case.get("requires_rejection") is True:
            inapplicable.append(case)
        else:
            applicable.append(case)

    return applicable, inapplicable


def validate_522_benchmark(
    cases: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Validate the frozen manuscript benchmark invariants.
    """
    ids = [str(c.get("test_id", "")) for c in cases]

    if len(cases) != 522:
        raise ValueError(
            f"Expected 522 cases, got {len(cases)}"
        )

    if any(not x for x in ids):
        raise ValueError("Empty test_id detected")

    if len(set(ids)) != 522:
        raise ValueError("Duplicate test_id detected")

    applicable, inapplicable = split_cases(cases)

    if len(applicable) != 434:
        raise ValueError(
            f"Expected 434 applicable cases, "
            f"got {len(applicable)}"
        )

    if len(inapplicable) != 88:
        raise ValueError(
            f"Expected 88 inapplicable cases, "
            f"got {len(inapplicable)}"
        )

    return {
        "total": 522,
        "applicable": 434,
        "inapplicable": 88,
    }


def normalize_prediction_status(
    value: Any
) -> str:
    """
    Normalize system decision into:

        ANSWER
        REJECT
        UNRESOLVED

    This function intentionally does not infer rejection from
    historical bucket labels.
    """
    if value is None:
        return "UNRESOLVED"

    s = str(value).strip().upper()

    reject_values = {
        "REJECT",
        "REJECTED",
        "REFUSE",
        "REFUSED",
        "INAPPLICABLE",
    }

    answer_values = {
        "ANSWER",
        "ANSWERED",
        "PASS",
        "APPLICABLE",
    }

    unresolved_values = {
        "UNRESOLVED",
        "UNKNOWN",
        "CLARIFY",
        "CLARIFICATION",
        "WARNING",
    }

    if s in reject_values:
        return "REJECT"

    if s in answer_values:
        return "ANSWER"

    if s in unresolved_values:
        return "UNRESOLVED"

    raise ValueError(
        f"Unknown prediction status: {value!r}"
    )


def rejection_confusion(
    cases: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
    status_field: str = "status",
) -> Dict[str, int]:
    """
    Compute rejection-decision confusion counts.

    Positive class = benchmark requires rejection.

    TP: inapplicable and system rejects
    FN: inapplicable and system does not reject
    FP: applicable but system rejects
    TN: applicable and system does not reject

    UNRESOLVED is conservatively treated as non-rejection here.
    Additional coverage metrics should report unresolved cases
    separately.
    """
    validate_522_benchmark(cases)

    tp = fp = tn = fn = unresolved = 0

    for case in cases:
        tid = str(case["test_id"])
        truth_reject = (
            case.get("requires_rejection") is True
        )

        pred = predictions.get(tid, {})
        status = normalize_prediction_status(
            pred.get(status_field)
        )

        if status == "UNRESOLVED":
            unresolved += 1

        pred_reject = status == "REJECT"

        if truth_reject and pred_reject:
            tp += 1
        elif truth_reject and not pred_reject:
            fn += 1
        elif not truth_reject and pred_reject:
            fp += 1
        else:
            tn += 1

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "unresolved": unresolved,
    }


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def rejection_metrics(
    confusion: Dict[str, int]
) -> Dict[str, float]:
    """
    Standard binary rejection metrics.

    These are intentionally named explicitly and should NOT yet
    be automatically equated with manuscript Safety/Coverage/F1
    until the manuscript metric definitions are independently
    frozen and verified.
    """
    tp = confusion["TP"]
    fp = confusion["FP"]
    tn = confusion["TN"]
    fn = confusion["FN"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    accuracy = safe_div(
        tp + tn,
        tp + tn + fp + fn,
    )

    return {
        "rejection_precision": precision,
        "rejection_recall": recall,
        "rejection_f1": f1,
        "decision_accuracy": accuracy,
    }
