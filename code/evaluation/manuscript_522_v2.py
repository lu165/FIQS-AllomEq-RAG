"""
Manuscript-aligned evaluator for the 522-query AllomEq-RAG benchmark.

Ground-truth semantics
----------------------
Applicable:
    requires_rejection is not True

Inapplicable:
    requires_rejection is True

Decision semantics
------------------
TP:
    Applicable query for which an expert-approved equation is returned.

FP:
    Inapplicable query for which an equation is accepted.

FN:
    Applicable query that is rejected, unresolved, missed, or returns
    no expert-approved equation.

TN:
    Inapplicable query that is correctly refused.

Primary manuscript metrics
--------------------------
Safety   = Precision = TP / (TP + FP)
Coverage = Recall    = TP / (TP + FN)
F1       = harmonic mean of Safety and Coverage
Specificity = TN / (TN + FP)
NPV         = TN / (TN + FN)

Important:
Returning any equation is NOT sufficient for TP on an applicable query.
The returned equation must belong to the expert-approved acceptable set.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Set


EXPECTED_TOTAL = 522
EXPECTED_APPLICABLE = 434
EXPECTED_INAPPLICABLE = 88


def safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def safety(self) -> float:
        """Manuscript Safety = Precision."""
        return safe_div(self.tp, self.tp + self.fp)

    @property
    def coverage(self) -> float:
        """Manuscript Coverage = Recall."""
        return safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p = self.safety
        r = self.coverage
        return safe_div(2 * p * r, p + r)

    @property
    def specificity(self) -> float:
        return safe_div(self.tn, self.tn + self.fp)

    @property
    def npv(self) -> float:
        return safe_div(self.tn, self.tn + self.fn)

    @property
    def rejection_rate(self) -> float:
        """Predicted inapplicable / all queries."""
        return safe_div(self.fn + self.tn, self.total)


def is_inapplicable(case: dict) -> bool:
    """Formal benchmark label."""
    return case.get("requires_rejection") is True


def is_applicable(case: dict) -> bool:
    return not is_inapplicable(case)


def validate_benchmark_split(cases: Iterable[dict]) -> dict:
    cases = list(cases)

    applicable = sum(is_applicable(x) for x in cases)
    inapplicable = sum(is_inapplicable(x) for x in cases)

    result = {
        "total": len(cases),
        "applicable": applicable,
        "inapplicable": inapplicable,
    }

    if len(cases) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} cases, got {len(cases)}"
        )

    if applicable != EXPECTED_APPLICABLE:
        raise ValueError(
            f"Expected {EXPECTED_APPLICABLE} applicable cases, "
            f"got {applicable}"
        )

    if inapplicable != EXPECTED_INAPPLICABLE:
        raise ValueError(
            f"Expected {EXPECTED_INAPPLICABLE} inapplicable cases, "
            f"got {inapplicable}"
        )

    return result


def normalize_equation_id(value: Any) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def normalize_acceptable_set(values: Iterable[Any]) -> Set[str]:
    out = set()

    for value in values:
        norm = normalize_equation_id(value)
        if norm is not None:
            out.add(norm)

    return out


def classify_decision(
    *,
    ground_truth_applicable: bool,
    predicted_status: str,
    predicted_equation: Any = None,
    acceptable_equations: Optional[Iterable[Any]] = None,
) -> str:
    """
    Convert one system decision to TP/FP/FN/TN.

    predicted_status:
        ANSWER
        REJECT
        UNRESOLVED

    For an applicable query:
        ANSWER counts as TP only when predicted_equation belongs to
        acceptable_equations. Otherwise it is FN.

    For an inapplicable query:
        ANSWER is FP.
        REJECT or UNRESOLVED is TN.

    UNRESOLVED is therefore conservative:
        applicable   -> FN
        inapplicable -> TN

    This function intentionally does not infer acceptable equations.
    They must be supplied by the benchmark/evaluation layer.
    """

    status = str(predicted_status).strip().upper()

    allowed = {
        "ANSWER",
        "REJECT",
        "UNRESOLVED",
    }

    if status not in allowed:
        raise ValueError(
            f"Unknown predicted_status: {predicted_status!r}"
        )

    if ground_truth_applicable:

        if status != "ANSWER":
            return "FN"

        if acceptable_equations is None:
            raise ValueError(
                "Applicable ANSWER requires an explicit "
                "acceptable_equations set."
            )

        acceptable = normalize_acceptable_set(
            acceptable_equations
        )

        predicted = normalize_equation_id(
            predicted_equation
        )

        if predicted is not None and predicted in acceptable:
            return "TP"

        return "FN"

    # Ground truth is inapplicable.
    if status == "ANSWER":
        return "FP"

    return "TN"


def confusion_from_labels(labels: Iterable[str]) -> ConfusionMatrix:
    counts = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
    }

    for label in labels:
        label = str(label).upper()

        if label not in counts:
            raise ValueError(
                f"Unknown confusion label: {label!r}"
            )

        counts[label] += 1

    return ConfusionMatrix(
        tp=counts["TP"],
        fp=counts["FP"],
        fn=counts["FN"],
        tn=counts["TN"],
    )


def manuscript_reference_counts() -> ConfusionMatrix:
    """
    Published manuscript counts.

    These values are for evaluator verification only.
    They are NOT generated predictions and must not be used as a
    substitute for rerunning the 522-query system.
    """
    return ConfusionMatrix(
        tp=378,
        fp=12,
        fn=56,
        tn=76,
    )


if __name__ == "__main__":
    cm = manuscript_reference_counts()

    print("TP =", cm.tp)
    print("FP =", cm.fp)
    print("FN =", cm.fn)
    print("TN =", cm.tn)
    print("TOTAL =", cm.total)

    print("Safety =", cm.safety)
    print("Coverage =", cm.coverage)
    print("F1 =", cm.f1)
    print("Specificity =", cm.specificity)
    print("NPV =", cm.npv)
    print("RejectionRate =", cm.rejection_rate)
