# -*- coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from evaluation.manuscript_522 import (
    split_cases,
    normalize_prediction_status,
    rejection_metrics,
)


def test_split():
    cases = [
        {"test_id": "A", "requires_rejection": False},
        {"test_id": "B", "requires_rejection": True},
        {"test_id": "C"},
    ]

    app, neg = split_cases(cases)

    assert len(app) == 2
    assert len(neg) == 1
    assert neg[0]["test_id"] == "B"


def test_status():
    assert normalize_prediction_status("reject") == "REJECT"
    assert normalize_prediction_status("PASS") == "ANSWER"
    assert normalize_prediction_status("unknown") == "UNRESOLVED"


def test_metrics():
    c = {
        "TP": 8,
        "FP": 2,
        "TN": 8,
        "FN": 2,
        "unresolved": 0,
    }

    m = rejection_metrics(c)

    assert abs(m["rejection_precision"] - 0.8) < 1e-12
    assert abs(m["rejection_recall"] - 0.8) < 1e-12
    assert abs(m["rejection_f1"] - 0.8) < 1e-12
    assert abs(m["decision_accuracy"] - 0.8) < 1e-12


if __name__ == "__main__":
    test_split()
    test_status()
    test_metrics()
    print("ALL_TESTS_PASS")
