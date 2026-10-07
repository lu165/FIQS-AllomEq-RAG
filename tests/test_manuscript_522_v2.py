from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]

MODULE = (
    ROOT
    / "code"
    / "evaluation"
    / "manuscript_522_v2.py"
)

spec = importlib.util.spec_from_file_location(
    "manuscript_522_v2",
    MODULE,
)

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


# ------------------------------------------------------------
# 1. Published confusion matrix reproduces manuscript metrics.
# ------------------------------------------------------------

cm = mod.manuscript_reference_counts()

assert cm.tp == 378
assert cm.fp == 12
assert cm.fn == 56
assert cm.tn == 76
assert cm.total == 522

assert round(cm.safety * 100, 1) == 96.9
assert round(cm.coverage * 100, 1) == 87.1
assert round(cm.f1, 3) == 0.917
assert round(cm.specificity * 100, 1) == 86.4
assert round(cm.npv * 100, 1) == 57.6

assert cm.tp + cm.fn == 434
assert cm.fp + cm.tn == 88


# ------------------------------------------------------------
# 2. Applicable + acceptable equation = TP.
# ------------------------------------------------------------

assert mod.classify_decision(
    ground_truth_applicable=True,
    predicted_status="ANSWER",
    predicted_equation="EQ001",
    acceptable_equations={"EQ001", "EQ002"},
) == "TP"


# ------------------------------------------------------------
# 3. Applicable + wrong equation = FN, not TP.
# ------------------------------------------------------------

assert mod.classify_decision(
    ground_truth_applicable=True,
    predicted_status="ANSWER",
    predicted_equation="EQ999",
    acceptable_equations={"EQ001", "EQ002"},
) == "FN"


# ------------------------------------------------------------
# 4. Applicable + reject/unresolved = FN.
# ------------------------------------------------------------

assert mod.classify_decision(
    ground_truth_applicable=True,
    predicted_status="REJECT",
    acceptable_equations={"EQ001"},
) == "FN"

assert mod.classify_decision(
    ground_truth_applicable=True,
    predicted_status="UNRESOLVED",
    acceptable_equations={"EQ001"},
) == "FN"


# ------------------------------------------------------------
# 5. Inapplicable + answer = FP.
# ------------------------------------------------------------

assert mod.classify_decision(
    ground_truth_applicable=False,
    predicted_status="ANSWER",
    predicted_equation="EQ001",
) == "FP"


# ------------------------------------------------------------
# 6. Inapplicable + reject/unresolved = TN.
# ------------------------------------------------------------

assert mod.classify_decision(
    ground_truth_applicable=False,
    predicted_status="REJECT",
) == "TN"

assert mod.classify_decision(
    ground_truth_applicable=False,
    predicted_status="UNRESOLVED",
) == "TN"


# ------------------------------------------------------------
# 7. Missing acceptable set must never silently become TP.
# ------------------------------------------------------------

try:
    mod.classify_decision(
        ground_truth_applicable=True,
        predicted_status="ANSWER",
        predicted_equation="EQ001",
        acceptable_equations=None,
    )
except ValueError:
    pass
else:
    raise AssertionError(
        "Missing acceptable set should raise ValueError"
    )


print("REFERENCE_COUNTS =", cm)
print(
    "Safety =",
    f"{cm.safety * 100:.1f}%"
)
print(
    "Coverage =",
    f"{cm.coverage * 100:.1f}%"
)
print(
    "F1 =",
    f"{cm.f1:.3f}"
)
print(
    "Specificity =",
    f"{cm.specificity * 100:.1f}%"
)
print(
    "NPV =",
    f"{cm.npv * 100:.1f}%"
)

print("ALL_TESTS_PASS")
