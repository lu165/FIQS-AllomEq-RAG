"""
Manuscript-aligned AllomEq-RAG runtime primitives.

IMPORTANT
---------
This is a reconstructed manuscript-aligned runtime module.
It does NOT modify or replace the historical biomass_expert_v24.py.

Candidate ranking:
    S_struct_norm = S_struct / 1000
    S_dense_norm  = (cosine + 1) / 2

    S_fuse =
        alpha * S_struct_norm
        + (1 - alpha) * S_dense_norm

Default:
    alpha = 0.3

Generalized taxonomy/geography matching is discovery-only.
Final recommendation permission must be decided separately by the
explicit five-constraint applicability gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .scoring import (
    normalize_structural_score,
    normalize_dense_score,
    fuse_scores,
)


DEFAULT_ALPHA = 0.3


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    structural_score: float
    cosine_similarity: float
    structural_normalized: float
    dense_normalized: float
    fused_score: float
    record: Dict[str, Any]


def candidate_id(record: Dict[str, Any], fallback: str) -> str:
    """
    Obtain a stable human-readable identifier when available.
    This identifier is for tracing only and is not used in scoring.
    """
    for key in ("id", "equation_id", "record_id", "qid"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def rank_candidates(
    candidates: Iterable[Dict[str, Any]],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> List[RankedCandidate]:
    """
    Rank already-discovered candidates using manuscript-defined fusion.

    Each candidate dictionary must contain:
        record
        structural_score      raw score in [0, 1000]
        cosine_similarity     raw cosine in [-1, 1]

    Discovery and final applicability are intentionally outside this
    function.
    """

    ranked: List[RankedCandidate] = []

    for index, item in enumerate(candidates):
        if "record" not in item:
            raise KeyError("candidate is missing 'record'")
        if "structural_score" not in item:
            raise KeyError("candidate is missing 'structural_score'")
        if "cosine_similarity" not in item:
            raise KeyError("candidate is missing 'cosine_similarity'")

        record = item["record"]
        structural = float(item["structural_score"])
        cosine = float(item["cosine_similarity"])

        s_struct = normalize_structural_score(structural)
        s_dense = normalize_dense_score(cosine)
        fused = fuse_scores(
            structural_score=structural,
            cosine_similarity=cosine,
            alpha=alpha,
        )

        ranked.append(
            RankedCandidate(
                candidate_id=candidate_id(
                    record,
                    fallback=f"candidate_{index:04d}",
                ),
                structural_score=structural,
                cosine_similarity=cosine,
                structural_normalized=s_struct,
                dense_normalized=s_dense,
                fused_score=fused,
                record=record,
            )
        )

    ranked.sort(
        key=lambda x: (
            -x.fused_score,
            -x.structural_normalized,
            -x.dense_normalized,
            x.candidate_id,
        )
    )

    return ranked


def trace_candidate(candidate: RankedCandidate) -> Dict[str, Any]:
    """Return the score trace required for reproducible evaluation."""
    return {
        "candidate_id": candidate.candidate_id,
        "structural_score_raw": candidate.structural_score,
        "structural_score_normalized": candidate.structural_normalized,
        "cosine_similarity_raw": candidate.cosine_similarity,
        "dense_score_normalized": candidate.dense_normalized,
        "alpha": DEFAULT_ALPHA,
        "fused_score": candidate.fused_score,
    }


if __name__ == "__main__":
    demo = [
        {
            "record": {"id": "A"},
            "structural_score": 1000,
            "cosine_similarity": -1.0,
        },
        {
            "record": {"id": "B"},
            "structural_score": 0,
            "cosine_similarity": 1.0,
        },
        {
            "record": {"id": "C"},
            "structural_score": 500,
            "cosine_similarity": 0.0,
        },
    ]

    result = rank_candidates(demo)

    by_id = {x.candidate_id: x for x in result}

    assert abs(by_id["A"].fused_score - 0.3) < 1e-12
    assert abs(by_id["B"].fused_score - 0.7) < 1e-12
    assert abs(by_id["C"].fused_score - 0.5) < 1e-12

    assert [x.candidate_id for x in result] == ["B", "C", "A"]

    print("PASS: manuscript runtime ranking")
    for x in result:
        print(trace_candidate(x))
