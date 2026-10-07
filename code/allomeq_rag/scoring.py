"""
AllomEq-RAG score normalization and fusion.

Manuscript specification
------------------------
Structural score:
    raw range: 0--1000
    S_struct_norm = S_struct / 1000

Dense score:
    cosine similarity range: -1--1
    S_dense_norm = (cosine_similarity + 1) / 2

Fusion:
    S_fuse =
        alpha * S_struct_norm
        + (1 - alpha) * S_dense_norm

Default:
    alpha = 0.3
"""

from typing import Union

Number = Union[int, float]


def normalize_structural_score(score: Number) -> float:
    """Map the structural score from [0, 1000] to [0, 1]."""
    score = float(score)

    if not 0.0 <= score <= 1000.0:
        raise ValueError(
            f"structural score must be in [0, 1000], got {score}"
        )

    return score / 1000.0


def normalize_dense_score(cosine_similarity: Number) -> float:
    """Map cosine similarity from [-1, 1] to [0, 1]."""
    cosine_similarity = float(cosine_similarity)

    if not -1.0 <= cosine_similarity <= 1.0:
        raise ValueError(
            "cosine similarity must be in [-1, 1], "
            f"got {cosine_similarity}"
        )

    return (cosine_similarity + 1.0) / 2.0


def fuse_scores(
    structural_score: Number,
    cosine_similarity: Number,
    alpha: float = 0.3,
) -> float:
    """Return the manuscript-defined normalized fusion score."""
    alpha = float(alpha)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    s_struct_norm = normalize_structural_score(structural_score)
    s_dense_norm = normalize_dense_score(cosine_similarity)

    return (
        alpha * s_struct_norm
        + (1.0 - alpha) * s_dense_norm
    )


if __name__ == "__main__":

    # Structural normalization
    assert normalize_structural_score(0) == 0.0
    assert normalize_structural_score(1000) == 1.0
    assert normalize_structural_score(500) == 0.5

    # Dense normalization
    assert normalize_dense_score(-1) == 0.0
    assert normalize_dense_score(0) == 0.5
    assert normalize_dense_score(1) == 1.0

    # Fusion sanity checks
    assert abs(fuse_scores(1000, 1.0, 0.3) - 1.0) < 1e-12

    # struct=1000 -> 1.0
    # cos=-1 -> 0.0
    # result = 0.3
    assert abs(fuse_scores(1000, -1.0, 0.3) - 0.3) < 1e-12

    # struct=0 -> 0.0
    # cos=1 -> 1.0
    # result = 0.7
    assert abs(fuse_scores(0, 1.0, 0.3) - 0.7) < 1e-12

    print("PASS: manuscript-defined AllomEq-RAG scoring")
