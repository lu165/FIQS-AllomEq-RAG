"""
Score fusion used in the manuscript.

S_fuse =
    alpha * S_struct_norm
    + (1 - alpha) * S_dense_norm

Default:
    alpha = 0.3

Therefore:
    structural weight = 0.3
    dense semantic weight = 0.7
"""

from typing import Union

Number = Union[int, float]


def fuse_scores(
    struct_norm: Number,
    dense_norm: Number,
    alpha: float = 0.3,
) -> float:
    """Fuse normalized structural and dense retrieval scores."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")

    return (
        alpha * float(struct_norm)
        + (1.0 - alpha) * float(dense_norm)
    )


if __name__ == "__main__":
    # Minimal deterministic sanity checks.
    assert abs(fuse_scores(1.0, 0.0) - 0.3) < 1e-12
    assert abs(fuse_scores(0.0, 1.0) - 0.7) < 1e-12
    assert abs(fuse_scores(1.0, 1.0) - 1.0) < 1e-12
    print("PASS: manuscript fusion formula")
