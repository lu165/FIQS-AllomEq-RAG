# FIQS / AllomEq-RAG

A reproducible retrieval framework for forestry allometric equation question answering.

## Overview

FIQS / AllomEq-RAG integrates:

- deterministic equation retrieval
- constraint-governed structural scoring
- semantic assistance
- applicability-aware evaluation

The framework is designed for reproducible retrieval and ranking of forestry allometric equations.

## Public Data

- FIDF: 115,375 instruction-response pairs
- FEDB: 7,394 standardized forestry allometric equations
- FETS: 510 held-out QA benchmark samples
- Evaluation benchmark: 522-query AllomEq-RAG evaluation set

## Core Method

S_fuse = alpha * S_struct_norm + (1 - alpha) * S_dense_norm, with alpha = 0.3

Structural score components: species, geography, diameter range, component, predictor completeness (maximum 1000).

## Reproducibility

Run:

    bash run_smoke_test.sh

Expected output: SMOKE_TEST_PASS

## Notes

Development artifacts preserve historical provenance. Public execution does not require private filesystem paths.
