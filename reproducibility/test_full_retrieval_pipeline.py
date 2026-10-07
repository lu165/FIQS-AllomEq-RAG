from allomeq_rag.retriever_manuscript_v1 import (
    load_fedb,
    retrieve,
)

from allomeq_rag.structural_score_manuscript_v2 import (
    HistoricalConfig,
    calculate_score_historical,
)

from allomeq_rag.runtime_manuscript_v1 import (
    rank_candidates,
)


db = load_fedb(
"<PRIVATE_ROOT>/homee/林业方程_标准化.json"
)


slots={
    "tree_species":"思茅松",
    "region":"云南普洱",
    "component":"干",
    "D":15,
    "H":10
}


candidates = retrieve(
    db,
    slots,
    top_k=5
)


print("DISCOVERY =",len(candidates))


cfg = HistoricalConfig()


runtime_candidates=[]


for c in candidates:

    score = calculate_score_historical(
        c["record"],
        slots,
        cfg
    )

    runtime_candidates.append(
        {
            "record":c["record"],
            "structural_score":score,
            "cosine_similarity":0.5
        }
    )


ranked = rank_candidates(
    runtime_candidates
)


print("RANKED =",len(ranked))


for r in ranked:

    print(
        r.candidate_id,
        r.structural_score,
        r.fused_score
    )


print("FULL_PIPELINE_PASS")
