"""
Manuscript-aligned FEDB retriever.

Public deterministic retrieval component.

Input:
    query slots

Output:
    ranked FEDB candidates

This module contains:
    - structural discovery
    - semantic assistance

It does NOT contain:
    - LLM generation
    - applicability judgment
    - final evaluation
"""

import re
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def normalize_text(x):
    return re.sub(r"\s+", "", str(x or ""))


def load_fedb(path):

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data["equations"]

    return data


def normalize_component(comp):

    mapping = {
        "干材":"干",
        "树干":"干",
        "枝条":"枝",
        "树枝":"枝",
        "树叶":"叶",
        "叶片":"叶",
        "树根":"根",
        "根系":"根",
        "全株":"整株",
        "整树":"整株",
    }

    return mapping.get(comp, comp)




def loc_match_score(u, e):
    """
    Geographic compatibility score.
    """
    if not u or not e:
        return 0

    un = normalize_text(u)
    en = normalize_text(e)

    if un == en:
        return 4

    if un in en or en in un:
        return 3

    if un[:2] == en[:2]:
        return 1

    return 0



def has_var(eq, var):

    return bool(
        re.search(
            r'(?<![A-Za-z0-9])'
            + re.escape(var)
            + r'(?![A-Za-z0-9_])',
            str(eq)
        )
    )



def param_score(eq_str, slots):

    score = 0

    if slots.get("H") is not None:
        if has_var(eq_str, "H"):
            score += 2

    if slots.get("D0") is not None:
        if has_var(eq_str, "D0"):
            score += 2

    if slots.get("V") is not None:
        if has_var(eq_str, "V"):
            score += 3

    if slots.get("A") is not None:
        if has_var(eq_str, "A"):
            score += 3

    return score



def parse_d_range(size_class):

    if not size_class:
        return None

    text = str(size_class)

    patterns = [
        r'胸径.*?(\d+\.?\d*)\s*[-~～–]\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*[-~～–]\s*(\d+\.?\d*)\s*cm'
    ]


    for pat in patterns:

        m = re.search(
            pat,
            text
        )

        if m:
            lo = float(m.group(1))
            hi = float(m.group(2))

            if lo > hi:
                lo, hi = hi, lo

            return lo, hi


    return None



def d_range_score(size_class, D):

    if D is None:
        return 0

    dr = parse_d_range(size_class)

    if not dr:
        return 0

    lo, hi = dr

    if lo <= D <= hi:
        return 2

    if D <= hi * 1.3:
        return 0

    if D > hi * 1.5:
        return -3

    return 0



def structural_discovery(
    db,
    slots,
    top_k=80
):
    """
    Manuscript-aligned structural discovery.

    Returns:
        [
          {
            "record": FEDB record,
            "structural_score": raw score
          }
        ]
    """

    tree = normalize_text(
        slots.get("tree_species","")
    )

    loc = normalize_text(
        slots.get("region","")
    )

    comp = normalize_component(
        slots.get("component","")
    )

    D = slots.get("D")


    scored=[]


    for r in db:

        score=0


        eq_tree = normalize_text(
            r.get("tree_type","")
        )

        eq_loc = normalize_text(
            r.get("location","")
        )

        eq_comp = normalize_component(
            r.get("component","")
        )


        # species
        if tree:

            if tree == eq_tree:
                score += 250

            elif tree in eq_tree or eq_tree in tree:
                score += 150


        # geography
        score += (
            loc_match_score(
                loc,
                eq_loc
            ) * 50
        )


        # component (hard preference)

        if comp:

            if comp == eq_comp:
                score += 150

            else:
                # component mismatch penalty
                score -= 300


        # diameter
        score += (
            d_range_score(
                r.get("size_class",""),
                D
            ) * 100
        )


        # variable completeness
        variables = r.get(
            "variables",
            []
        )

        if slots.get("H") is not None:
            if "H" in variables:
                score += 50


        if score > 0:

            scored.append(
                {
                    "record": r,
                    "structural_score": float(score)
                }
            )


    scored.sort(
        key=lambda x:
        -x["structural_score"]
    )


    return scored[:top_k]



def retrieve(
    db,
    slots,
    top_k=5
):
    """
    Public deterministic retriever.

    Discovery only.
    No applicability judgement.
    """

    candidates = structural_discovery(
        db,
        slots,
        top_k=80
    )


    for c in candidates:

        # placeholder semantic score
        # real embedding is optional
        c["cosine_similarity"] = 0.0


    return candidates[:top_k]


