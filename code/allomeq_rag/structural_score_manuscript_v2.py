"""
Manuscript-aligned structural scorer derived from:
    the internal historical implementation

This module implements the manuscript-aligned nominal 1000-point structural score.

Important:
- This is a new manuscript-aligned implementation.
- The five nominal contributions are 250 + 250 + 300 + 150 + 50 = 1000.
- The diameter contribution is capped at its nominal 300-point weight.
- Therefore the structural score is bounded by [0, 1000].
- This file must not be represented as the historical implementation.
"""

import re
from dataclasses import dataclass
from typing import Dict


@dataclass
class HistoricalConfig:
    use_species_constraint: bool = True
    use_location_constraint: bool = True
    use_diameter_constraint: bool = True
    use_component_constraint: bool = True
    alpha: float = 0.3


def normalize_component(comp):
    if not comp:
        return comp

    mapping = {
        '干': '干', '干材': '干', '树干': '干', '主干': '干',
        '枝': '枝', '树枝': '枝', '枝条': '枝',
        '叶': '叶', '树叶': '叶', '叶片': '叶',
        '根': '根', '根系': '根', '树根': '根',
        '皮': '皮', '树皮': '皮',
        '地上': '地上', '地上部分': '地上',
        '整株': '整株', '全株': '整株', '整树': '整株',
    }
    return mapping.get(comp, comp)


def parse_d_range(size_str):
    if not size_str:
        return None

    m = re.search(
        r'胸径[范围：:\s]*(\d+\.?\d*)[,，~\-\u2013]\s*(\d+\.?\d*)',
        size_str
    )
    if m:
        return float(m.group(1)), float(m.group(2))

    m2 = re.search(
        r'(\d+\.?\d*)\s*[-\u2013~]\s*(\d+\.?\d*)\s*cm',
        size_str
    )
    if m2:
        return float(m2.group(1)), float(m2.group(2))

    return None


def calculate_score_historical(
    eq_data: Dict,
    extracted: Dict,
    config: HistoricalConfig
) -> float:

    score = 0.0

    tree = extracted.get('tree_type', '')
    loc = extracted.get('location', '')
    D = extracted.get('D')
    comp = extracted.get('component', '')

    eq_tree = eq_data.get('tree_type', '')
    eq_loc = eq_data.get('location', '')
    eq_size = eq_data.get('size_class', '')
    eq_comp = eq_data.get('component', '')
    eq_vars = eq_data.get('variables', [])

    # 1. Species
    if config.use_species_constraint and tree:
        if tree in eq_tree:
            score += 250
        elif tree[:2] in eq_tree or eq_tree[:2] in tree:
            score += 150
    else:
        score += 125

    # 2. Geography
    if config.use_location_constraint and loc:
        if not eq_loc or eq_loc == '全国':
            score += 200
        else:
            loc_clean = loc.replace('省', '').replace('市', '').replace('县', '')
            eq_loc_clean = (
                eq_loc.replace('省', '').replace('市', '').replace('县', '')
            )

            if eq_loc_clean in loc_clean or loc_clean in eq_loc_clean:
                score += 250
            elif len(eq_loc_clean) >= 2 and len(loc_clean) >= 2:
                if eq_loc_clean[:2] == loc_clean[:2]:
                    score += 180
                else:
                    score += 80
            else:
                score += 100
    else:
        score += 125

    # 3. Diameter
    if config.use_diameter_constraint and D is not None:
        d_range = parse_d_range(eq_size)

        if d_range:
            lo, hi = d_range

            if lo <= D <= hi:
                # Manuscript-aligned diameter contribution is capped at 300.
                # The historical implementation added an extra +50 width bonus,
                # which could raise the nominal 1000-point total to 1050.
                score += 300
            else:
                over = max(D - hi, lo - D)
                score += max(0, 300 - int(over * 10))
        else:
            score += 150
    else:
        score += 150

    # 4. Component
    if config.use_component_constraint and comp:
        comp_norm = normalize_component(comp)
        eq_comp_norm = normalize_component(eq_comp)

        if comp_norm == eq_comp_norm:
            score += 150
    else:
        score += 75

    # 5. Predictor completeness
    H = extracted.get('H')
    has_H = any(v in eq_vars for v in ['H', 'h'])

    if H is not None and has_H:
        score += 50
    elif H is None and not has_H:
        score += 40
    else:
        score += 20

    return score


if __name__ == "__main__":
    cfg = HistoricalConfig()

    eq = {
        "tree_type": "樟子松",
        "location": "黑龙江省",
        "size_class": "胸径5-20cm",
        "component": "地上",
        "variables": ["D", "H"],
    }

    query = {
        "tree_type": "樟子松",
        "location": "黑龙江省",
        "D": 15.0,
        "H": 10.0,
        "component": "地上",
    }

    score = calculate_score_historical(eq, query, cfg)

    print("MANUSCRIPT_STRUCTURAL_SCORE =", score)

    assert score == 1000.0, score

    print("PASS: manuscript-aligned 1000-point maximum reproduced")
