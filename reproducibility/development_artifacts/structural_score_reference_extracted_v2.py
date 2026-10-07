import re
from typing import Dict

class ExperimentConfig:
    """实验配置"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        
        # 结构化约束开关
        self.use_species_constraint = kwargs.get('use_species_constraint', True)
        self.use_location_constraint = kwargs.get('use_location_constraint', True)
        self.use_diameter_constraint = kwargs.get('use_diameter_constraint', True)
        self.use_component_constraint = kwargs.get('use_component_constraint', True)
        
        # 混合检索权重 (α)
        self.alpha = kwargs.get('alpha', 0.3)
        
        # Judge模块
        self.use_judge = kwargs.get('use_judge', True)
        
        # 其他参数
        self.min_score_threshold = kwargs.get('min_score_threshold', 0)
        
    def is_structural(self) -> bool:
        """是否使用结构化约束"""
        return (self.use_species_constraint or 
                self.use_location_constraint or 
                self.use_diameter_constraint or 
                self.use_component_constraint)
    
    def __str__(self):
        constraints_str = (f"种{int(self.use_species_constraint)}/"
                          f"地{int(self.use_location_constraint)}/"
                          f"径{int(self.use_diameter_constraint)}/"
                          f"组{int(self.use_component_constraint)}")
        return (f"Config[{self.name}]: "
                f"约束({constraints_str}) "
                f"α={self.alpha} Judge={'ON' if self.use_judge else 'OFF'}")

def normalize_component(comp):
    """组分标准化"""
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
    """解析径级范围"""
    if not size_str:
        return None
    m = re.search(r'胸径[范围：:\s]*(\d+\.?\d*)[,，~\-\u2013]\s*(\d+\.?\d*)', size_str)
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.search(r'(\d+\.?\d*)\s*[-\u2013~]\s*(\d+\.?\d*)\s*cm', size_str)
    if m2:
        return float(m2.group(1)), float(m2.group(2))
    return None

def calculate_score_with_config(
    eq_data: Dict,
    extracted: Dict,
    config: ExperimentConfig
) -> float:
    """
    根据配置计算方程得分
    
    评分系统：
    - 树种: 250分
    - 地区: 250分
    - 径级: 300分
    - 组分: 150分
    - 变量完备性: 50分
    总计: 1000分
    """
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
    
    # 1. 树种约束得分 (250分)
    if config.use_species_constraint and tree:
        if tree in eq_tree:
            score += 250
        elif tree[:2] in eq_tree or eq_tree[:2] in tree:
            score += 150
        else:
            score += 0
    else:
        score += 125  # 不使用约束时给中等分
    
    # 2. 地区约束得分 (250分)
    if config.use_location_constraint and loc:
        if not eq_loc or eq_loc == '全国':
            score += 200
        else:
            loc_clean = loc.replace('省', '').replace('市', '').replace('县', '')
            eq_loc_clean = eq_loc.replace('省', '').replace('市', '').replace('县', '')
            
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
    
    # 3. 径级约束得分 (300分)
    if config.use_diameter_constraint and D is not None:
        d_range = parse_d_range(eq_size)
        if d_range:
            lo, hi = d_range
            if lo <= D <= hi:
                score += 300
                width = hi - lo
                if 10 <= width <= 25:
                    score += 50
            else:
                over = max(D - hi, lo - D)
                score += max(0, 300 - int(over * 10))
        else:
            score += 150
    else:
        score += 150
    
    # 4. 组分约束得分 (150分)
    if config.use_component_constraint and comp:
        comp_norm = normalize_component(comp)
        eq_comp_norm = normalize_component(eq_comp)
        
        if comp_norm == eq_comp_norm:
            score += 150
        else:
            score += 0
    else:
        score += 75
    
    # 5. 变量完备性 (50分)
    H = extracted.get('H')
    has_H = any(v in eq_vars for v in ['H', 'h'])
    if H is not None and has_H:
        score += 50
    elif H is None and not has_H:
        score += 40
    else:
        score += 20
    
    return score
