# -*- coding: utf-8 -*-
"""
Historical v24 evaluator helpers.

Automatically extracted from biomass_expert_v24.py for historical
reproducibility auditing.

This is historical evaluator logic, NOT the manuscript-aligned
AllomEq-RAG runtime.

The original v24 source remains unchanged.
"""

import re
import math
import json
from collections import defaultdict

LOC_ALIAS = {
    'dongbei':       ['东北','东北地区','东北林区','东三省','黑龙江','吉林','辽宁','黑龙江省','吉林省','辽宁省'],
    'neimenggu':     ['内蒙古','内蒙古地区','内蒙古自治区','半干旱区','内蒙古林区'],
    'xiaoqinling':   ['小秦岭','小秦岭国家级自然保护区','小秦岭自然保护区'],
    'xiaoxinganling':['小兴安岭','小兴安岭国家级自然保护区'],
    'daxinganling':  ['大兴安岭','大兴安岭林区','大兴安岭地区'],
    'qinling':       ['秦岭','陕西秦岭','秦岭地区'],
    'guangxi':       ['广西','广西省','广西壮族自治区'],
    'yunnan':  ['云南','云南省'],   'fujian':   ['福建','福建省'],
    'zhejiang':['浙江','浙江省'],   'hunan':    ['湖南','湖南省'],
    'hubei':   ['湖北','湖北省'],   'sichuan':  ['四川','四川省'],
    'guizhou': ['贵州','贵州省'],   'guangdong':['广东','广东省'],
    'anhui':   ['安徽','安徽省'],   'jiangxi':  ['江西','江西省'],
    'jiangsu': ['江苏','江苏省'],   'shandong': ['山东','山东省'],
    'shanxi':  ['山西','山西省'],   'hebei':    ['河北','河北省'],
    'henan':   ['河南','河南省'],   'gansu':    ['甘肃','甘肃省'],
    'shaanxi': ['陕西','陕西省'],   'qinghai':  ['青海','青海省'],
    'xinjiang':['新疆','新疆维吾尔自治区'],
    'xizang':  ['西藏','西藏自治区'], 'hainan':['海南','海南省'],
    'beijing': ['北京','北京市'],   'shanghai': ['上海','上海市'],
    'beijing_miao':['北京妙峰山','北京妙峰山林场'],
}

_LOC_REV = {}

for k, aliases in LOC_ALIAS.items():
    for a in aliases:
        _LOC_REV[re.sub(r'\s+','',a)] = k

def loc_canon(s):
    return _LOC_REV.get(re.sub(r'\s+','',str(s or '')), re.sub(r'\s+','',str(s or '')))

def loc_match_score(u, e):
    if not u or not e: return 0
    un = re.sub(r'\s+','',u); en = re.sub(r'\s+','',e)
    if loc_canon(un) == loc_canon(en): return 4
    if un == en: return 4
    if un in en or en in un: return 3
    if un[:2] == en[:2]: return 1
    return 0

COMP_TO_STD = {
    '干':'干','干总量':'干','干材':'干','干生物量':'干','干重':'干','树干':'干','茎干':'干','茎生物量':'干',
    '枝':'枝','枝总量':'枝','枝材':'枝','枝条':'枝','枝生物量':'枝','树枝':'枝',
    '叶':'叶','叶总量':'叶','叶片':'叶','叶生物量':'叶','树叶':'叶',
    '根':'根','根系':'根','总根':'根','根生物量':'根','粗根':'根','细根':'根',
    '大根':'根','小根':'根','侧根':'根','树根':'根','树桩':'根','树桩和主根':'根',
    '地上':'地上','地上总生物量':'地上','地上生物量':'地上','地上部分':'地上',
    '地上部分总量':'地上','森林地上生物量':'地上',
    '地下生物量':'地下','地下部分总量':'地下','地下':'地下',
    '整株':'整株','整株生物量':'整株','全株':'整株','全树总量':'整株','全树':'整株',
    '总生物量':'整株','总计':'整株','生物量':'整株','全株生物量':'整株',
    '单株生物量':'整株','单株总量':'整株','整体':'整株','整体生物量':'整株',
    '皮':'皮','树皮':'皮','干皮':'皮',
    '树冠':'树冠','volume':'材积','材积量':'材积','林分':'材积','蓄积量':'材积','蓄积':'材积',
    '碳储量':'碳储量','地上碳':'碳储量','碳汇':'碳储量','有机碳':'碳储量','碳密度':'碳储量',
    '果实':'果实','花':'果实','花和果实':'果实',
    '叶面积':'叶','叶面积指数':'叶','LAI':'叶',
    '单位面积生物量':'单位面积生物量',
    '异速生长':'整株','异速生长方程':'整株','生物量方程':'整株',
    '形率':'干','树干形率':'干','胸高形率':'干',
    '材积方程':'材积','立木材积':'材积',
    '基径方程':'整株','地径方程':'整株',
}

def normalize_eq(eq):
    eq = re.sub(r'\s+','',str(eq or ''))
    eq = eq.replace('\u00b9','^1').replace('\u00b2','^2').replace('\u00b3','^3')
    eq = eq.replace('^^','^').replace('\u00d7','*').replace('\uff0a','*')
    eq = eq.replace('\uff08','(').replace('\uff09',')')
    for pat, repl in [
        (r'^ln\(W\)=(.+)$',  lambda m: 'W=exp('+m.group(1)+')'),
        (r'^lnW=(.+)$',      lambda m: 'W=exp('+m.group(1)+')'),
        (r'^log\(W\)=(.+)$', lambda m: 'W=exp(2.303*('+m.group(1)+'))'),
        (r'^logW=(.+)$',     lambda m: 'W=exp(2.303*('+m.group(1)+'))'),
    ]:
        m = re.match(pat, eq)
        if m: eq = repl(m)
    eq = re.sub(r'(\.\d*[1-9])0+\b',r'\1',eq)
    eq = re.sub(r'\.0+\b','',eq)
    return eq

def is_wrong_prem(c):
    return ('\u9519\u8bef\u524d\u63d0' in (c.get('gen_type_tags') or []) or
            c.get('test_id','').startswith('GEN_\u9519\u8bef'))

def is_valid_equation(eq_str):
    return bool(re.search(r'[=^*/]', str(eq_str)))

def build_gold_set(case, eq_override=None):
    sp=re.sub(r'\s+','',case.get('tree_type','') or '')
    loc=re.sub(r'\s+','',case.get('location','') or '')
    comp_raw=case.get('component','') or ''; comp_std=COMP_TO_STD.get(comp_raw,comp_raw)
    eq_src=eq_override or str(case.get('equation','') or ''); primary=normalize_eq(eq_src)
    if not primary: return set(),''
    gold_set={primary}
    if sp and loc and comp_std:
        for r in db:
            if (re.sub(r'\s+','',r.get('tree_type',''))==sp and
                loc_match_score(loc,re.sub(r'\s+','',r.get('location','')))>=3 and
                r.get('component','')==comp_std):
                gold_set.add(normalize_eq(r.get('equation','')))
    return gold_set, primary
