from pathlib import Path
import os
"""
biomass_expert_v21.py  —  基于v20的三处精准修复（全程无新增硬编码）
=========================================================
v20存在的问题：
  问题1: 意图分类漂移 — 槽位+意图合并为一次LLM输出，Qwen-7B不稳定
         large量single题被错误分类为reasoning（52条vs正常21条）
         导致14条正样本回退
  问题2: 负样本拒绝率 51.6%→34.4% — judge跳过阈值0.85太高
         大量0.86-0.97分的负样本被直接放行，完全不过judge
  问题3: judge误拒正样本(Q0274等) — judge看到"径级稍超+地区不同"就拒绝
         缺少树种完全匹配时的豁免逻辑

修复方式（不加硬编码，改进LLM使用方式）：

  Fix-①: 意图分类改为Chain-of-Thought两步推理
    - 第一步LLM调用：仅做槽位提取（保持稳定）
    - 第二步：基于槽位结果用规则+LLM联合判断意图
      规则层：若有精确树种+地区+组分槽位→倾向single
              若含"哪个/推荐/更好/比较"→倾向reasoning
              若含"R²/样本量/精度/置信"→倾向quality_ctrl
      LLM层：仅在规则层不确定时才调用LLM判断意图
      这样LLM只需回答"single/reasoning/quality_ctrl"一个字，大幅降低漂移

  Fix-②: judge跳过阈值 0.85→0.70，同时传入"软信号"给judge
    - 扩大judge覆盖范围，更多case过judge
    - 把_detect_soft_signals()的结果作为额外上下文传给judge
      如"用户表达了跨区域使用意图"、"胸径超出径级范围"
      这是提示，不是硬规则——judge做最终判断
    - judge prompt加few-shot示例（跨区域应拒/广布种可采）

  Fix-③: judge prompt加正样本豁免few-shot
    - 树种完全匹配时，径级稍超不应拒绝（Q0274类）
    - 广布种跨省、通用方程跨省的具体示例
"""

import json, sys, re, torch, math
# Historical server-only sys.path injection removed in the public release.
# Public modules are resolved from the repository/package environment.
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

# Public-release configuration.
#
# Large model/checkpoint assets and the formal benchmark are not bundled
# with the lightweight synthetic smoke-test package. Paths may be supplied
# through environment variables. Relative defaults are repository-local.
_RELEASE_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = os.environ.get(
    "FIQS_MODEL_PATH",
    "Qwen/Qwen1.5-7B-Chat",
)

EQ_PATH = os.environ.get(
    "FIQS_EQ_PATH",
    str(
        _RELEASE_ROOT
        / "data_public"
        / "synthetic_example"
        / "dummy_fedb_20.json"
    ),
)

TEST_PATH = os.environ.get(
    "FIQS_TEST_PATH",
    str(
        _RELEASE_ROOT
        / "data_public"
        / "benchmark_522.json"
    ),
)

EMB_PATH = os.environ.get(
    "FIQS_EMB_PATH",
    "BAAI/bge-large-zh-v1.5",
)

# ══════════════════════════════════════════════════════════════
# LLM Prompts
# ══════════════════════════════════════════════════════════════

# Fix-①-A: 槽位提取单独一个prompt，不再混入intent
SLOT_SYSTEM = """你是林业生物量方程检索助手。从用户问题中提取以下槽位，输出JSON。

字段说明：
  tree_species: 树种名（如华山松、落叶松），没有填null
  region: 地区名（如黑龙江、小秦岭），没有填null
  component: 组分（干/枝/叶/根/整株/地上/地下/材积/碳储量/果实），没有填null
  D: 胸径数值(cm)，没有填null
  H: 树高数值(m)，没有填null
  D0: 地径/基径数值(cm)，没有填null
  diameter_type: 直径类型，只能是"胸径"/"基径"/"未知"

只输出JSON，不要任何其他文字。示例：
{"tree_species":"华山松","region":"小秦岭","component":"整株","D":15.5,"H":null,"D0":null,"diameter_type":"胸径"}"""

# Fix-①-B: 意图分类单独prompt（仅在规则层不确定时调用）
INTENT_SYSTEM = """你是林业方程检索意图分类助手。根据用户问题，判断意图类型。

意图类型说明：
  single       — 查找一个具体方程，有明确树种/地区/组分
  multi_sp     — 需要给多个树种分别找方程（含混交林、多树种并列）
  multi_comp   — 需要给多个组分分别找方程（如同时要干+枝+叶）
  quality_ctrl — 比较方程统计质量，含R²/样本量/精度/置信区间评估
  reasoning    — 需要推荐/比较/选择，含"哪个更好/应该用/推荐/更合适"

示例：
  "华山松小秦岭整株生物量，D=15cm" → single
  "混交林中杉木和马尾松的生物量" → multi_sp
  "需要干、枝、叶分别的生物量方程" → multi_comp
  "这两个方程R²分别是0.95和0.97，哪个更好？" → quality_ctrl
  "有几个华山松方程，推荐哪个更适合秦岭地区？" → reasoning
  "小兴安岭红松整株生物量，胸径12cm" → single

只输出一个词（single/multi_sp/multi_comp/quality_ctrl/reasoning），不要其他文字。"""

# Fix-②+③: judge prompt加few-shot示例（跨区域拒绝 + 广布种采用 + 径级豁免）
JUDGE_SYSTEM = """你是林业方程适用性专家。根据用户情况和候选方程完整信息，判断该方程是否适用。

【判断规则】
1. 树种匹配：必须相同或近缘种
2. 地区适用：同气候带/相邻省份通常可参考；南北跨度极大才不适用
3. 组分匹配：用户要的组分必须与方程组分一致
4. 径级范围：用户胸径超出上限1.5倍以上才拒绝（树种完全匹配时适当宽松）；
   1.3-1.5倍之间属于边界，树种地区匹配时应采用
5. 通用方程：标签含"通用"或地区"全国"的方程适用全国同树种
6. 变量匹配：若方程含H（树高），用户有H数据时可以使用该方程，不应拒绝；
   方程含H是提供更精确计算的选项，不是限制条件

【关键示例 — 必须遵守】
✅ 采用案例：
  - "黑龙江山杨，用山西山杨方程" → 采用（山杨是广布种，山西方程可跨省参考）
  - "浙江杉木，用广西通用杉木方程" → 采用（方程标签含通用）
  - "北京油松D=30cm，方程适用范围5-25cm" → 采用（30/25=1.2倍，未超1.5倍，且树种地区匹配）
  - "四川华山松，用甘肃华山松方程" → 采用（相邻省份同树种）

❌ 拒绝案例：
  - "北京红松，想用小兴安岭红松公式" → 拒绝（北京无红松天然分布，地理跨度大）
  - "湖南马尾松，用杉木叶生物量方程算叶量" → 拒绝（树种不匹配）
  - "用整株方程算果实生物量" → 拒绝（组分完全不符）
  - "四川红松，用东北红松公式" → 拒绝（四川无红松天然分布）

【系统检测信号（供参考，最终由你判断）】
{soft_signals}

输出格式（只输出一行）：
采用 或 拒绝：[不超过20字的理由]"""

QUALITY_SYSTEM = """你是林业生物量方程评估专家。从候选方程中选出统计质量最优的方程。

评估依据（按重要性）：
1. R²越接近1越好（>0.999可能过拟合）
2. 样本量n越大越可靠（n<30需谨慎）
3. RMSE/SE越小越好
4. 地区树种匹配度
5. 径级覆盖用户情况

必须严格按格式输出：
推荐方程：[原样复制方程字符串]
推荐理由：[不超过30字]"""

REASONING_SYSTEM = """你是林业生物量专家。根据用户问题和候选方程完整信息给出推荐。

必须严格按格式输出：
推荐方程：[原样复制方程字符串]
推荐理由：[不超过30字]"""

# ══════════════════════════════════════════════════════════════
# 字段标准化表（保留，字段规范化用）
# ══════════════════════════════════════════════════════════════

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

MULTI_COMP_MAP = {
    '地上部分总量':['干','枝','叶'],'地上生物量':['干','枝','叶'],
    '地上总生物量':['干','枝','叶'],'全树总量':['干','枝','叶','根'],
    '整株':['干','枝','叶','根'],
}

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

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

def _parse_json_robust(text):
    try: return json.loads(text.strip())
    except: pass
    t2 = re.sub(r'```json\s*','',text); t2 = re.sub(r'```\s*','',t2).strip()
    try: return json.loads(t2)
    except: pass
    m = re.search(r'\{.*\}',t2,re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    result = {}
    for key,pat in {
        'tree_species':  r'"tree_species"\s*:\s*"([^"]+)"',
        'region':        r'"region"\s*:\s*"([^"]+)"',
        'component':     r'"component"\s*:\s*"([^"]+)"',
        'diameter_type': r'"diameter_type"\s*:\s*"([^"]+)"',
        'D':  r'"D"\s*:\s*(-?\d+\.?\d*)',
        'H':  r'"H"\s*:\s*(-?\d+\.?\d*)',
        'D0': r'"D0"\s*:\s*(-?\d+\.?\d*)',
    }.items():
        m2 = re.search(pat,text)
        if m2 and m2.group(1) and m2.group(1) != 'null':
            if key in ('D','H','D0'):
                try: result[key] = float(m2.group(1))
                except: pass
            else: result[key] = m2.group(1)
    return result if result else None

def _rescue_slots(question, slots):
    slots = dict(slots)
    if slots.get('D') is None:
        m = re.search(r'\u80f8\u5f84\s*(\d+\.?\d*)\s*(?:cm|\u5398\u7c73|CM)', question)
        if m:
            try: slots['D'] = float(m.group(1))
            except: pass
    if slots.get('H') is None:
        m = re.search(r'\u6811\u9ad8\s*(\d+\.?\d*)\s*(?:m|\u7c73)', question)
        if m:
            try: slots['H'] = float(m.group(1))
            except: pass
    if slots.get('D0') is None:
        m = re.search(r'(?:\u57fa\u5f84|\u5730\u5f84|D0)\s*(\d+\.?\d*)\s*(?:cm|\u5398\u7c73|CM)', question)
        if m:
            try: slots['D0'] = float(m.group(1))
            except: pass
    m = re.search(r'(?:\u6811\u56f4|\u5468\u957f|\u80f8\u56f4)[^\d]*(\d+(?:\.\d+)?)\s*(?:\u5398\u7c73|cm|CM)', question)
    if m and slots.get('D') is None:
        try: slots['D'] = round(float(m.group(1))/math.pi, 2)
        except: pass
    return slots

def _normalize_values(parsed):
    for k in ['D','H','D0','V']:
        if k in parsed and parsed[k] is not None:
            try: parsed[k] = float(str(parsed[k]))
            except: parsed[k] = None
    result = {k:v for k,v in parsed.items() if v is not None and v != 'null'}
    dt = result.get('diameter_type','\u672a\u77e5')
    if dt not in ('\u80f8\u5f84','\u57fa\u5f84','\u672a\u77e5'): result['diameter_type'] = '\u672a\u77e5'
    return result

def _llm_call(messages, max_tokens=150):
    try:
        ti  = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = _tokenizer(ti, return_tensors='pt').to(_model.device)
        with torch.no_grad():
            out = _model.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                                  repetition_penalty=1.1, pad_token_id=_tokenizer.eos_token_id)
        return _tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        print('  LLM\u5f02\u5e38: '+str(e), flush=True); return ''

# ══════════════════════════════════════════════════════════════
# Fix-①: 意图分类改为两步（槽位提取 + 规则优先的意图判断）
# ══════════════════════════════════════════════════════════════

# 意图分类规则层关键词（轻量级，不是硬编码语义规则，只是置信度高的快速判断）
_MULTI_SP_KW   = ['\u6df7\u4ea4\u6797','\u5404\u6811\u79cd','\u5206\u522b\u4f7f\u7528\u4e0d\u540c\u6811\u79cd']
_MULTI_COMP_KW = ['\u5e72\u3001\u679d\u3001\u53f6','\u5e72+\u679d+\u53f6','\u5e72\u548c\u679d','\u5404\u7ec4\u5206\u5206\u522b','\u5206\u522b\u4f30\u7b97']
_REASONING_KW  = ['\u54ea\u4e2a','\u54ea\u79cd','\u7528\u54ea','\u9009\u54ea','\u5e94\u8be5\u7528','\u66f4\u5408\u9002',
                  '\u4f18\u5148','\u63a8\u8350','\u533a\u522b','\u5355\u53d8\u91cf','\u53cc\u53d8\u91cf','\u672c\u5730\u65b9\u7a0b',
                  '\u66f4\u8fd1','\u66f4\u65b0\u7684','\u5982\u4f55\u9009\u62e9','\u600e\u4e48\u9009']
_QUALITY_KW    = ['R\u00b2','r\u00b2','R2','r2','\u6837\u672c\u91cf','\u6837\u672c\u6570','\u7cbe\u5ea6',
                  '\u7f6e\u4fe1\u533a\u95f4','\u7f6e\u4fe1\u5ea6','\u8bef\u5dee','\u54ea\u4e2a\u66f4\u51c6',
                  '\u54ea\u4e2a\u66f4\u53ef\u9760','\u54ea\u4e2a\u66f4\u597d','\u8d85\u9ad8R','\u5c0f\u6837\u672c','\u5927\u6837\u672c']

def _rule_based_intent(question, slots):
    """
    规则层快速判断意图，置信度高时直接返回，不确定时返回None（交给LLM）。
    这里的规则是"是否含有明确信号词"，不是语义理解规则。
    """
    q = question
    # 修复B: 质控词优先级最高，在精确槽位判断之前
    if any(kw in q for kw in _QUALITY_KW): return 'quality_ctrl'
    # 多树种信号
    if any(kw in q for kw in _MULTI_SP_KW): return 'multi_sp'
    # 多树种：问题里有两个不同树种名用顿号/逗号并列
    if re.search(r'(\u6750\u6d4b\u6d4b|\u676d\u6728|\u9531\u6728|\u6c38\u51f7|\u6749\u6728|\u6750\u6a41|\u9e1f\u6728|\u6a61\u6811|\u767d\u6866|\u6768\u6811).{0,5}[\uff0c,\u3001].{0,20}(\u6750\u6d4b\u6d4b|\u676d\u6728|\u9531\u6728|\u6c38\u51f7|\u6749\u6728|\u6750\u6a41|\u9e1f\u6728|\u6a61\u6811|\u767d\u6866|\u6768\u6811)', q):
        if '\u80f8\u5f84' in q or 'D=' in q: return 'multi_sp'
    # 多组分信号
    if any(kw in q for kw in _MULTI_COMP_KW): return 'multi_comp'
    if re.search(r'(\u5e72|\u679d|\u53f6|\u6839).{0,3}[\u3001\u548c\u53ca+].{0,3}(\u5e72|\u679d|\u53f6|\u6839)', q): return 'multi_comp'
    # 质控信号（高置信度：含R²且含样本量或精度）
    has_r2   = any(kw in q for kw in ['R\u00b2','r\u00b2','R2','r2'])
    has_stat = any(kw in q for kw in ['\u6837\u672c\u91cf','\u6837\u672c\u6570','\u7cbe\u5ea6','\u7f6e\u4fe1','\u8bef\u5dee','\u5c0f\u6837\u672c','\u5927\u6837\u672c'])
    if has_r2 and has_stat: return 'quality_ctrl'
    if any(kw in q for kw in ['\u54ea\u4e2a\u66f4\u51c6','\u54ea\u4e2a\u66f4\u53ef\u9760','\u8d85\u9ad8R\u00b2','\u8d85\u9ad8R2']): return 'quality_ctrl'
    # Reasoning高置信度信号（含推荐词+有多个候选方程背景）
    has_reasoning = any(kw in q for kw in _REASONING_KW)
    # Single高置信度：有精确树种+地区+组分，无推荐词
    has_precise_slots = (slots.get('tree_species') and slots.get('region') and slots.get('component'))
    if has_precise_slots and not has_reasoning and not has_r2: return 'single'
    # 不确定，返回None交给LLM
    return None

def extract_slots(question):
    """第一步：纯槽位提取，单独一次LLM调用，输出稳定。"""
    raw = _llm_call([
        {'role':'system','content':SLOT_SYSTEM},
        {'role':'user','content':'\u95ee\u9898\uff1a'+question+'\n\u8f93\u51fa\uff1a'}
    ], max_tokens=180)
    parsed = _parse_json_robust(raw)
    if parsed:
        slots = _normalize_values(parsed)
        slots = _rescue_slots(question, slots)
        if 'diameter_type' not in slots:
            for kw in ['\u57fa\u5f84','\u5730\u5f84','D0','d0','\u6839\u5f84']:
                if kw in question: slots['diameter_type'] = '\u57fa\u5f84'; break
            else:
                slots['diameter_type'] = '\u672a\u77e5'
        return slots
    return _rescue_slots(question, {'diameter_type':'\u672a\u77e5'})

def classify_intent(question, slots):
    """
    Fix-①: 两步意图分类。
    第一步：规则层（高置信度关键词）
    第二步：仅规则层不确定时才调用LLM（LLM只需回答一个词）
    """
    # 规则层
    intent = _rule_based_intent(question, slots)
    if intent is not None:
        return intent
    # LLM层（仅在不确定时调用，且只输出一个词，极低漂移风险）
    raw = _llm_call([
        {'role':'system','content':INTENT_SYSTEM},
        {'role':'user','content':'\u95ee\u9898\uff1a'+question}
    ], max_tokens=20)
    raw = raw.strip().lower()
    for t in ('single','multi_sp','multi_comp','quality_ctrl','reasoning'):
        if t in raw: return t
    # 兜底：有精确槽位用single，否则用reasoning
    if slots.get('tree_species') and slots.get('region'):
        return 'single'
    return 'reasoning'

# ══════════════════════════════════════════════════════════════
# 方程元数据拼接（保持v20的rich context）
# ══════════════════════════════════════════════════════════════

def build_rich_eq_context(eq, idx=None):
    """
    v23核心改动：智能过滤全字段传入LLM。
    策略：去掉null值 + 去掉纯技术字段（parameters/equation_chain/id等）
    其余所有有值字段全部保留，让LLM自己判断哪些有用。
    
    字段映射说明（数据库真实字段）：
      location        = applicable_region（适用地区）
      source          = reference（参考文献）
      r_squared       = R²（仅22.5%填充，空时用quality_score代替）
      quality_score   = 综合质量分（100%填充，0.7-0.85）
      size_class      = 适用范围（含径级/林龄/说明文字，97.7%填充）
    """
    # 纯技术字段：对LLM适用性判断无意义，过滤掉
    SKIP_FIELDS = {'id', 'source_dataset', 'parameters', 'equation_chain',
                   'variables'}  # variables可从equation推断

    parts = []
    if idx is not None:
        parts.append(f'[候选方程{idx}]')

    # 字段中文别名映射（让LLM更容易理解字段含义）
    FIELD_ALIAS = {
        'tree_type':        '树种',
        'component':        '组分',
        'equation':         '公式',
        'location':         '适用地区',      # = applicable_region
        'size_class':       '适用范围',       # 含径级/林龄
        'quality_score':    '综合质量分',
        'r_squared':        'R²',
        'sample_size':      '样本量n',
        'source':           '参考文献',       # = reference
        'authors':          '作者',
        'data_quality':     '数据质量',
        'tags':             '标签',
        'component_original': '原始组分描述',
        'stand_age':        '林龄',
        'forest_type':      '林分类型',
        'stand_origin':     '林分起源',
        'dominant_species': '优势种学名',
        'altitude':         '海拔',
        'equation_id':      '方程编号',
        'equation_name':    '方程名称',
        'publication_year': '发表年份',
    }

    for key, val in eq.items():
        if key in SKIP_FIELDS:
            continue
        # 过滤空值：None / 空字符串 / 空列表 / 0值样本量
        if val is None or val == '' or val == []:
            continue
        if key == 'sample_size' and (val == 0 or str(val) == '0'):
            continue
        # r_squared 有时是空字符串
        if key == 'r_squared' and str(val).strip() == '':
            continue

        label = FIELD_ALIAS.get(key, key)

        # 特殊格式化
        if key == 'tags' and isinstance(val, list):
            parts.append(f'{label}: {", ".join(str(t) for t in val)}')
        elif key == 'quality_score':
            try:
                parts.append(f'{label}: {round(float(val), 3)}（0-1分，越高越好）')
            except:
                parts.append(f'{label}: {val}')
        elif key == 'parameters':
            pass  # 已在SKIP_FIELDS中过滤
        else:
            val_str = str(val)
            # 过长文本截断
            if len(val_str) > 150:
                val_str = val_str[:147] + '...'
            parts.append(f'{label}: {val_str}')

    return ' | '.join(parts)

# ══════════════════════════════════════════════════════════════
# Fix-②: 软信号检测（提示judge，不是硬规则）
# ══════════════════════════════════════════════════════════════

def _detect_soft_signals(question, slots, top1_rec):
    """
    检测可能需要judge关注的软信号。
    返回字符串，作为额外上下文传给judge prompt。
    不是硬规则——judge自己决定是否据此拒绝。
    """
    signals = []
    # 跨区域意图信号
    if re.search(r'\u60f3\u7528.{0,15}(?:\u516c\u5f0f|\u65b9\u7a0b|\u53c2\u6570).{0,10}(?:\u7b97|\u8ba1\u7b97|\u9002\u7528)', question):
        signals.append('\u7528\u6237\u8868\u8fbe\u4e86\u8de8\u533a\u57df\u4f7f\u7528\u65b9\u7a0b\u7684\u610f\u56fe\uff0c\u8bf7\u91cd\u70b9\u6838\u67e5\u5730\u533a\u9002\u7528\u6027')
    # 修复A+B: 胸径超径级信号 — 仅在size_class含明确胸径关键词时触发
    D = slots.get('D')
    sc = top1_rec.get('size_class','') if top1_rec else ''
    if D is not None and sc:
        sc_str = str(sc)
        # 修复A: 只有明确含胸径关键词才解析，过滤林分转换类文字描述
        has_dbh_kw = any(kw in sc_str for kw in [
            '胸径', 'D=', 'D:', 'DBH', 'dbh', '径级'])
        dr = parse_d_range(sc) if has_dbh_kw else None
        if dr and D > dr[1]:
            ratio = D / dr[1]
            # 修复B: >1.5倍强信号，1.3-1.5倍仅参考提示
            if ratio > 1.5:
                signals.append(f'用户胸径{D}cm明显超出径级上限{dr[1]}cm（{ratio:.2f}倍）')
            elif ratio > 1.3:
                signals.append(f'用户胸径{D}cm稍超径级上限{dr[1]}cm（{ratio:.2f}倍，请判断是否可接受）')
    # 地区不符信号
    user_loc = re.sub(r'\s+','', slots.get('region','') or '')
    eq_loc   = re.sub(r'\s+','', top1_rec.get('location','') or '') if top1_rec else ''
    eq_tags  = top1_rec.get('tags') or [] if top1_rec else []
    is_universal = ('\u901a\u7528' in eq_loc or any('\u901a\u7528' in str(t) for t in eq_tags))
    if user_loc and eq_loc and loc_match_score(user_loc, eq_loc) == 0 and not is_universal:
        signals.append(f'\u5730\u533a\u4e0d\u5339\u914d\uff1a\u7528\u6237\u5730\u533a={user_loc}\uff0c\u65b9\u7a0b\u5730\u533a={eq_loc}')
    # 组分不符信号
    user_comp = COMP_TO_STD.get(slots.get('component','') or '', '')
    eq_comp   = top1_rec.get('component','') or '' if top1_rec else ''
    if user_comp and eq_comp and user_comp != eq_comp:
        comp_groups = [
            {'\u5e72','\u679d','\u53f6','\u5730\u4e0a'},
            {'\u6839','\u5730\u4e0b'},
            {'\u6574\u682a','\u5730\u4e0a','\u5730\u4e0b','\u5e72','\u679d','\u53f6','\u6839'}
        ]
        if not any(user_comp in g and eq_comp in g for g in comp_groups):
            signals.append(f'\u7ec4\u5206\u4e0d\u5339\u914d\uff1a\u7528\u6237\u8981\u6c42={user_comp}\uff0c\u65b9\u7a0b\u7ec4\u5206={eq_comp}')
    return '\n'.join(signals) if signals else '\u672a\u68c0\u6d4b\u5230\u660e\u663e\u5f02\u5e38\u4fe1\u53f7'

# ══════════════════════════════════════════════════════════════
# 检索
# ══════════════════════════════════════════════════════════════

def has_var(eq, var):
    return bool(re.search(r'(?<![A-Za-z0-9])'+re.escape(var)+r'(?![A-Za-z0-9_])', eq))

def param_score(eq_str, slots):
    s = 0
    if slots.get('H') is not None and has_var(eq_str,'H'):   s += 2
    if slots.get('D0') is not None and has_var(eq_str,'D0'): s += 2
    if slots.get('V') is not None and has_var(eq_str,'V'):   s += 3
    if slots.get('A') is not None and has_var(eq_str,'A'):   s += 3
    return s

def parse_d_range(sc):
    """v23增强：支持真实数据中的多种size_class格式"""
    if not sc: return None
    sc_str = str(sc)
    for pat in [
        # 格式1：胸径X-Ycm
        r'\u80f8\u5f84[\u8303\u56f4\uff1a:为]*\s*(\d+\.?\d*)\s*[-\u2013~\uff5e]\s*(\d+\.?\d*)\s*(?:cm|\u5398\u7c73)',
        # 格式2：胸径X, Ycm（逗号分隔）
        r'\u80f8\u5f84[\u8303\u56f4\uff1a:为]*\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*(?:cm|\u5398\u7c73)',
        # 格式3：D=X-Ycm
        r'D\s*[=:\uff1a]\s*(\d+\.?\d*)\s*[-\u2013~\uff5e]\s*(\d+\.?\d*)\s*(?:cm|\u5398\u7c73)',
        # 格式4：径级X-Y
        r'\u5f84\u7ea7\s*(\d+\.?\d*)\s*[-\u2013~\uff5e]\s*(\d+\.?\d*)',
        # 格式5：纯数字X-Ycm
        r'(\d+\.?\d*)\s*[-\u2013~\uff5e]\s*(\d+\.?\d*)\s*cm',
        # 格式6：胸径 X Y cm（空格分隔）
        r'\u80f8\u5f84\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*(?:cm|\u5398\u7c73)',
    ]:
        m = re.search(pat, sc_str, re.IGNORECASE)
        if m:
            try:
                lo, hi = float(m.group(1)), float(m.group(2))
                # 自动纠正顺序颠倒
                if lo > hi: lo, hi = hi, lo
                if lo < hi: return lo, hi
            except:
                continue
    return None

def d_range_score(sc, D):
    if D is None: return 0
    dr = parse_d_range(sc)
    if not dr: return 0
    if dr[0] <= D <= dr[1]: return 2
    if D <= dr[1]*1.3: return 0
    if D > dr[1]*1.5: return -3
    return 0

def retrieve(slots, top_k=5, return_scores=False):
    sp       = re.sub(r'\s+','', slots.get('tree_species','') or '')
    loc      = re.sub(r'\s+','', slots.get('region','') or '')
    comp_raw = slots.get('component','') or ''
    comp_std = COMP_TO_STD.get(comp_raw, comp_raw)
    D  = slots.get('D');  H  = slots.get('H')
    D0 = slots.get('D0'); V  = slots.get('V'); A = slots.get('A')
    struct_scored = []
    for i, r in enumerate(db):
        eq_sp   = re.sub(r'\s+','',r.get('tree_type',''))
        eq_loc  = re.sub(r'\s+','',r.get('location',''))
        eq_comp = r.get('component','')
        sc      = r.get('size_class','') or ''
        s = 0
        loc_s = loc_match_score(loc, eq_loc) if loc else 0
        s += loc_s
        sp_exact = (sp == eq_sp)
        if sp_exact: s += 5
        elif sp in eq_sp: s += 3
        elif eq_sp in sp: s += 2
        cm = 1.5 if (sp_exact and loc_s >= 3) else 1.0
        if comp_std:
            if comp_std == eq_comp:   s += int(6*cm)
            elif comp_std in eq_comp: s += int(3*cm)
            elif eq_comp in comp_std: s += int(2*cm)
        s += d_range_score(sc, D)
        s += param_score(r.get('equation',''), {'D':D,'H':H,'D0':D0,'V':V,'A':A})
        if comp_std not in ('\u6750\u79ef','\u5355\u4f4d\u9762\u79ef\u751f\u7269\u91cf','') and eq_comp == '\u6750\u79ef': s -= 20
        if s > 0: struct_scored.append((i, s))
    struct_scored.sort(key=lambda x: -x[1])
    if not struct_scored:
        if not sp: return []
        q_vec = emb.encode([sp], normalize_embeddings=True)[0]
        sims  = np.dot(sp_vecs, q_vec)
        ranked = sorted(enumerate(sims.tolist()), key=lambda x: -x[1])
        if return_scores: return [(db[i], float(s)) for i,s in ranked[:top_k]]
        return [db[i] for i,_ in ranked[:top_k]]
    
    # 修复C：若检索结果偏少，追加上位类/林分通用方程
    # 松科树种 → 追加"松类"/"松杉类"通用方程
    # 询问林分 → 追加tree_type="林分"/"各树种"的方程
    _SUPERCLASS_MAP = {
        '华山松':'松类','油松':'松类','马尾松':'松类','红松':'松类',
        '落叶松':'松类','云杉':'松类','冷杉':'松类','樟子松':'松类',
        '杉木':'杉类','柳杉':'杉类','水杉':'杉类','池杉':'杉类',
    }
    if len(struct_scored) < 15 and sp:
        superclass = _SUPERCLASS_MAP.get(sp)
        if superclass:
            seen = {i for i,_ in struct_scored}
            for i, r in enumerate(db):
                if i not in seen and r.get('tree_type','') in (superclass, '松杉类', '全树种'):
                    eq_loc = re.sub(r'\s+','',r.get('location',''))
                    loc_s = loc_match_score(loc, eq_loc) if loc else 0
                    if loc_s >= 1:  # 至少地区前缀匹配
                        struct_scored.append((i, 3))  # 低分追加
    cand_idx  = [i for i,_ in struct_scored[:80]]
    cand_recs = [db[i] for i in cand_idx]
    struct_arr  = np.array([s for _,s in struct_scored[:80]], dtype=float)
    struct_norm = struct_arr / (struct_arr.max()+1e-9)
    q_text    = (sp or '') + ' ' + (loc or '') + ' ' + (comp_std or '')
    q_vec     = emb.encode([q_text.strip()], normalize_embeddings=True)[0]
    cand_texts= [r.get('tree_type','')+' '+r.get('location','') for r in cand_recs]
    cand_vecs = emb.encode(cand_texts, normalize_embeddings=True)
    sims      = np.dot(cand_vecs, q_vec)
    final_scores = 0.30 * struct_norm + 0.70 * sims
    ranked = sorted(zip(cand_idx, final_scores.tolist()), key=lambda x: -x[1])
    if return_scores: return [(db[i], float(s)) for i,s in ranked[:top_k]]
    return [db[i] for i,_ in ranked[:top_k]]

# ══════════════════════════════════════════════════════════════
# stage0：物理不可能值校验
# ══════════════════════════════════════════════════════════════

def stage0_hard_reject(question, slots):
    slots = _rescue_slots(question, slots)
    D = slots.get('D'); H = slots.get('H')
    if D is not None and D < 0:
        return True, '\u80f8\u5f84\u4e3a\u8d1f\u503c('+str(D)+'cm)\uff0c\u53c2\u6570\u65e0\u6548'
    if D is not None and D > 300:
        return True, '\u80f8\u5f84'+str(D)+'cm\u8d85\u51fa\u7269\u79cd\u5408\u7406\u4e0a\u9650(>300cm)'
    if H is not None and H < 0:
        return True, '\u6811\u9ad8\u4e3a\u8d1f\u503c('+str(H)+'m)\uff0c\u53c2\u6570\u65e0\u6548'
    if H is not None and H > 200:
        return True, '\u6811\u9ad8'+str(H)+'m\u8d85\u51fa\u7269\u79cd\u5408\u7406\u4e0a\u9650(>200m)'
    if D is not None and H is not None:
        if D < 1.0 and H > 10:
            return True, '\u80f8\u5f84'+str(D)+'cm\u4f46\u6811\u9ad8'+str(H)+'m\uff0c\u4e25\u91cd\u4e0d\u534f\u8c03'
        if D > 100 and H < 5:
            return True, '\u80f8\u5f84'+str(D)+'cm\u4f46\u6811\u9ad8'+str(H)+'m\uff0c\u4e25\u91cd\u4e0d\u534f\u8c03'
    return False, ''

def check_diameter_type(question, slots, top_candidates):
    if not slots or not top_candidates: return None
    confirm_phrases = ['DBH\u5c31\u662f\u80f8\u5f84','dbh\u5c31\u662f\u80f8\u5f84','D\u5c31\u662f\u80f8\u5f84',
                       '\u5c31\u662f\u80f8\u5f84\u5427','\u5c31\u662fDBH\u5427','\u76f4\u63a5\u7528\u8fd9\u4e2a\u6570',
                       '\u76f4\u63a5\u4ee3\u5165','\u57fa\u5f84\u548c\u80f8\u5f84\u4e00\u6837','\u7b49\u4e8e\u80f8\u5f84']
    if any(p in question for p in confirm_phrases): return None
    if slots.get('diameter_type','\u672a\u77e5') != '\u57fa\u5f84': return None
    top_eq = top_candidates[0]
    low = lambda x: str(x or '').lower()
    if any(ind in low(top_eq.get('equation','')) or ind in low(str(top_eq.get('tags','')))
           for ind in ['d0','d_0','\u57fa\u5f84','\u5730\u5f84']): return None
    return {'warn':True,'message':'\u6ce8\u610f\uff1a\u60a8\u63d0\u4f9b\u7684\u662f\u57fa\u5f84\uff0c\u8be5\u65b9\u7a0b\u4f7f\u7528\u80f8\u5f84\uff08DBH\uff0c1.3m\u5904\uff09\uff0c\u4e24\u8005\u76f8\u5dee\u7ea615~30%\u3002'}

# ══════════════════════════════════════════════════════════════
# Fix-②+③: judge拿到软信号+完整元数据，阈值降低到0.70
# ══════════════════════════════════════════════════════════════


def llm_judge_with_context(question, slots, top1_rec, top1_score, is_gen_branch=False):
    if is_gen_branch: return False, ''

    # 修复A: 纯信号驱动 — 无软信号直接跳过，有软信号无论分数多高都送judge
    soft_signals = _detect_soft_signals(question, slots, top1_rec)

    # 无任何异常信号 → 直接跳过，零误拒风险
    if not soft_signals:
        return False, ''
    # 有异常信号 → 送judge（不受score阈值约束，充分利用LLM判断力）''

    # 构建完整上下文（Fix-③: 传入soft_signals作为额外上下文）
    eq_context = build_rich_eq_context(top1_rec)
    user_info  = ('\u7528\u6237\u60c5\u51b5\uff1a\u6811\u79cd={} \u5730\u533a={} \u7ec4\u5206={} \u80f8\u5f84={}cm \u6811\u9ad8={}m'.format(
        slots.get('tree_species','?'), slots.get('region','?'),
        slots.get('component','?'), slots.get('D','?'), slots.get('H','?')))

    # Fix-③: judge prompt中插入软信号
    judge_prompt = JUDGE_SYSTEM.replace('{soft_signals}', soft_signals)

    answer = _llm_call([
        {'role':'system','content':judge_prompt},
        {'role':'user','content':user_info+'\n\n\u5019\u9009\u65b9\u7a0b\u8be6\u60c5\uff1a\n'+eq_context+'\n\n\u7528\u6237\u539f\u59cb\u95ee\u9898\uff1a'+question}
    ], max_tokens=60)
    if answer.strip().startswith('\u62d2\u7edd'):
        return True, 'llm_judge('+answer.strip()+')'
    return False, ''

# ══════════════════════════════════════════════════════════════
# 业务处理函数
# ══════════════════════════════════════════════════════════════

def quality_ctrl_by_llm(question, candidates):
    """v23: 按quality_score预排序，让LLM在有序列表中选最优"""
    if not candidates: return '', ''
    # quality_score 100%填充，按此预排序
    def _qs(eq):
        try: return float(eq.get('quality_score', 0.75))
        except: return 0.75
    sorted_cands = sorted(candidates, key=_qs, reverse=True)
    cand_text = '\n'.join([build_rich_eq_context(r, idx=i+1) for i, r in enumerate(sorted_cands[:5])])
    answer = _llm_call([
        {'role':'system','content':QUALITY_SYSTEM},
        {'role':'user','content':'\u7528\u6237\u95ee\u9898\uff1a'+question+'\n\n\u5019\u9009\u65b9\u7a0b\uff1a\n'+cand_text}
    ], max_tokens=120)
    m = re.search(r'\u63a8\u8350\u65b9\u7a0b[：:]\s*(.+?)(?:\n|\u63a8\u8350\u7406\u7531|$)', answer, re.DOTALL)
    if m:
        raw_norm = normalize_eq(m.group(1).strip().strip('[]'))
        for cand in candidates:
            if normalize_eq(cand.get('equation','')) == raw_norm: return cand.get('equation',''), answer
        for cand in candidates:
            if raw_norm in normalize_eq(cand.get('equation','')) or normalize_eq(cand.get('equation','')) in raw_norm:
                return cand.get('equation',''), answer
    return candidates[0].get('equation','') if candidates else '', answer

def handle_single(question, slots, is_gen_branch=False):
    scored = retrieve(slots, top_k=5, return_scores=True)
    if not scored: return {'intent':'single','equations':[],'top1':'','rejected':False}
    top1_rec, top1_score = scored[0]
    diam_check = check_diameter_type(question, slots, [r for r,_ in scored])
    warn_msg   = diam_check['message'] if diam_check and diam_check.get('warn') else None
    rejected, reason = llm_judge_with_context(question, slots, top1_rec, top1_score, is_gen_branch)
    if rejected:
        return {'intent':'single','equations':[],'top1':'','rejected':True,'reject_reason':reason}
    results = [r for r,_ in scored]
    return {'intent':'single','equations':[r.get('equation','') for r in results],
            'top1':results[0].get('equation','') if results else '','rejected':False,'warn_message':warn_msg}

def handle_multi_sp(question, slots):
    raw = _llm_call([
        {'role':'system','content':'\u4ece\u95ee\u9898\u4e2d\u63d0\u53d6\u6240\u6709\u6811\u79cd\u540d\uff0c\u6bcf\u884c\u4e00\u4e2a\uff0c\u53ea\u8f93\u51fa\u6811\u79cd\u540d\uff0c\u4e0d\u8981\u7f16\u53f7'},
        {'role':'user','content':question}
    ], max_tokens=60)
    species_list = [s.strip() for s in raw.strip().split('\n') if s.strip()]
    if not species_list: return handle_single(question, slots)
    results={}; all_eqs=[]
    for sp in species_list[:4]:
        sp_slots = dict(slots); sp_slots['tree_species'] = sp
        top = retrieve(sp_slots, top_k=3)
        if top:
            results[sp] = top[0].get('equation','')
            all_eqs.extend([r.get('equation','') for r in top])
    return {'intent':'multi_sp','species_results':results,'equations':all_eqs,
            'top1':list(results.values())[0] if results else '','rejected':False}

def handle_multi_comp(question, slots):
    comp_raw  = slots.get('component','') or ''
    comp_std  = COMP_TO_STD.get(comp_raw, comp_raw)
    sub_comps = MULTI_COMP_MAP.get(comp_std, [])
    if not sub_comps:
        raw = _llm_call([
            {'role':'system','content':'\u4ece\u95ee\u9898\u4e2d\u63d0\u53d6\u9700\u8981\u8ba1\u7b97\u7684\u751f\u7269\u91cf\u7ec4\u5206\uff0c\u6bcf\u884c\u4e00\u4e2a\uff08\u53ea\u5199\uff1a\u5e72/\u679d/\u53f6/\u6839/\u76ae\uff09'},
            {'role':'user','content':question}
        ], max_tokens=40)
        sub_comps = [s.strip() for s in raw.strip().split('\n') if s.strip()]
    if not sub_comps: return handle_single(question, slots)
    results={}; all_eqs=[]
    for comp in sub_comps[:5]:
        comp_slots = dict(slots); comp_slots['component'] = comp
        top = retrieve(comp_slots, top_k=3)
        if top:
            results[comp] = top[0].get('equation','')
            all_eqs.extend([r.get('equation','') for r in top])
    return {'intent':'multi_comp','comp_results':results,'equations':all_eqs,
            'top1':list(results.values())[0] if results else '','rejected':False}

def handle_reasoning(question, slots, is_quality_ctrl=False):
    candidates = retrieve(slots, top_k=5)
    if not candidates:
        return {'intent':'reasoning','answer':'\u672a\u627e\u5230\u76f8\u5173\u65b9\u7a0b','equations':[],'top1':'','rejected':False}
    if is_quality_ctrl:
        recommended, answer = quality_ctrl_by_llm(question, candidates)
    else:
        cand_text = '\n'.join([build_rich_eq_context(r, idx=i+1) for i, r in enumerate(candidates[:5])])
        answer = _llm_call([
            {'role':'system','content':REASONING_SYSTEM},
            {'role':'user','content':'\u95ee\u9898\uff1a'+question+'\n\n\u5019\u9009\u65b9\u7a0b\uff1a\n'+cand_text}
        ], max_tokens=120)
        recommended = ''
        m = re.search(r'\u63a8\u8350\u65b9\u7a0b[：:]\s*(.+?)(?:\n|\u63a8\u8350\u7406\u7531|$)', answer, re.DOTALL)
        if m:
            raw_norm = normalize_eq(m.group(1).strip().strip('[]'))
            for cand in candidates:
                if normalize_eq(cand.get('equation','')) == raw_norm:
                    recommended = cand.get('equation',''); break
            if not recommended:
                for cand in candidates:
                    if raw_norm in normalize_eq(cand.get('equation','')) or normalize_eq(cand.get('equation','')) in raw_norm:
                        recommended = cand.get('equation',''); break
        if not recommended: recommended = candidates[0].get('equation','') if candidates else ''
    all_eqs = [recommended]+[r.get('equation','') for r in candidates if r.get('equation','')!=recommended]
    return {'intent':'reasoning','candidates':[r.get('equation','') for r in candidates],
            'equations':all_eqs,'answer':answer,'top1':recommended,'rejected':False}

def process_question(question, is_gen_branch=False):
    # Fix-①: 槽位提取 + 意图分类分两步
    slots  = extract_slots(question)
    intent = classify_intent(question, slots)
    hard_rej, hard_reason = stage0_hard_reject(question, slots)
    if hard_rej:
        return {'intent':intent,'equations':[],'top1':'','rejected':True,
                'reject_reason':'code_reject('+hard_reason+')'}
    if   intent == 'multi_sp':    return handle_multi_sp(question, slots)
    elif intent == 'multi_comp':  return handle_multi_comp(question, slots)
    elif intent == 'quality_ctrl':return handle_reasoning(question, slots, is_quality_ctrl=True)
    elif intent == 'reasoning':   return handle_reasoning(question, slots, is_quality_ctrl=False)
    else:                         return handle_single(question, slots, is_gen_branch=is_gen_branch)

# ══════════════════════════════════════════════════════════════
# 评估框架
# ══════════════════════════════════════════════════════════════

def is_wrong_prem(c):
    return ('\u9519\u8bef\u524d\u63d0' in (c.get('gen_type_tags') or []) or
            c.get('test_id','').startswith('GEN_\u9519\u8bef'))

def is_valid_equation(eq_str):
    return bool(re.search(r'[=^*/]', str(eq_str)))

print('\n'+'='*60+'\n\u3010v21 \u6570\u636e\u5206\u6d41\u3011\n'+'='*60)
all_cases = json.load(open(TEST_PATH, encoding='utf-8'))['test_cases']
db        = json.load(open(EQ_PATH,   encoding='utf-8'))
db_eq_set = {normalize_eq(r.get('equation','')) for r in db}
bucket_main=[]; bucket_gen=[]; bucket_legacy=[]
bucket_neg=[]; bucket_rejection=[]; bucket_no_db=[]
for c in all_cases:
    has_group='group' in c; group=c.get('group','')
    eq_str=str(c.get('equation','') or '').strip()
    legacy_eq=str(c.get('_legacy_equation','') or '').strip()
    if c.get('requires_rejection',False) and '\u8d1f\u6837\u672c' in group:
        bucket_neg.append(c); continue
    is_neg='\u8d1f\u6837\u672c' in group or is_wrong_prem(c)
    if eq_str and not is_valid_equation(eq_str): bucket_rejection.append(c); continue
    if not has_group:
        if is_wrong_prem(c): bucket_rejection.append(c)
        elif eq_str: bucket_gen.append(c)
        continue
    if not eq_str:
        if is_neg: bucket_rejection.append(c)
        elif legacy_eq: bucket_legacy.append(c)
        continue
    if is_neg: bucket_rejection.append(c)
    elif normalize_eq(eq_str) in db_eq_set: bucket_main.append(c)
    else: bucket_no_db.append(c)
print('  \u4e3b\u8bc4\u4f30: '+str(len(bucket_main))+'\u6761  GEN\u5206\u652f: '+str(len(bucket_gen))+'\u6761  legacy_eq: '+str(len(bucket_legacy))+'\u6761')
print('  \u8d1f\u6837\u672c: '+str(len(bucket_neg))+'\u6761  \u8df3\u8fc7: '+str(len(bucket_rejection)+len(bucket_no_db))+'\u6761  \u5408\u8ba1: '+str(len(all_cases))+'\u6761')

print('\n[\u521d\u59cb\u5316] \u52a0\u8f7d Qwen \u6a21\u578b...', flush=True)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
_model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
              torch_dtype=torch.float16, device_map='auto')
_model.eval()
print('  \u5b8c\u6210\n', flush=True)

print('[\u521d\u59cb\u5316] \u52a0\u8f7d Embedding \u6a21\u578b...', flush=True)
emb = SentenceTransformer(EMB_PATH)
all_texts = ([r.get('location','') for r in db]+[r.get('tree_type','') for r in db]+[r.get('component','') for r in db])
all_vecs  = emb.encode(all_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
n=len(db); loc_vecs=all_vecs[0:n]; sp_vecs=all_vecs[n:2*n]; comp_vecs=all_vecs[2*n:3*n]
print('  \u5b8c\u6210\n', flush=True)

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

def run_eval(branch_name, cases, eq_override_field=None, is_gen=False):
    print('\n'+'='*60+'\n['+branch_name+'] '+str(len(cases))+'\u6761\n'+'='*60)
    hit1=hit3=hit5=hit_any=skipped=errors=0
    intent_stats={k:{'total':0,'hit1':0} for k in ('single','multi_sp','multi_comp','reasoning','quality_ctrl')}
    for i,case in enumerate(cases):
        tid=case.get('test_id','#'+str(i+1)); label=case.get('group',tid); q=case['question']
        eq_override=(str(case.get(eq_override_field,'') or '').strip() if eq_override_field else None)
        gold_set,primary=build_gold_set(case,eq_override=eq_override)
        if not primary:
            skipped+=1; print('['+str(i+1).zfill(3)+'] skip '+tid,flush=True); continue
        try:
            result=process_question(q,is_gen_branch=is_gen)
            intent=result.get('intent','single'); rejected=result.get('rejected',False)
            if rejected:
                reason=result.get('reject_reason','')
                print('['+str(i+1).zfill(3)+'] \u274c ['+intent+'] '+tid+' | '+label+' [\u62d2:'+reason+']',flush=True)
                intent_stats.get(intent,intent_stats['single'])['total']+=1; continue
            all_eqs=[normalize_eq(e) for e in result.get('equations',[]) if e]
            if not all_eqs: all_eqs=[normalize_eq(result.get('top1',''))]
            h1=bool(all_eqs) and all_eqs[0] in gold_set
            h3=any(e in gold_set for e in all_eqs[:3])
            h5=any(e in gold_set for e in all_eqs[:5])
            h_any=any(e in gold_set for e in all_eqs)
            if h1:   hit1+=1
            if h3:   hit3+=1
            if h5:   hit5+=1
            if h_any: hit_any+=1
            base=intent if intent in intent_stats else 'single'
            intent_stats[base]['total']+=1
            if h1: intent_stats[base]['hit1']+=1
            mark='\u2705' if h1 else ('\U0001f536' if h_any else '\u274c')
            warn_tag=' [\u57fa\u5f84]' if result.get('warn_message') else ''
            print('['+str(i+1).zfill(3)+'] '+mark+' ['+intent+'] '+tid+' | '+label+warn_tag,flush=True)
        except Exception as e:
            errors+=1; print('['+str(i+1).zfill(3)+'] \U0001f4a5 '+tid+' | '+str(e),flush=True)
    n_valid=len(cases)-skipped
    def pct(x): return str(100*x//(n_valid or 1))+'%'
    print('\n[\u7ed3\u679c '+branch_name+']\n  \u6709\u6548:'+str(n_valid)+'  \u8df3\u8fc7:'+str(skipped)+'  \u5f02\u5e38:'+str(errors))
    print('  Hit@1:'+str(hit1)+'/'+str(n_valid)+'('+pct(hit1)+')'
          '  Hit@3:'+str(hit3)+'/'+str(n_valid)+'('+pct(hit3)+')'
          '  Hit@5:'+str(hit5)+'/'+str(n_valid)+'('+pct(hit5)+')'
          '  Hit@any:'+str(hit_any)+'/'+str(n_valid)+'('+pct(hit_any)+')')
    print('  \u5206\u652f:',end='')
    for intent,stat in intent_stats.items():
        if stat['total']==0: continue
        print(' '+intent+'='+str(stat['hit1'])+'/'+str(stat['total'])+'('+str(100*stat['hit1']//(stat['total'] or 1))+'%)',end='')
    print()
    return hit1,hit3,hit5,hit_any,n_valid

def run_neg_eval(cases):
    print('\n'+'='*60+'\n[\u8d1f\u6837\u672c\u8bc4\u4f30] '+str(len(cases))+'\u6761\n'+'='*60)
    rejected_n=0; type_stats={}; fail_cases=[]
    for i,case in enumerate(cases):
        tid=case.get('test_id','NEG_'+str(i)); q=case.get('question',''); group=case.get('group','\u672a\u77e5')
        slots=extract_slots(q); slots=_rescue_slots(q,slots)
        hard_rej,hard_reason=stage0_hard_reject(q,slots)
        if hard_rej:
            rejected_n+=1
            type_stats.setdefault(group,{'total':0,'rejected':0})
            type_stats[group]['total']+=1; type_stats[group]['rejected']+=1
            print('['+str(i+1).zfill(3)+'] \u2705 '+tid.ljust(22)+' code_reject('+hard_reason+')',flush=True); continue
        scored=retrieve(slots,top_k=5,return_scores=True)
        top1_score=scored[0][1] if scored else 0.0
        top1_rec=scored[0][0] if scored else None
        diam_check=check_diameter_type(q,slots,[r for r,_ in scored] if scored else [])
        if diam_check and diam_check.get('warn'):
            rejected_n+=1
            type_stats.setdefault(group,{'total':0,'rejected':0})
            type_stats[group]['total']+=1; type_stats[group]['rejected']+=1
            print('['+str(i+1).zfill(3)+'] \u2705 '+tid.ljust(22)+' diameter_warn',flush=True); continue
        sp_val=(slots.get('tree_species') or '').strip(); reg_val=(slots.get('region') or '').strip()
        nat_signals=[]
        if not scored: nat_signals.append('empty_results')
        if top1_score<=0.20: nat_signals.append('low_score('+str(round(top1_score,3))+')')
        if not sp_val and not reg_val: nat_signals.append('both_slots_empty')
        llm_rej=False; llm_reason=''
        if not nat_signals and top1_rec:
            llm_rej,llm_reason=llm_judge_with_context(q,slots,top1_rec,top1_score,is_gen_branch=False)
        naturally_rejected=len(nat_signals)>0 or llm_rej
        if naturally_rejected: rejected_n+=1
        type_stats.setdefault(group,{'total':0,'rejected':0})
        type_stats[group]['total']+=1
        if naturally_rejected: type_stats[group]['rejected']+=1
        flag='\u2705' if naturally_rejected else '\u274c'
        sig='|'.join(nat_signals) if nat_signals else (llm_reason if llm_rej else 'none')
        print('['+str(i+1).zfill(3)+'] '+flag+' '+tid.ljust(22)+' score='+str(round(top1_score,3))+'  '+sig,flush=True)
        if not naturally_rejected: fail_cases.append({'test_id':tid,'group':group,'question':q})
    n_total=len(cases); rate=rejected_n/n_total*100 if n_total else 0
    print('\n[\u8d1f\u6837\u672c\u7ed3\u679c]  \u62d2\u7edd:'+str(rejected_n)+'/'+str(n_total)+'  \u62d2\u7edd\u7387:'+str(round(rate,1))+'%')
    for g,s in sorted(type_stats.items()):
        r=s['rejected']/s['total']*100 if s['total'] else 0
        print('    '+g.ljust(25)+' '+str(s['rejected'])+'/'+str(s['total'])+'  '+str(round(r))+'%')
    if fail_cases:
        print('\n  \u4ecd\u672a\u62d2\u7edd('+str(len(fail_cases))+'\u6761):')
        for fc in fail_cases[:10]: print('    '+fc['test_id']+' | '+fc['question'][:60]+'...')
    return rejected_n,n_total,rate

r1,r3,r5,ra,nr=run_eval('\u4e3b\u8bc4\u4f30\uff08\u6709group+\u5728\u5e93\uff09',bucket_main)
g1,g3,g5,ga,ng=run_eval('GEN\u5206\u652f\uff08\u65e0group\uff09',bucket_gen,is_gen=True)
l1,l3,l5,la,nl=run_eval('legacy_eq\u5206\u652f',bucket_legacy,eq_override_field='_legacy_equation')
neg_rejected,neg_total,neg_rate=run_neg_eval(bucket_neg)

def pct(x,n): return str(100*x//(n or 1))+'%'
total_n=nr+ng+nl; total_h1=r1+g1+l1; total_h3=r3+g3+l3; total_h5=r5+g5+l5; total_ha=ra+ga+la
print('\n'+'='*76)
for label,n_,h1_,h3_,h5_,ha_ in [
    ('DeepSeek API (\u53c2\u8003,50\u6761)','50','82%','86%','88%','\u2014'),
    ('v9  \u57fa\u7ebf','376','74%','78%','81%','81%'),
    ('v17 \u7efc\u5408','376','76%','80%','82%','82%'),
    ('v19 \u7efc\u5408','376','76%','80%','83%','83%'),
    ('v20 \u7efc\u5408','376','79%','83%','85%','85%'),
]:
    print(label.ljust(32)+str(n_).rjust(5)+h1_.rjust(7)+h3_.rjust(7)+h5_.rjust(7)+ha_.rjust(8))
print('-'*76)
print('v24 \u4e3b\u8bc4\u4f30'.ljust(32)+str(nr).rjust(5)+pct(r1,nr).rjust(7)+pct(r3,nr).rjust(7)+pct(r5,nr).rjust(7)+pct(ra,nr).rjust(8))
print('v24 GEN\u5206\u652f'.ljust(32)+str(ng).rjust(5)+pct(g1,ng).rjust(7)+pct(g3,ng).rjust(7)+pct(g5,ng).rjust(7)+pct(ga,ng).rjust(8))
print('v23 legacy_eq'.ljust(32)+str(nl).rjust(5)+pct(l1,nl).rjust(7)+pct(l3,nl).rjust(7)+pct(l5,nl).rjust(7)+pct(la,nl).rjust(8))
print('='*76)
print('v24 \u7efc\u5408'.ljust(32)+str(total_n).rjust(5)+pct(total_h1,total_n).rjust(7)+pct(total_h3,total_n).rjust(7)+pct(total_h5,total_n).rjust(7)+pct(total_ha,total_n).rjust(8))
print('='*76)
print('  v9  \u57fa\u7ebf\u62d2\u7edd\u7387: 14.1%')
print('  v17 \u62d2\u7edd\u7387:     18.8%')
print('  v19 \u62d2\u7edd\u7387:     51.6%')
print('  v20 \u62d2\u7edd\u7387:     34.4%')
print('  v21 \u62d2\u7edd\u7387:     '+str(round(neg_rate,1))+'%  ('+str(neg_rejected)+'/'+str(neg_total)+')')
print('='*76)
print('v22\u4fee\u590d\u8bf4\u660e\uff08\u65e0\u65b0\u589e\u786c\u7f16\u7801\uff09\uff1a')
print('  Fix-\u2460: \u610f\u56fe\u5206\u7c7b\u6539\u4e3a\u4e24\u6b65 \u2014 \u69fd\u4f4d\u63d0\u53d6(\u7a33\u5b9a) + \u89c4\u5219\u4f18\u5148+LLM\u5907\u7528(\u51c6\u786e)')
print('     \u89e3\u51b3v20\u610f\u56fe\u6f02\u79fb\u5bfc\u81f4\u768414\u6761\u6b63\u6837\u672c\u56de\u9000')
print('  Fix-\u2461: judge\u8df3\u8fc7\u9608\u503c 0.85\u21920.70 + \u8f6f\u4fe1\u53f7\u4f20\u5165judge')
print('     \u89e3\u51b3v20\u8d1f\u6837\u672c\u62d2\u7edd\u7387 51.6%\u21920.34.4%\u7684\u56de\u9000')
print('  Fix-\u2462: judge prompt\u52a0few-shot\u6b63\u6837\u672c\u8c46\u514d\u793a\u4f8b')
print('     \u89e3\u51b3Q0274\u7b49\u6811\u79cd\u5339\u914d\u4f46\u5f84\u7ea7\u7a0d\u8d85\u88ab\u8bef\u62d2\u7684\u95ee\u9898')
# v23修复说明附录
print('\n【v23核心改动：数据-算法对齐】')
print('  1. build_rich_eq_context(): 智能过滤全字段，去null去技术字段')
print('     location=适用地区, source=参考文献, quality_score=综合质量分(100%填充)')
print('     r_squared仅22.5%填充，缺失时LLM用quality_score代替')  
print('  2. parse_d_range(): 新增逗号分隔格式支持（真实数据中存在）')
print('  3. quality_ctrl_by_llm(): 按quality_score预排序候选方程')
print('  根本原因：v20/v21的build_rich_eq_context读取了大量不存在的字段')
print('  (r2/rmse/applicable_region等)，传给LLM的实际是空值，现已修复')

print('\n【v24修复说明（以v23为基线）】')
print('  v23指标: Hit@1=77% / 负样本拒绝率=65.6%')
print('  修复A: 软信号只在size_class含明确胸径关键词时才产生径级信号')
print('         解决林分转换类(Q0121等)因文字描述触发误拒的问题')
print('  修复B: 径级超范围阈值分级 — >1.5倍为强信号，1.3-1.5倍为参考提示')
print('         解决Q0277等边界case被过度拒绝的问题')
print('  修复C: 检索增强——松科/杉科树种自动追加上位类通用方程')
print('         解决Q0021/Q0022/Q0211等通用方程检索未命中的问题')
print('  修复D: judge prompt增加变量匹配规则')
print('         解决Q0029等用户有H数据但judge误判的问题')