# Exact extracted historical function.
# Source: <PRIVATE_ROOT>/homee/abl_full.py
# Original source file was NOT modified.

import re

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
