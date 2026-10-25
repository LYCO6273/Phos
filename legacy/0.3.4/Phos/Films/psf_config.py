# 赛博请神
# psf_config.py —— 邻域效应运行时选项
# 三档强度：adjacency_strength ∈ {0.0（关闭）, 0.5（默认）, 1.0（满强度）}。
# 关闭 = 0.3.3 legacy 逐位一致；100% = 过冲预算标定强度（阶跃过冲 ≤15%、
# 组合 MTF 峰值 ≤ datasheet 峰值）；50% = 观感保守档（增强减半）。
import json

DEFAULTS = {
    "adjacency_strength": 0.5,
    # 测试用数值覆盖：{"a": 0.1, "sn": 0.004, "sw": 0.2}（mm），改参数无需重新拟合
    "override": {},
}

_CURRENT = dict(DEFAULTS)


def get():
    return dict(_CURRENT)


def set(**kwargs):
    for k, v in kwargs.items():
        if k not in _CURRENT:
            raise KeyError(f"未知选项：{k}")
        _CURRENT[k] = v


def reset():
    _CURRENT.clear()
    _CURRENT.update(DEFAULTS)


def fingerprint():
    return json.dumps(_CURRENT, sort_keys=True, ensure_ascii=False)
