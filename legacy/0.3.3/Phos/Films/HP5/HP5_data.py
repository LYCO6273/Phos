#赛博请神
# HP5_data.py —— ILFORD HP5 Plus（ISO 400 黑白负片）数据加载器
# 作用：集中加载 Films/HP5/data/ 下处理好的数据，带缓存。
# 数据清单：
#   char_curves_hp5.npz      感光特性曲线（密度 vs 绝对 logH，ILFOTEC HC 显影）
#   granularity_hp5.npz      RMS 颗粒度（σD vs 净密度，48µm 孔径）
#   spectral_mtf_hp5.npz     MTF（KODAK TRI-X 400 代理，同级 ISO 400 黑白卷）
#   spectral_sensitivity_hp5.csv  光谱灵敏度（单列，352-660nm，migrate 直接读取）
# 数据来源：胶片技术文档/Ilford/黑白/HP5 Plus + Kodak/黑白/TRI-X 400。
# 查表轴统一为严格均匀网格，模型走免二分快速插值。

import os

import numpy as np


# 数据目录（相对本文件：Films/HP5/data）
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 缓存
_CACHE = {}


def _load(name):
    """读取 NPZ 并缓存。"""
    if name in _CACHE:
        return _CACHE[name]
    path = os.path.join(_DATA_DIR, name)
    data = np.load(path)
    _CACHE[name] = data
    return data


def _is_uniform(ax):
    """网格是否严格等距（标准流程要求所有查表轴统一均匀网格）。"""
    ax = np.asarray(ax, dtype=np.float64)
    dx = np.diff(ax)
    return bool(np.allclose(dx, dx[0], rtol=1e-9, atol=1e-9))


def get_char_curves():
    """特性曲线：绝对 logH 网格 + 单层密度、D-min/D-max（网格两端值）。

    logE 轴已按 ISO 400 标定（logH_gray=log10(12.5/400)=-1.50515），
    网格严格均匀（步长 0.05），可直接快速插值。
    """
    data = _load("char_curves_hp5.npz")
    logE = data["log10_exposure_lux_seconds"]
    return {
        "logE": logE,
        "logE_uniform": _is_uniform(logE),
        "density": data["density"],
        "dmin": float(data["density"][0]),
        "dmax": float(data["density"][-1]),
    }


def get_granularity():
    """颗粒度：净密度网格 + σD（黑白 rms 曲线 X 轴=密度，直接密度域查表）。"""
    data = _load("granularity_hp5.npz")
    gden = data["density"]
    return {
        "density": gden,
        "density_uniform": _is_uniform(gden),
        "sigma_D": data["sigma_D"],
    }


def get_mtf():
    """模量传递函数（Tri-X 400 代理，单层共用）。

    频率单位：周期/毫米；响应：freq_raw/resp_raw 为描点百分比，
    frequency/response 为 PCHIP 稠密网格（0-1 比例，高频对数线性衰减外推）。
    """
    data = _load("spectral_mtf_hp5.npz")
    return {
        "freq_raw": data["freq_raw"],
        "resp_raw": data["resp_raw"],
        "frequency": data["frequency"],
        "response": data["response"],
    }
