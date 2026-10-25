#赛博请神
# Provia100F_data.py —— FUJICHROME Provia 100F（RDP III，ISO 100 反转片）数据加载器
# 作用：集中加载 Films/Provia100F/data/ 下处理好的数据，带缓存。
# 数据清单：
#   Provia100F_characteristic_curve.npz  特性曲线（三层负斜率正像曲线，绝对 logH）
#   Provia100F_granularity.npz           颗粒度（5213 形状代理，rms=8@D=1.0）
#   Provia100F_dye_density.npz           染料光谱密度（三层 + 峰值归一化）
#   Provia100F_MTF.npz                   MTF（单层描点，三层共用）
# 查表轴统一为严格均匀网格，模型走免二分快速插值。

import os

import numpy as np


# 数据目录（相对本文件：Films/Provia100F/data）
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
    """特性曲线：绝对 logH 网格 + 三层密度（负斜率正像曲线）。

    logE 轴已按 ISO 100 标定（logH_gray=log10(12.5/100)=-0.90309），
    网格严格均匀（步长 0.05）。层映射：R=红感光/青、G=绿感光/品红、B=蓝感光/黄。
    """
    data = _load("Provia100F_characteristic_curve.npz")
    logE = data["log10_exposure_lux_seconds"]
    return {
        "logE": logE,
        "logE_uniform": _is_uniform(logE),
        "density_R": data["density_R"],
        "density_G": data["density_G"],
        "density_B": data["density_B"],
        "dmin_R": float(data["density_R"][-1]),   # 正片高曝光端 → D_min
        "dmin_G": float(data["density_G"][-1]),
        "dmin_B": float(data["density_B"][-1]),
        "dmax_R": float(data["density_R"][0]),    # 低曝光端 → D_max
        "dmax_G": float(data["density_G"][0]),
        "dmax_B": float(data["density_B"][0]),
    }


def get_granularity():
    """颗粒度：绝对 logE 网格 + 三层 σD（5213 形状代理）。"""
    data = _load("Provia100F_granularity.npz")
    logE = data["log10_exposure_lux_seconds"]
    return {
        "logE": logE,
        "logE_uniform": _is_uniform(logE),
        "sigma_R": data["sigma_R"],
        "sigma_G": data["sigma_G"],
        "sigma_B": data["sigma_B"],
    }


def get_dye_data():
    """染料光谱：400-700nm 网格、归一化染料曲线与 D-min。

    D-min 用典型反转片近似常数 0.1（正片灰雾近中性，待实测替换）。
    """
    data = _load("Provia100F_dye_density.npz")
    dmin = np.full(61, 0.1, dtype=np.float32)
    return {
        "wavelength": data["wavelength_nm"],
        "yellow_norm": data["dye_yellow_norm"],
        "magenta_norm": data["dye_magenta_norm"],
        "cyan_norm": data["dye_cyan_norm"],
        "midscale": np.full(61, 0.6, dtype=np.float32),  # 占位（待实测中灰）
        "dmin": dmin,
    }


def get_mtf():
    """模量传递函数（单层描点，三层共用）。

    频率单位：周期/毫米；freq_raw/resp_raw 为描点，frequency/response 为
    PCHIP 稠密网格（0-1 比例，高频对数线性衰减外推）。
    """
    data = _load("Provia100F_MTF.npz")
    return {
        "freq_raw": data["freq_raw"],
        "resp_raw": data["resp_raw"],
        "frequency": data["frequency"],
        "response": data["response"],
    }
