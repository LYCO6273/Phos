#赛博请神
# 5213_data.py —— 5213（Vision3 200T）数据加载器
# 作用：集中加载 Films/5213/data/ 下所有由用户手动描点、经平滑插值
#       处理后的数据，带缓存，避免每次处理都重复读盘。
# 数据清单：
#   char_curves_5213.npz         感光特性曲线（Status M 密度 vs logE）
#   granularity_5213.npz         RMS 颗粒度（Sigma D vs logE）
#   spectral_mtf_5213.npz        模量传递函数 MTF（反应率 vs cyc/mm）
#   spectral_dye_density_5213.npz 染料光谱密度（黄/品红/青归一化 + 中灰/D-min）
# 注：spectral_sensitivity_5213.npz 是光谱灵敏度的处理记录，
#     运行时由 Spectral/migrate.py 直接读取其 CSV 版本。

import os

import numpy as np


# 数据目录（相对本文件：Films/5213/data）
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
    """网格是否严格等距（0.3.1 起所有查表轴统一插值到均匀网格）。"""
    ax = np.asarray(ax, dtype=np.float64)
    dx = np.diff(ax)
    return bool(np.allclose(dx, dx[0], rtol=1e-9, atol=1e-9))


def get_char_curves():
    """特性曲线：返回 logE 网格与三层密度、D-min/D-max（网格两端值）。

    logE 网格与密度数组均按 logE 升序排列，可直接 np.interp。
    层映射：R=红感光（青染料层）、G=绿感光（品红染料层）、B=蓝感光（黄染料层）。
    logE_uniform=True 表示网格严格等距，模型可走免二分快速插值。
    """
    data = _load("char_curves_5213.npz")
    logE = data["log10_exposure_lux_seconds"]
    return {
        "logE": logE,
        "logE_uniform": _is_uniform(logE),
        "density_R": data["density_R"],
        "density_G": data["density_G"],
        "density_B": data["density_B"],
        # 网格两端值即渐近的 D-min / D-max（PCHIP + 指数外推的结果）
        "dmin_R": float(data["density_R"][0]),
        "dmin_G": float(data["density_G"][0]),
        "dmin_B": float(data["density_B"][0]),
        "dmax_R": float(data["density_R"][-1]),
        "dmax_G": float(data["density_G"][-1]),
        "dmax_B": float(data["density_B"][-1]),
    }


def get_granularity():
    """颗粒度：返回 logE 网格与三层 Sigma D（按 logE 升序）。"""
    data = _load("granularity_5213.npz")
    logE = data["log10_exposure_lux_seconds"]
    return {
        "logE": logE,
        "logE_uniform": _is_uniform(logE),
        "sigma_R": data["sigma_R"],
        "sigma_G": data["sigma_G"],
        "sigma_B": data["sigma_B"],
    }


def get_mtf():
    """模量传递函数：返回原始描点与插值稠密网格。

    频率单位：周期/毫米（cyc/mm，对数刻度）；响应为 0-1 比例（100%=1.0）。
    原始点（freq_*_raw / resp_*_raw）供 PSF 频域拟合使用；
    稠密网格（frequency / response_*）为 PCHIP 插值 + 两端外推的记录。
    """
    data = _load("spectral_mtf_5213.npz")
    return {
        "freq_R_raw": data["freq_R_raw"],
        "freq_G_raw": data["freq_G_raw"],
        "freq_B_raw": data["freq_B_raw"],
        "resp_R_raw": data["resp_R_raw"],
        "resp_G_raw": data["resp_G_raw"],
        "resp_B_raw": data["resp_B_raw"],
        "frequency": data["frequency"],
        "response_R": data["response_R"],
        "response_G": data["response_G"],
        "response_B": data["response_B"],
    }


def get_dye_data():
    """染料光谱：返回 400-800nm 网格、归一化染料曲线与中灰/D-min 绝对密度。"""
    data = _load("spectral_dye_density_5213.npz")
    return {
        "wavelength": data["wavelength_nm"],
        "yellow_norm": data["dye_yellow_norm"],
        "magenta_norm": data["dye_magenta_norm"],
        "cyan_norm": data["dye_cyan_norm"],
        "midscale": data["midscale_density"],
        "dmin": data["dmin_density"],
    }
