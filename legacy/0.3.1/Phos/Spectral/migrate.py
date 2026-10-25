#赛博请神
# migrate.py —— 输入端光谱迁移（共享地基）
# 作用：把线性 RGB 升级为“到达胶片的光谱 → 按乳剂层灵敏度积分”，
#       输出仍是三个浮点通道（层曝光量），量纲与输入一致
#       （白场≈1，高光>1 按比例保留）。
# 唯一迁移路径：
#   migrate_rgb_primary() CIE-RGB 色匹配函数伪发射源 → 实测灵敏度吸收积分
#   migrate_bw_primary()  黑白单通道版（同一思路，灵敏度为单列）
# 说明：5213 / Gold 200 的灵敏度数据均由用户手动描点，存放在各自
#       Films/<胶片>/data/ 下；HP5 为单通道全色灵敏度（352-660nm 描点）。

import csv
import os

import numpy as np

from . import cie_data


# Spectral 目录（相对本文件：Phos/Spectral）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 每款胶片对应的灵敏度 CSV（film_key -> 相对 Spectral 的路径）
_SENSITIVITY_CSV = {
    "5213": os.path.join("..", "Films", "5213", "data", "spectral_sensitivity_5213.csv"),
    "gold200": os.path.join("..", "Films", "Gold200", "data", "spectral_sensitivity_gold200.csv"),
    "hp5": os.path.join("..", "Films", "HP5", "data", "spectral_sensitivity_hp5.csv"),
}

# 每款胶片对应的灵敏度数据（film_key -> (61,3) 归一化灵敏度）
_SENSITIVITY_CACHE = {}
_BW_SENSITIVITY_CACHE = {}


def _load_sensitivities_csv(path):
    """读取胶片描点的 B/G/R 光谱灵敏度（线性值）CSV，插值对齐到 CIE 波长网格。"""
    wavelengths = []
    blue = []
    green = []
    red = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wavelengths.append(float(row["wavelength_nm"]))
            blue.append(float(row["blue_lin"]))
            green.append(float(row["green_lin"]))
            red.append(float(row["red_lin"]))
    wl = np.asarray(wavelengths, dtype=np.float64)
    sensitivity = np.stack([blue, green, red], axis=1)  # (N,3)
    # 用线性插值对齐到本模块的 61 个波长（数据本身已是 5nm，这一步只是兜底）
    aligned = np.column_stack([
        np.interp(cie_data.WAVELENGTHS, wl, sensitivity[:, i])
        for i in range(3)
    ])
    return np.clip(aligned, 0.0, None)


def _normalize_sensitivities(sensitivity):
    """按白场曝光归一化：让纯白（r=1）在每层都得到曝光量 1。"""
    weights = sensitivity * cie_data.D65_SPD[:, None] * cie_data.DELTA_LAMBDA
    white_exposure = weights.sum(axis=0)
    return sensitivity / white_exposure


def _get_sensitivities(film):
    """按胶片名返回归一化灵敏度矩阵 (61,3)，列顺序为 B/G/R（蓝/绿/红）。"""
    if film in _SENSITIVITY_CACHE:
        return _SENSITIVITY_CACHE[film]

    if film in ("5213", "gold200"):
        # 各胶片使用自己的用户描点数据
        sensitivity = _load_sensitivities_csv(os.path.join(_BASE_DIR, _SENSITIVITY_CSV[film]))
    elif film == "hp5":
        # HP5 数据表只有定性光谱图，用平坦全色响应近似
        sensitivity = np.ones((61, 3), dtype=np.float64)
    else:
        raise ValueError(f"未知胶片：{film}，目前支持 hp5 / gold200 / 5213")

    sensitivity = _normalize_sensitivities(sensitivity)
    _SENSITIVITY_CACHE[film] = sensitivity
    return sensitivity


def _load_bw_sensitivities_csv(path):
    """读取黑白胶片单列光谱灵敏度（wavelength_nm, relative_sensitivity）。"""
    wavelengths = []
    sensitivity = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wavelengths.append(float(row["wavelength_nm"]))
            sensitivity.append(float(row["relative_sensitivity"]))
    wl = np.asarray(wavelengths, dtype=np.float64)
    sens = np.asarray(sensitivity, dtype=np.float64)
    aligned = np.interp(cie_data.WAVELENGTHS, wl, sens)
    return np.clip(aligned, 0.0, None)


def _get_bw_sensitivities(film):
    """按胶片名返回单通道归一化灵敏度 (61,)，白场（1,1,1）→ 曝光 1。"""
    if film in _BW_SENSITIVITY_CACHE:
        return _BW_SENSITIVITY_CACHE[film]
    if film in _SENSITIVITY_CSV:
        sensitivity = _load_bw_sensitivities_csv(
            os.path.join(_BASE_DIR, _SENSITIVITY_CSV[film]))
    else:
        raise ValueError(f"未知黑白胶片：{film}")
    # 白场归一化：让伪发射光谱（r̄+ḡ+b̄）在灵敏度上的积分 = 1
    white = (sensitivity * (cie_data.RGB_CMF_1931_2.sum(axis=1))
             * cie_data.DELTA_LAMBDA).sum()
    sensitivity = sensitivity / white
    _BW_SENSITIVITY_CACHE[film] = sensitivity
    return sensitivity


_BW_COEFF_CACHE = {}


def _bw_exposure_coeffs(film):
    """黑白单通道 1×3 权重：CIE-RGB 伪发射 → 灵敏度积分（列序 R/G/B）。"""
    if film in _BW_COEFF_CACHE:
        return _BW_COEFF_CACHE[film]
    sens = _get_bw_sensitivities(film)          # (61,)
    rgb_cmf = cie_data.RGB_CMF_1931_2           # (61,3)，列序 R/G/B
    w = (sens[:, None] * rgb_cmf).sum(axis=0) * cie_data.DELTA_LAMBDA
    w = w / w.sum()                             # 白场 R=G=B=1 → E=1
    _BW_COEFF_CACHE[film] = w.astype(np.float32)
    return _BW_COEFF_CACHE[film]


def migrate_bw_primary(lux_r, lux_g, lux_b, film="hp5"):
    """黑白单通道光谱迁移：CIE-RGB 伪发射源 → 实测灵敏度吸收积分。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（同形状二维数组）。
        film：黑白胶片名，目前支持 "hp5"。
    返回：
        单通道曝光量浮点数组（H,W），白场（1,1,1）→ 1。
    """
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")
    rgb = np.stack(
        [np.asarray(lux_r, dtype=np.float32),
         np.asarray(lux_g, dtype=np.float32),
         np.asarray(lux_b, dtype=np.float32)],
        axis=-1,
    )
    w = _bw_exposure_coeffs(film)
    return rgb @ w  # (H,W)


# ---------- 机械式 CIE-RGB 色匹配函数伪发射迁移 ----------
# 用 CIE 1931 RGB 标准色匹配函数 r̄(λ)/ḡ(λ)/b̄(λ) 直接作为三条
# “伪发射光谱”，按像素线性 RGB 加权合成到达胶片的光谱：
#   L(λ) = R·r̄(λ) + G·ḡ(λ) + B·b̄(λ)
# 再按实测的乳剂层灵敏度做“吸收”积分得到三层曝光：
#   E_i = Σ_λ S_i(λ)·L(λ)·Δλ
# 注意 r̄/ḡ/b̄ 自带负瓣（尤其 r̄ 在 440~530nm），伪发射允许“负光”，
# 由此算出的负曝光会在管线里被 E_FLOOR 下限夹掉。
# 白场归一化：白（R=G=B=1）时每层曝光量=1，
# 保证 18% 灰仍落在特性曲线的参考 logE 上，曝光标定不受影响。

# 每款胶片的伪发射曝光矩阵缓存（film -> (3,3) float32）
_PRIMARY_MATRIX_CACHE = {}


def _primary_exposure_matrix(film):
    """CIE-RGB 色匹配伪发射 → 三层曝光的 3×3 矩阵（行=B/G/R 层，列=R/G/B 原色）。"""
    if film in _PRIMARY_MATRIX_CACHE:
        return _PRIMARY_MATRIX_CACHE[film]
    sensitivity = _get_sensitivities(film)  # (61,3)，列序 B/G/R
    rgb_cmf = cie_data.RGB_CMF_1931_2  # (61,3)，列序 R/G/B
    matrix = np.empty((3, 3), dtype=np.float64)
    for primary in range(3):  # 原色：R/G/B
        for layer in range(3):  # 层：B/G/R
            matrix[layer, primary] = (
                (sensitivity[:, layer] * rgb_cmf[:, primary]).sum()
                * cie_data.DELTA_LAMBDA
            )
    # 白场归一化：每层对三个原色求和为 1（白 → 每层曝光 1）
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    _PRIMARY_MATRIX_CACHE[film] = matrix.astype(np.float32)
    return _PRIMARY_MATRIX_CACHE[film]


def migrate_rgb_primary(lux_r, lux_g, lux_b, film="5213"):
    """机械式 CIE-RGB 色匹配函数伪发射光谱迁移主入口。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（同形状二维数组）。
        film：胶片名，目前支持 "hp5" / "gold200" / "5213"。
    返回：
        (lux_r, lux_g, lux_b)：三层曝光量浮点通道（R/G/B 顺序），
        白场（1,1,1）→ 每层曝光 1。
    """
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")

    rgb = np.stack(
        [np.asarray(lux_r, dtype=np.float32),
         np.asarray(lux_g, dtype=np.float32),
         np.asarray(lux_b, dtype=np.float32)],
        axis=-1,
    )
    matrix = _primary_exposure_matrix(film)
    exposure_bgr = rgb @ matrix.T  # (H,W,3)，列序 B/G/R
    # 层接口约定 lux_r/lux_g/lux_b 对应 红/绿/蓝 感光层，做通道换位
    out_r = np.ascontiguousarray(exposure_bgr[..., 2])
    out_g = np.ascontiguousarray(exposure_bgr[..., 1])
    out_b = np.ascontiguousarray(exposure_bgr[..., 0])
    return out_r, out_g, out_b
