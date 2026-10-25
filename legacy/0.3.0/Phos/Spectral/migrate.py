#赛博请神
# migrate.py —— 输入端光谱迁移（共享地基）
# 作用：把线性 RGB 升级为“到达胶片的光谱 → 按乳剂层灵敏度积分”，
#       输出仍是三个浮点通道（层曝光量），量纲与输入一致
#       （白场≈1，高光>1 按比例保留）。
# 唯一迁移路径：
#   migrate_rgb_primary() CIE-RGB 色匹配函数伪发射源 → 实测灵敏度吸收积分
# 说明：5213 的灵敏度数据由用户手动描点，存放在 Films/5213/data/ 下；
#       hp5 仍使用平坦全色响应近似；gold200 沿用 5213 灵敏度作代理。

import csv
import os

import numpy as np

from . import cie_data


# 5213 数据目录（相对本文件：Phos/Spectral/../Films/5213/data）
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Films", "5213", "data")

# 每款胶片对应的灵敏度数据（film_key -> (61,3) 归一化灵敏度）
_SENSITIVITY_CACHE = {}


def _load_vision_sensitivities():
    """读取 5213 手动描点的 B/G/R 光谱灵敏度（线性值）。"""
    path = os.path.join(_DATA_DIR, "spectral_sensitivity_5213.csv")
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
    sensitivity = np.stack([blue, green, red], axis=1)  # (61,3)
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
        # 5213 使用用户手描数据；gold200 暂用同一份作代理（柯达彩负同族）
        sensitivity = _load_vision_sensitivities()
    elif film == "hp5":
        # HP5 数据表只有定性光谱图，用平坦全色响应近似
        sensitivity = np.ones((61, 3), dtype=np.float64)
    else:
        raise ValueError(f"未知胶片：{film}，目前支持 hp5 / gold200 / 5213")

    sensitivity = _normalize_sensitivities(sensitivity)
    _SENSITIVITY_CACHE[film] = sensitivity
    return sensitivity


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
