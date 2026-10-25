#赛博请神
# reconstruct.py —— 光谱重建迁移（RGB → 三层曝光统计映射）
#
# 思路（为什么叫"光谱重建"）：
#   sRGB 是"场景光谱 → 标准观察者"的 3 维投影；胶片三层曝光是同一光谱
#   "→ 乳剂层灵敏度"的 3 维投影。真实物体光谱是连续、平滑、非负的，
#   且反射率不超过 1——这些约束构成"物理光谱先验"。
#   我们不直接重建完整光谱（3 维→61 维是病态问题，误差会在曝光积分里
#   放大），而是在该先验下直接学习 RGB→三层曝光 的 3×4 线性映射：
#      E = [R, G, B, 1] · M_film
#   训练数据全部由合成光谱生成（无需实拍）：每条合成光谱同时投影到
#   CIE 1931 2° 观察者（输入侧 sRGB）和胶片灵敏度（输出侧曝光），
#   回归矩阵就是"真实光谱先验"下的最优统计映射。
#
# 历史（0.3.2 A/B 实验结论）：
#   CMF 伪发射把 sRGB 当作"CIE-RGB 色匹配函数形状的光谱"（数转胶假设）；
#   光谱重建把 sRGB 当作"最可能产生该颜色的真实物体光谱"（模拟拍摄假设）。
#   对比实验中重建在纯红偏橙修正、暗部灰阶（无红雾）上明显更好，
#   因此 0.3.3 起 CMF 伪发射整体移除，光谱重建是唯一迁移路径，
#   下游管线（特性曲线/反转/扫描）按同一相对中灰轴工作。
#
# 数据：合成光谱（本模块自足）+ 色卡 24 色真实光谱（Spectral/data/，
#       仅用于独立验证，不参与训练）。

import csv
import os

import numpy as np

from . import cie_data


# Spectral 目录（相对本文件：Phos/Spectral）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 每款胶片灵敏度 CSV（与 migrate.py 同一份数据）
_SENSITIVITY_CSV = {
    "5213": os.path.join("..", "Films", "5213", "data", "spectral_sensitivity_5213.csv"),
    "gold200": os.path.join("..", "Films", "Gold200", "data", "spectral_sensitivity_gold200.csv"),
    "hp5": os.path.join("..", "Films", "HP5", "data", "spectral_sensitivity_hp5.csv"),
    "provia100f": os.path.join("..", "Films", "Provia100F", "data",
                               "Provia100F_spectral_sensitivity.csv"),
}

# 色卡光谱（独立验证集）：Spectral/data/colorchecker_babel.csv
_CC_CSV = os.path.join("data", "colorchecker_babel.csv")

# 映射缓存：Phos/Data/cache/spectral_reconstruct.npz
_CACHE_DIR = os.path.normpath(os.path.join(_BASE_DIR, "..", "Data", "cache"))
_CACHE_FILE = os.path.join(_CACHE_DIR, "spectral_reconstruct.npz")

# 训练集规模与随机种子（固定，保证可复现）
_N_TRAIN = 3000
_SEED = 42
# 缓存版本：合成光谱生成器/训练流程变更时递增，强制重新训练
_CACHE_VERSION = 7

# 每胶片映射缓存（film -> (4,3) 或 (4,1) float32）
_MAP_CACHE = {}

# 注：0.3.x 阶段曾实验 density/density_err 加权映射（A/B 对比），
# 双轴报告证明其在常见色上负收益，已放弃；本项目态度是"线性问题
# 定量准确、复杂问题定性正确"，光谱迁移保持 uniform 统计映射即可。


# ---------- 合成光谱生成器（物理先验的编码处） ----------

def _synth_spectrum(rng):
    """生成一条物理合理的光谱反射率曲线（(61,)）。

    先验规则：
      1. 由 1-4 个宽高斯峰叠加（峰 410-690nm、σ 25-90nm）——真实物体
         光谱是平滑连续的宽带，不是任意高频形状；
      2. 若光谱重心偏红（600nm 以上为主），把 460-560nm 压到 15-45%、
         560-650nm 线性回升——真实红色颜料/染料在蓝绿端强吸收、红边
         平滑陡升（红色偏橙修正的关键先验）；
      3. 若光谱以蓝端为主（400-490nm 显著高于红端），把 600nm 以上压到
         5-20%——真实蓝色颜料/染料在红端强吸收（避免蓝色/蓝绿色相
         偏红；此先验与红色先验对称）；
      4. 加轻微平滑起伏模拟真实反射率的不规则性；
      5. 截断到 [0, 1]——反射率物理上不能为负、不能超过 1。
    """
    n_gauss = int(rng.integers(1, 5))
    rho = np.zeros(len(cie_data.WAVELENGTHS), dtype=np.float64)
    wl = cie_data.WAVELENGTHS
    for _ in range(n_gauss):
        mu = rng.uniform(410.0, 690.0)
        sigma = rng.uniform(30.0, 90.0)
        amp = rng.uniform(0.3, 1.5)
        rho += amp * np.exp(-0.5 * ((wl - mu) / sigma) ** 2)
    # 红色先验：光谱重心在 600nm 以上时，压低蓝绿端反射。
    # 平滑斜坡：460-560nm 压低（乘 low），560-650nm 线性回升到 1，
    # 斜率和真实红色颜料（色卡 red 红边 ~0.07/5nm）同量级。
    red_weight = np.clip(rho * (wl > 600.0), 0.0, None)
    red_center = float(np.average(wl, weights=red_weight))
    if np.isfinite(red_center) and red_center > 590.0:
        low = rng.uniform(0.15, 0.45)
        # 560nm 前乘 low，560-650nm 线性回升到 1
        window = low + (1.0 - low) * np.clip((wl - 560.0) / (650.0 - 560.0), 0.0, 1.0)
        window = np.clip(window, 0.0, 1.0)
        rho *= window
    # 蓝色先验：全局峰值落在蓝区（400-490nm）时，压低红端（590nm 以上）。
    # 平滑斜坡：590-650nm 线性下降，650nm 以上保持低位。
    # low_b 取 0.02-0.08（真实纯蓝颜料红端 <0.05），避免高斯长尾
    # 给"蓝色"光谱注入过量红端反射——这是蓝绿色相偏红的根源。
    peak_wl = float(wl[int(np.argmax(rho))])
    if peak_wl < 490.0 and float(rho[(wl >= 400.0) & (wl < 490.0)].mean()) > 0.10:
        low_b = rng.uniform(0.02, 0.08)
        window_b = 1.0 - (1.0 - low_b) * np.clip((wl - 590.0) / (650.0 - 590.0), 0.0, 1.0)
        window_b = np.clip(window_b, 0.0, 1.0)
        rho *= window_b
    # 轻微平滑起伏
    rho += 0.02 * np.sin(wl * rng.uniform(0.02, 0.08) + rng.uniform(0.0, 6.0))
    return np.clip(rho, 0.0, 1.0)


def _synth_spectra(n=_N_TRAIN, seed=_SEED):
    """生成 n 条合成光谱（(n,61)），固定种子可复现。

    在随机宽带光谱之外追加 21 条平坦灰阶（反射率 0.00-1.00 每档 0.05），
    保证白场/灰场落在训练数据里，避免 (1,1,1) 纯白成为外推点；
    再追加一组窄峰高饱和光谱（σ=15/22nm，410-690nm 每 20nm 一个），
    补足色域角落（尤其纯蓝区）——随机宽带光谱的 sRGB 很少落在
    蓝色色域边缘（纯蓝附近样本仅个位数），会导致纯蓝外推不可靠。
    """
    rng = np.random.default_rng(seed)
    synth = np.array([_synth_spectrum(rng) for _ in range(n)])
    gray = np.tile(np.linspace(0.0, 1.0, 21)[:, None], (1, len(cie_data.WAVELENGTHS)))
    # 高饱和补充：窄高斯峰（保留红/蓝先验的平滑斜坡处理）
    wl = cie_data.WAVELENGTHS
    narrow = []
    for mu in np.arange(410.0, 691.0, 20.0):
        for sigma in (15.0, 22.0):
            rho = np.exp(-0.5 * ((wl - mu) / sigma) ** 2)
            red_weight = np.clip(rho * (wl > 600.0), 0.0, None)
            red_center = float(np.average(wl, weights=red_weight))
            if np.isfinite(red_center) and red_center > 590.0:
                low = 0.2
                window = low + (1.0 - low) * np.clip((wl - 560.0) / (650.0 - 560.0), 0.0, 1.0)
                rho *= np.clip(window, 0.0, 1.0)
            peak_wl = float(wl[int(np.argmax(rho))])
            if peak_wl < 490.0 and float(rho[(wl >= 400.0) & (wl < 490.0)].mean()) > 0.10:
                low_b = 0.05
                window_b = 1.0 - (1.0 - low_b) * np.clip((wl - 590.0) / (650.0 - 590.0), 0.0, 1.0)
                rho *= np.clip(window_b, 0.0, 1.0)
            narrow.append(np.clip(rho, 0.0, 1.0))
    return np.vstack([synth, gray, np.array(narrow)])


# ---------- 灵敏度读取（白场 ρ=1 归一化，量纲：白场每层曝光 = 1） ----------

def _load_sensitivities(film):
    """读彩负/正片三层灵敏度 CSV，按"完美漫反射白 ρ=1 → 每层曝光 1"归一化。

    返回 (61,3)，列序 B/G/R。白场归一化：白场（ρ=1）时每层曝光 = 1，
    保证特性曲线的 logE 标定不受影响。
    """
    if film == "hp5":
        return _load_bw_sensitivities(film)
    path = os.path.join(_BASE_DIR, _SENSITIVITY_CSV[film])
    wl, blue, green, red = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wl.append(float(row["wavelength_nm"]))
            blue.append(float(row["blue_lin"]))
            green.append(float(row["green_lin"]))
            red.append(float(row["red_lin"]))
    wl = np.asarray(wl, dtype=np.float64)
    sens = np.stack([np.interp(cie_data.WAVELENGTHS, wl, blue),
                     np.interp(cie_data.WAVELENGTHS, wl, green),
                     np.interp(cie_data.WAVELENGTHS, wl, red)], axis=1)
    sens = np.clip(sens, 0.0, None)
    # 白场（ρ=1 全反射）曝光归一化：Σ D65·S·Δλ → 每层曝光 1
    white = (sens * cie_data.D65_SPD[:, None] * cie_data.DELTA_LAMBDA).sum(axis=0)
    return sens / white


def _load_bw_sensitivities(film):
    """读黑白单通道灵敏度 CSV，白场 ρ=1 归一化（返回 (61,1)）。"""
    path = os.path.join(_BASE_DIR, _SENSITIVITY_CSV[film])
    wl, sens = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wl.append(float(row["wavelength_nm"]))
            sens.append(float(row["relative_sensitivity"]))
    wl = np.asarray(wl, dtype=np.float64)
    sens = np.clip(np.interp(cie_data.WAVELENGTHS, wl, sens), 0.0, None)
    white = (sens * cie_data.D65_SPD * cie_data.DELTA_LAMBDA).sum()
    return (sens / white)[:, None]


# ---------- 训练数据标注 ----------

def _spec_to_srgb(rho):
    """反射光谱 → 线性 sRGB（D65 下标准观察者看到的颜色）。

    白场基准 = ρ(λ)=1 的完美漫反射白（XYZ 的 Y 归一化到 1），
    因此 sRGB(1,1,1) 对应白，与输入图像的量纲一致。
    """
    illum = rho * cie_data.D65_SPD[None, :]
    xyz = np.einsum("bw,wc->bc", illum, cie_data.CMF_1931_2) * cie_data.DELTA_LAMBDA
    white_xyz = (cie_data.D65_SPD[:, None] * cie_data.CMF_1931_2
                 * cie_data.DELTA_LAMBDA).sum(axis=0)
    xyz = xyz / white_xyz[1]
    return np.clip(xyz @ cie_data.XYZ_TO_SRGB.T, 0.0, None)


def _spec_to_exposure(rho, sens):
    """反射光谱 → 三层曝光：E = Σ D65·ρ·S·Δλ（sens 已白场归一化）。

    sens 形状 (61,3)（彩负，列 B/G/R）或 (61,1)（黑白）。
    """
    return np.einsum("bw,wc->bc", rho * cie_data.D65_SPD[None, :], sens) \
        * cie_data.DELTA_LAMBDA


# ---------- 训练与缓存 ----------

def _train_map(film):
    """用合成光谱训练 RGB→曝光 映射矩阵。

    彩负/正片返回 (4,3)（行 [R,G,B,1]，列 B/G/R 层）；
    HP5 返回 (4,1)（单通道）。
    """
    synth = _synth_spectra()
    srgb = _spec_to_srgb(synth)
    sens = _load_sensitivities(film)
    exposure = _spec_to_exposure(synth, sens)
    design = np.column_stack([srgb, np.ones(len(srgb))])
    matrix, *_ = np.linalg.lstsq(design, exposure, rcond=None)
    # 偏置行清零：暗部（RGB→0）的曝光必须趋近 0——"无光=黑"是物理事实。
    # 不带偏置约束的回归会让偏置项吸收训练数据暗部样本的统计色偏
    # （合成暗色光谱的 R 端反射略高），给全图暗部注入 R 层曝光，
    # 表现为罩上一层"红雾"并抹掉青蓝阴影。清零后暗部强制为黑。
    matrix[3] = 0.0
    # 白场后归一化：让 (1,1,1,0)·M = 每层曝光 1（RGB 部分）。
    # 回归是统计最优，白场 (1,1,1) 可能略微偏离精确值（约 3-7%）；
    # 下游特性曲线/反转锚点都以"白场曝光=1"为标定基准，必须精确归一。
    white_response = np.array([1.0, 1.0, 1.0, 0.0]) @ matrix
    matrix[:3] = matrix[:3] / white_response[None, :]
    matrix[3] = 0.0
    return matrix.astype(np.float32)


def _ensure_cache():
    """训练并缓存所有胶片映射到 npz；已存在则直接加载。"""
    global _MAP_CACHE
    if _MAP_CACHE:
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    films = ["gold200", "5213", "provia100f", "hp5"]
    if os.path.exists(_CACHE_FILE):
        data = np.load(_CACHE_FILE, allow_pickle=True)
        version_ok = (
            int(data.get("cache_version", 0)) == _CACHE_VERSION
            and int(data.get("n_spectra", 0)) == _N_TRAIN
            and int(data.get("seed", -1)) == _SEED
        )
        if version_ok and all(f"map_{film}" in data for film in films):
            for film in films:
                _MAP_CACHE[film] = data[f"map_{film}"]
            return
    save = {}
    for film in films:
        mat = _train_map(film)
        _MAP_CACHE[film] = mat
        save[f"map_{film}"] = mat
    save["n_spectra"] = np.array(_N_TRAIN)
    save["seed"] = np.array(_SEED)
    save["cache_version"] = np.array(_CACHE_VERSION)
    np.savez(_CACHE_FILE, **save)


def _get_map(film):
    """返回某胶片的映射矩阵（float32 (4,3) 或 (4,1)）。"""
    _ensure_cache()
    if film not in _MAP_CACHE:
        raise ValueError(f"未知胶片：{film}（光谱重建支持 gold200/5213/provia100f/hp5）")
    return _MAP_CACHE[film]


# ---------- 运行时迁移主入口 ----------

def migrate_rgb_reconstruct(lux_r, lux_g, lux_b, film="5213"):
    """光谱重建 RGB→三层曝光（彩负/正片）。

    参数/返回值与 migrate.migrate_rgb_primary 完全一致：
        输入线性 RGB 浮点通道；返回 (out_r, out_g, out_b) 三层曝光，
        白场 (1,1,1) → 每层曝光 ≈1。
    """
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")
    rgb = np.stack([np.asarray(lux_r, dtype=np.float32),
                    np.asarray(lux_g, dtype=np.float32),
                    np.asarray(lux_b, dtype=np.float32)], axis=-1)
    matrix = _get_map(film)                       # (4,3) 行 R/G/B/1，列 B/G/R
    # 广播矩阵乘：E = rgb·M[:3] + M[3]（不物化 (N,4) 常数列数组，
    # 3024×4536 时省约 220MB 瞬时内存；与旧实现的浮点差 ~1e-7）
    flat = rgb.reshape(-1, 3)
    exposure_bgr = (flat @ matrix[:3] + matrix[3]).reshape(rgb.shape)
    return (np.ascontiguousarray(exposure_bgr[..., 2]),
            np.ascontiguousarray(exposure_bgr[..., 1]),
            np.ascontiguousarray(exposure_bgr[..., 0]))


def migrate_bw_reconstruct(lux_r, lux_g, lux_b, film="hp5"):
    """光谱重建 RGB→单通道曝光（黑白）。

    参数/返回值与 migrate.migrate_bw_primary 一致：返回单通道曝光 (H,W)，
    白场 (1,1,1) → 曝光 ≈1。
    """
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")
    rgb = np.stack([np.asarray(lux_r, dtype=np.float32),
                    np.asarray(lux_g, dtype=np.float32),
                    np.asarray(lux_b, dtype=np.float32)], axis=-1)
    matrix = _get_map(film)                       # (4,1)
    flat = rgb.reshape(-1, 3)
    return (flat @ matrix[:3] + matrix[3]).reshape(rgb.shape[:2]).astype(np.float32)


# ---------- 独立验证（色卡 24 色，不参与训练） ----------

def _load_colorchecker():
    """读色卡 24 色光谱（(24,61)，行序同 CSV 表头）。"""
    path = os.path.join(_BASE_DIR, _CC_CSV)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    names = [n for n in reader.fieldnames if n != "wavelength_nm"]
    wl = np.array([float(r["wavelength_nm"]) for r in rows], dtype=np.float64)
    rho = np.array([
        np.interp(cie_data.WAVELENGTHS, wl,
                  np.array([float(r[n]) for r in rows], dtype=np.float64))
        for n in names
    ])  # (24,61)，行=色块
    return rho


def verify_on_colorchecker(film):
    """用色卡 24 色真实光谱验证映射的曝光误差（训练集外）。

    返回 (mae, max_err)：三层曝光平均绝对误差 / 最大绝对误差。
    曝光量纲为 0-1（白场=1），MAE < 0.04 视为通过。
    """
    rho_cc = _load_colorchecker()
    sens = _load_sensitivities(film)
    e_true = _spec_to_exposure(rho_cc, sens)
    srgb_cc = _spec_to_srgb(rho_cc)
    design = np.column_stack([srgb_cc, np.ones(len(srgb_cc))])
    e_pred = design @ _get_map(film)
    err = np.abs(e_pred - e_true)
    return float(err.mean()), float(err.max())


# ---------- metamer 对数据集（"压制哨兵"，固定种子可复现） ----------

_METAMER_FILE = os.path.join("data", "metamer_pairs.npz")
_METAMER_N_PAIRS = 20
_METAMER_SEED = 7
_METAMER_DR_THRESHOLD = 0.02
_METAMER_SCAN = 150000


def build_metamer_pairs(n_pairs=_METAMER_N_PAIRS, seed=_METAMER_SEED,
                        dr_threshold=_METAMER_DR_THRESHOLD,
                        scan=_METAMER_SCAN):
    """扫描合成光谱找近 metamer 对（Δrgb 小、胶片曝光差大），固定种子。

    基准胶片用 gold200 的曝光差异排序（四卷共享同一批对，各卷曝光差异
    在验证时按各自灵敏度现算）。返回 (rho_a, rho_b, srgb_a, srgb_b)，
    形状分别为 (n,61) / (n,61) / (n,3) / (n,3)。
    """
    rng = np.random.default_rng(seed)
    sens = _load_sensitivities("gold200")
    cands = []
    for _ in range(int(scan)):
        r1, r2 = _synth_spectrum(rng), _synth_spectrum(rng)
        s1, s2 = _spec_to_srgb(r1[None, :]), _spec_to_srgb(r2[None, :])
        if float(np.linalg.norm(s1 - s2)) >= dr_threshold:
            continue
        e1 = _spec_to_exposure(r1[None, :], sens)[0]
        e2 = _spec_to_exposure(r2[None, :], sens)[0]
        cands.append((float(np.linalg.norm(e1 - e2)), r1, r2, s1[0], s2[0]))
    cands.sort(key=lambda x: -x[0])
    top = cands[:n_pairs]
    return (np.array([c[1] for c in top], dtype=np.float32),
            np.array([c[2] for c in top], dtype=np.float32),
            np.array([c[3] for c in top], dtype=np.float32),
            np.array([c[4] for c in top], dtype=np.float32))


def load_metamer_pairs():
    """加载固定 metamer 对；文件缺失时按固定种子重建并保存。

    返回 (rho_a, rho_b, srgb_a, srgb_b)。数据集是"压制哨兵"：
    同一个 sRGB 对应的真实胶片曝光本应不同，映射输出只能给条件均值——
    这个数据集量化映射输出距两端真实值的距离，用于回归看板。
    """
    path = os.path.join(_BASE_DIR, _METAMER_FILE)
    if not os.path.exists(path):
        rho_a, rho_b, s_a, s_b = build_metamer_pairs()
        np.savez(path, rho_a=rho_a, rho_b=rho_b, srgb_a=s_a, srgb_b=s_b)
        return rho_a, rho_b, s_a, s_b
    d = np.load(path)
    return d["rho_a"], d["rho_b"], d["srgb_a"], d["srgb_b"]
