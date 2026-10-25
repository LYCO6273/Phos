#赛博请神
# Provia100F.py —— FUJICHROME Provia 100F（RDP III，ISO 100 彩色反转片）模型
# 彩色正片 = 彩负去掉"反转 + 校色 + 去色罩"（第三段 Scanner 只做显示映射）：
#   线性 RGB
#     → 光谱输入（固定）光谱重建迁移 → 层曝光 E_R/E_G/E_B
#     → [PSF] 物理核（核心高斯 + K0 扩散晕 − 邻界高斯，MTF 拟合，三层共用）
#     → 对数曝光（统一对数工作区 + 曝光偏移）
#     → 特性曲线 → Status M 密度（负斜率正像曲线：高曝光→低密度）
#     → 密度域颗粒（σ 按 logH 查表，5213 形状代理）
#     → 染料光谱堆叠 → CIE-RGB 客观扫描（扫描值即正像，无需反转）
#     → 显示映射：三锚点平滑校正（黑/灰/白）→ 对比度 → sRGB 输出
#
# 输出端无反转、无校色：正片色彩是它自己的（包括 D-min 灰雾），
# 忠实保留即"肉眼直接观看"的观感。

import os

import cv2
import numpy as np

import Films.Provia100F.Provia100F_data as _data


# ---------- 曝光标定常量 ----------
# 曝光轴已按静照测光常数 K=12.5、EI=100 标准化到绝对物理轴：
#   18% 灰的曝光量 H_gray = K / EI = 12.5/100 = 0.125 lux·s
#   logH_gray = log10(0.125) = -0.90309
# 灰点取三层 D=0.9（正片中灰）处相对轴 x=2.305，数据轴平移 -3.20809；
# 旧模型灰点 1.3088 已弃用（其把 18% 灰放在 D≈2.9，不符正片中灰）。
REF_LOG_E = -0.9030899869919435
# 输入 E_mid 为相对中灰（中灰=1.0）：logH = log10(E_mid) + REF_LOG_E，
# 中灰恒落在特性曲线 stop 0（纯物理中灰锚定，不再需要 CAL 平移）。
# 曝光量下限，防止 log(0)
E_FLOOR = 1e-4
# 曝光偏移（档位）→ log10 曝光单位
LOG10_2 = 0.3010299956639812


# ---------- 数据缓存 ----------
_CHAR = None
_GRAN = None
_DYE = None


def _get_char():
    global _CHAR
    if _CHAR is None:
        _CHAR = _data.get_char_curves()
    return _CHAR


def _get_gran():
    global _GRAN
    if _GRAN is None:
        _GRAN = _data.get_granularity()
    return _GRAN


def _get_dye():
    global _DYE
    if _DYE is None:
        _DYE = _data.get_dye_data()
    return _DYE


# ---------- 光扩散（PSF） ----------
# 0.3.4 邻界测试阶段：基础扩散 = 0.3.3 legacy 核（强度 0 时逐位一致）；
# 邻界增强（密度域 DoG）按强度 0/50%/100% 叠加。v2 宽晕/反射晕/曝光自适应
# 等计算保留在 psf_engine，待后续接线，当前不参与运算。
PX_PER_MM = 100.0
_PSF_V1_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "psf_params_provia100f_v1.npz")
_PSF_ADJ_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "psf_params_provia100f_adj.npz")
_PSF_FIT = None
_PSF_V1 = None
_PSF_KERNEL = None
_PSF_STACK = None


def _get_mtf_layers():
    mtf = _data.get_mtf()
    return {"R": (mtf["freq_raw"], mtf["resp_raw"])}


def _get_psf_fit():
    """密度域邻界拟合参数（a/σ窄/σ宽，基础核=legacy），磁盘缓存。"""
    global _PSF_FIT
    if _PSF_FIT is None:
        from Films import psf_engine
        _PSF_FIT = psf_engine.load_or_fit_adjacency(
            "provia100f", _get_mtf_layers(), PX_PER_MM,
            _PSF_V1_FILE, _PSF_ADJ_FILE)
    return _PSF_FIT


def _get_psf_v1():
    """0.3.3 legacy 拟合参数（基础扩散核），只读缓存。"""
    global _PSF_V1
    if _PSF_V1 is None:
        from Films import psf_engine
        _PSF_V1 = psf_engine.load_legacy_v1(_PSF_V1_FILE)
    return _PSF_V1


def _get_psf_kernel():
    """0.3.3 legacy 核（与关闭开关时的 0.3.3 输出逐位一致）。"""
    global _PSF_KERNEL
    if _PSF_KERNEL is None:
        from Films import psf_engine
        _PSF_KERNEL = psf_engine.build_legacy_kernel(
            _get_psf_v1()["R"], PX_PER_MM)
    return _PSF_KERNEL


def _get_psf_stack():
    """0.3.3 legacy 高斯堆（n_pos=8 + 负邻界），逐位一致。"""
    global _PSF_STACK
    if _PSF_STACK is None:
        from Films import psf_engine
        sigma_adj = _get_psf_v1()["R"][2] * PX_PER_MM
        _PSF_STACK = psf_engine.gauss_decomp_v1(
            _get_psf_kernel(), sigma_adj=sigma_adj)
    return _PSF_STACK


def _apply_psf(e):
    """0.3.3 legacy 高斯堆卷积（邻域开关 OFF 时即原样基线）。"""
    out = np.zeros_like(e)
    for sign, wi, s in _get_psf_stack():
        b = cv2.GaussianBlur(e, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        if sign == "+":
            out += (wi * b).astype(np.float32)
        else:
            out -= (wi * b).astype(np.float32)
    return out


# ---------- 对数曝光与特性曲线 ----------
def _interp_uniform(x, x0, dx, fp):
    """均匀网格线性插值：语义等价 np.interp（两端夹紧），快约 5 倍。"""
    t = (x.astype(np.float64) - x0) / dx
    idx = np.floor(t).astype(np.int32)
    np.clip(idx, 0, len(fp) - 2, out=idx)
    frac = t - idx
    fpd = np.asarray(fp, dtype=np.float64)
    return (fpd[idx] + (fpd[idx + 1] - fpd[idx]) * frac).astype(np.float32)


def _to_log_exposure(e, exposure_offset):
    """线性曝光 → logH（对数曝光量工作区），含曝光偏移（档位）。"""
    e = np.maximum(np.asarray(e, dtype=np.float32), E_FLOOR)
    return np.log10(e) + REF_LOG_E + exposure_offset * LOG10_2


def _char_density(logh_r, logh_g, logh_b):
    """logH → 密度（三层各自查表，负斜率正像曲线）。"""
    char = _get_char()
    logE = char["logE"]
    x0, dx = float(logE[0]), float(logE[1] - logE[0])
    if char.get("logE_uniform", False):
        d_r = _interp_uniform(logh_r, x0, dx, char["density_R"])
        d_g = _interp_uniform(logh_g, x0, dx, char["density_G"])
        d_b = _interp_uniform(logh_b, x0, dx, char["density_B"])
    else:
        d_r = np.interp(logh_r, logE, char["density_R"]).astype(np.float32)
        d_g = np.interp(logh_g, logE, char["density_G"]).astype(np.float32)
        d_b = np.interp(logh_b, logE, char["density_B"]).astype(np.float32)
    return d_r, d_g, d_b


# ---------- 颗粒（三层，密度域，48µm 孔径标定） ----------
# 白噪声 → Cholesky 层间相关（ρ=0.5）→ sign·χ² 伽马聚簇 →
# 可分离高斯组合赋形 → 可选 2x 超采样 → 48µm 孔径闭环标定。
_GRAIN_RHO = 0.5
_GRAIN_COV = None
_GRAIN_SHAPE_EXPONENT = 2.0
_GRAIN_KERNEL_SIGMA = 0.7
_GRAIN_UNSHARP = 0.8
_GRAIN_UNSHARP_SIGMA = 2.0
_GRAIN_CALIB_SEED = 12345
_GRAIN_CALIB_SIZE = 768
_GRAIN_CACHE = {}             # ss -> (aperture, scale)
_GRAIN_CACHE_VERSION = 1
_GRAIN_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "grain_calib_provia100f_ss{}.npz")


def _grain_fingerprint(ss):
    """颗粒标定的参数指纹：任何影响标定的常量变化都会使缓存失效。"""
    return np.array(
        [float(ss), _GRAIN_SHAPE_EXPONENT,
         _GRAIN_KERNEL_SIGMA, _GRAIN_UNSHARP, _GRAIN_UNSHARP_SIGMA,
         float(_GRAIN_CALIB_SEED), float(_GRAIN_CALIB_SIZE),
         float(PX_PER_MM)],
        dtype=np.float64,
    )


def _load_grain_calib(ss):
    """读取颗粒标定磁盘缓存；参数指纹不符返回 None。"""
    path = _GRAIN_CACHE_DIR.format(ss)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    if int(d["cache_version"]) != _GRAIN_CACHE_VERSION:
        return None
    if not np.array_equal(d["fingerprint"], _grain_fingerprint(ss)):
        return None
    return (d["aperture"], float(d["scale"]))


def _save_grain_calib(ss, aperture, scale):
    """把颗粒标定结果写回磁盘缓存（按 ss 分文件）。"""
    path = _GRAIN_CACHE_DIR.format(ss)
    np.savez(path,
             cache_version=np.int32(_GRAIN_CACHE_VERSION),
             fingerprint=_grain_fingerprint(ss),
             aperture=np.asarray(aperture, np.float32),
             scale=np.float64(scale))


def _grain_convolve(g, ss):
    """颗粒成形后卷积：可分离高斯组合（唯一算法）。"""
    s = float(ss)
    sigma1 = _GRAIN_KERNEL_SIGMA * s
    sigma2 = float(np.hypot(_GRAIN_KERNEL_SIGMA, _GRAIN_UNSHARP_SIGMA)) * s
    u = _GRAIN_UNSHARP
    return ((1.0 + u) * cv2.GaussianBlur(g, (0, 0), sigma1)
            - u * cv2.GaussianBlur(g, (0, 0), sigma2)).astype(np.float32)


def _grain_cholesky():
    """三层噪声的协方差 Cholesky 分解（ρ=0.5）。"""
    global _GRAIN_COV
    if _GRAIN_COV is None:
        rho = _GRAIN_RHO
        cov = np.array([[1.0, rho, rho], [rho, 1.0, rho], [rho, rho, 1.0]])
        _GRAIN_COV = np.linalg.cholesky(cov).astype(np.float32)
    return _GRAIN_COV


def _grain_shaped(corr):
    """伽马幅度成形：sign·|z|^p（p=2 即 χ²₁，散粒噪声幅度模型）。"""
    return np.sign(corr) * (np.abs(corr) ** _GRAIN_SHAPE_EXPONENT)


def _grain_parts(ss=2):
    """按超采样级 ss 构建孔径核与标定系数（磁盘缓存）。"""
    ss = int(ss)
    if ss in _GRAIN_CACHE:
        return _GRAIN_CACHE[ss]
    cached = _load_grain_calib(ss)
    if cached is not None:
        _GRAIN_CACHE[ss] = cached
        return cached
    radius = 24.0 * PX_PER_MM / 1000.0
    rr = int(np.ceil(radius))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    aperture = ((xx ** 2 + yy ** 2) <= radius ** 2).astype(np.float32)
    aperture = aperture / aperture.sum()
    c = _GRAIN_CALIB_SIZE
    rng = np.random.default_rng(_GRAIN_CALIB_SEED)
    z = rng.standard_normal((c * ss, c * ss, 3)).astype(np.float32)
    corr = z @ _grain_cholesky().T
    g = _grain_shaped(corr)
    n = _grain_convolve(g, ss)
    n_d = n.reshape(c, ss, c, ss, 3).mean(axis=(1, 3))
    ap = np.empty((c, c, 3), dtype=np.float32)
    for ch in range(3):
        ap[..., ch] = cv2.filter2D(n_d[..., ch], cv2.CV_32F, aperture,
                                   borderType=cv2.BORDER_REFLECT)
    scale = float(1.0 / ap.std())
    result = (aperture.astype(np.float32), scale)
    _GRAIN_CACHE[ss] = result
    _save_grain_calib(ss, aperture, scale)
    return result


def _grain_field(ss, scale, h, w):
    """颗粒场：按超采样级生成（ss×白噪声→成形→块平均降采样）。"""
    if ss == 1:
        z = np.random.randn(h, w, 3).astype(np.float32)
        corr = z @ _grain_cholesky().T
        g = _grain_shaped(corr)
        return _grain_convolve(g, ss) * scale
    hs, ws = h * ss, w * ss
    out = np.empty((h, w, 3), dtype=np.float32)
    s = float(ss)
    sigma_max = float(np.hypot(_GRAIN_KERNEL_SIGMA, _GRAIN_UNSHARP_SIGMA)) * s
    margin = int(np.ceil(3.0 * sigma_max))
    stripe_out = max(1, 1024 // (ss * ss))
    chol = _grain_cholesky()
    for i0 in range(0, h, stripe_out):
        i1 = min(i0 + stripe_out, h)
        r0 = max(i0 * ss - margin, 0)
        r1 = min(i1 * ss + margin, hs)
        z = np.random.randn(r1 - r0, ws, 3).astype(np.float32)
        corr = z @ chol.T
        g = _grain_shaped(corr)
        n = _grain_convolve(g, ss)
        crop = n[(i0 * ss - r0):(i1 * ss - r0)]
        rows = crop.shape[0]
        n_d = crop.reshape(rows // ss, ss, w, ss, 3).mean(axis=(1, 3))
        out[i0:i1] = n_d * scale
    return out


def _grain_sigmas(logh_r, logh_g, logh_b):
    """按 logH 查颗粒曲线，返回三层 σ_D（与曝光相关的钟形调制）。"""
    gran = _get_gran()
    logE_grid = gran["logE"]
    x0, dx = float(logE_grid[0]), float(logE_grid[1] - logE_grid[0])
    if gran.get("logE_uniform", False):
        sigma_r = _interp_uniform(logh_r, x0, dx, gran["sigma_R"])
        sigma_g = _interp_uniform(logh_g, x0, dx, gran["sigma_G"])
        sigma_b = _interp_uniform(logh_b, x0, dx, gran["sigma_B"])
    else:
        sigma_r = np.interp(logh_r, logE_grid, gran["sigma_R"]).astype(np.float32)
        sigma_g = np.interp(logh_g, logE_grid, gran["sigma_G"]).astype(np.float32)
        sigma_b = np.interp(logh_b, logE_grid, gran["sigma_B"]).astype(np.float32)
    return sigma_r, sigma_g, sigma_b


def _apply_grain(d_r, d_g, d_b, logh_r, logh_g, logh_b, ss=2):
    """颗粒：伽马聚簇 + 可分离高斯赋形 + 可选 2x 超采样 + 孔径闭环。"""
    sigma_r, sigma_g, sigma_b = _grain_sigmas(logh_r, logh_g, logh_b)
    aperture, scale = _grain_parts(ss)
    n = _grain_field(ss, scale, d_r.shape[0], d_r.shape[1])
    d_r = d_r + n[..., 0] * sigma_r
    d_g = d_g + n[..., 1] * sigma_g
    d_b = d_b + n[..., 2] * sigma_b
    return d_r, d_g, d_b


# ---------- 染料堆叠 + 客观 CIE-RGB 扫描（正像读数） ----------
_SCAN_WEIGHTS = None
_DYE_TILE = 512
_LN10 = 2.3025850929940455


def _scan_weights():
    """客观扫描仪的 CIE 权重：k·D65·CMF·Δλ，Y 归一化到 1。"""
    global _SCAN_WEIGHTS
    if _SCAN_WEIGHTS is None:
        from Spectral import cie_data
        k = 1.0 / (cie_data.D65_SPD * cie_data.CMF_1931_2[:, 1] * cie_data.DELTA_LAMBDA).sum()
        w = k * cie_data.D65_SPD[:, None] * cie_data.CMF_1931_2 * cie_data.DELTA_LAMBDA
        _SCAN_WEIGHTS = w.astype(np.float32)  # (61,3)：X/Y/Z
    return _SCAN_WEIGHTS


def _dye_stack_scan(d_r, d_g, d_b):
    """染料堆叠 + 客观扫描（分块）：扫描值即正像透光率，无反转。

    D(λ) = (D_B-Dmin_B)·s_y + (D_G-Dmin_G)·s_m + (D_R-Dmin_R)·s_c + D_min(λ)
    T = exp(-ln10·D)；XYZ = Σ T·D65·CMF·Δλ → 线性 sRGB。
    """
    from Spectral import cie_data
    char = _get_char()
    dye = _get_dye()
    s_y = dye["yellow_norm"][:61].astype(np.float32)
    s_m = dye["magenta_norm"][:61].astype(np.float32)
    s_c = dye["cyan_norm"][:61].astype(np.float32)
    dmin = dye["dmin"][:61].astype(np.float32)
    weights = _scan_weights()
    srgb_to_xyz_inv = cie_data.XYZ_TO_SRGB.astype(np.float32).T
    h, w = d_r.shape
    out = np.empty((h, w, 3), dtype=np.float32)
    tile = _DYE_TILE
    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            d_b_t = np.maximum(d_b[y0:y1, x0:x1] - char["dmin_B"], 0.0)[..., None]
            d_g_t = np.maximum(d_g[y0:y1, x0:x1] - char["dmin_G"], 0.0)[..., None]
            d_r_t = np.maximum(d_r[y0:y1, x0:x1] - char["dmin_R"], 0.0)[..., None]
            d_lambda = d_b_t * s_y + d_g_t * s_m + d_r_t * s_c + dmin
            t = np.exp(-_LN10 * d_lambda)
            xyz = t @ weights  # (h,w,61) @ (61,3) → (h,w,3)
            lin_srgb = xyz @ srgb_to_xyz_inv
            out[y0:y1, x0:x1] = np.clip(lin_srgb, 0.0, None)
    return out


# ---------- 主入口（第二段 + 第三段组合） ----------
def process_scan(lux_r, lux_g, lux_b, grain_ss=2, exposure_offset=0.0):
    """第二段：线性光照度 → 正像扫描（float32 透光率语义，无需反转）。

    处理链（0.3.4）：
    光谱输入 → PSF（v2 引擎，选项见 psf_config）→ [曝光自适应晕] →
    logH → 特性曲线 → [密度域邻界增强] → 密度域颗粒 →
    染料堆叠 + 客观 CIE-RGB 扫描（正像）。
    """
    from Films import psf_config, psf_engine
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")
    if grain_ss not in (1, 2):
        raise ValueError("grain_ss 只支持 1 / 2")

    lux_r = np.asarray(lux_r, dtype=np.float32)
    lux_g = np.asarray(lux_g, dtype=np.float32)
    lux_b = np.asarray(lux_b, dtype=np.float32)

    # 1. 光谱输入（固定开启）：光谱重建 RGB→三层曝光
    from Spectral import migrate_rgb_primary
    e_r, e_g, e_b = migrate_rgb_primary(lux_r, lux_g, lux_b, film="provia100f")

    # 2. PSF 光扩散（单层核三层共用，MTF 拟合）
    e_r = _apply_psf(e_r)
    e_g = _apply_psf(e_g)
    e_b = _apply_psf(e_b)

    # 3. 对数曝光工作区
    logh_r = _to_log_exposure(e_r, exposure_offset)
    logh_g = _to_log_exposure(e_g, exposure_offset)
    logh_b = _to_log_exposure(e_b, exposure_offset)

    # 4. 特性曲线 → Status M 密度（负斜率：高曝光 → 低密度）
    d_r, d_g, d_b = _char_density(logh_r, logh_g, logh_b)

    # 4.5 密度域邻域增强（复合核 FFT；零 DC，平场不变）
    if psf_config.get()["adjacency_strength"] > 0:
        params = psf_engine.apply_overrides(_get_psf_fit()["R"],
                                            psf_config.get())
        d_r = psf_engine.apply_adjacency(d_r, params, PX_PER_MM)
        d_g = psf_engine.apply_adjacency(d_g, params, PX_PER_MM)
        d_b = psf_engine.apply_adjacency(d_b, params, PX_PER_MM)

    # 5. 密度域颗粒（伽马聚簇+可分离高斯赋形+超采样+孔径闭环）
    d_r, d_g, d_b = _apply_grain(d_r, d_g, d_b, logh_r, logh_g, logh_b,
                                   ss=grain_ss)

    # 6. 输出端：染料光谱堆叠 + CIE-RGB 客观扫描（正像，无需反转）
    return _dye_stack_scan(d_r, d_g, d_b)


def process(lux_r, lux_g, lux_b, grain_ss=2, exposure_offset=0.0,
            contrast=0.0, return_scan=False):
    """便捷入口：第二段 + 第三段（0.3.3 三段式）。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（第一段 Input 输出）。
        grain_ss：颗粒超采样级（1=1x 不超采样，2=2x；默认 2）。
        exposure_offset：曝光偏移（档位，正=过曝）。
        contrast：风格对比度（±100%，0 线性；三锚点已保证曝光正确）。
        return_scan：True 时同时返回 (成片 uint8, 正像扫描 float32)。
    返回：
        (H,W,3) uint8（RGB）；return_scan=True 时返回 (uint8, scan)。
    """
    scan = process_scan(lux_r, lux_g, lux_b, grain_ss, exposure_offset)
    from Scanner import render
    out = render(scan, "provia100f", contrast=contrast)
    if return_scan:
        return out, scan
    return out
