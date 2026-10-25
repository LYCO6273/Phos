#赛博请神
# HP5.py —— ILFORD HP5 Plus（ISO 400 黑白负片）光谱模型
# 黑白负片 = 彩负的单层简化（见 STANDARD_FLOW.md 第二部分）：
#   线性 RGB
#     → 光谱输入（固定）光谱重建 → 实测灵敏度吸收积分（单通道）
#     → [PSF] 物理核（核心高斯 + K0 扩散晕 − 邻界高斯，MTF 拟合）
#     → 对数曝光（统一对数工作区 + 曝光偏移）
#     → 特性曲线 → 密度
#     → 密度域颗粒（σ 按密度查表，rms 曲线 X 轴=密度）
#     → [第三段 Scanner] 反转：三锚点平滑校正（黑/灰/白）→ 对比度 → uint8
#
# 数据：特性曲线/颗粒度为 HP5 实测描点；MTF 用 KODAK TRI-X 400 代理
#       （同级 ISO 400 黑白卷，HP5 文档无 MTF）。

import os

import cv2
import numpy as np

import Films.HP5.HP5_data as _data


# ---------- 曝光标定常量 ----------
# 曝光轴已按静照测光常数 K=12.5、EI=400 标准化到绝对物理轴：
#   18% 灰的曝光量 H_gray = K / EI = 12.5/400 = 0.03125 lux·s
#   logH_gray = log10(0.03125) = -1.50515
# 旧相对轴灰点（老模型 lux=0.18 映射位置）为 1.9896，数据轴平移 -3.49475。
REF_LOG_E = -1.5051499783199058
# 输入 E_mid 为相对中灰（中灰=1.0）：logH = log10(E_mid) + REF_LOG_E，
# 中灰恒落在特性曲线 stop 0（纯物理中灰锚定，不再需要 CAL 平移）。
# 曝光量下限，防止 log(0)
E_FLOOR = 1e-4
# 曝光偏移（档位）→ log10 曝光单位
LOG10_2 = 0.3010299956639812


# ---------- 数据缓存 ----------
_CHAR = None
_GRAN = None


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


# ---------- 光扩散（PSF） ----------
# 0.3.4 邻界测试阶段：基础扩散 = 0.3.3 legacy 核（强度 0 时逐位一致）；
# 邻界增强（密度域 DoG）按强度 0/50%/100% 叠加；v2 宽晕等保留在
# psf_engine 待接线。
PX_PER_MM = 100.0
_PSF_FIT = None
_PSF_V1 = None
_PSF_KERNEL = None
_PSF_STACK = None

_PSF_V1_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "psf_params_hp5_v1.npz")
_PSF_ADJ_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "psf_params_hp5_adj.npz")


def _get_mtf_layers():
    mtf = _data.get_mtf()
    return {"R": (mtf["freq_raw"], mtf["resp_raw"])}


def _get_psf_fit():
    """密度域邻界拟合参数（a/σ窄/σ宽，基础核=legacy），磁盘缓存。"""
    global _PSF_FIT
    if _PSF_FIT is None:
        from Films import psf_engine
        _PSF_FIT = psf_engine.load_or_fit_adjacency(
            "hp5", _get_mtf_layers(), PX_PER_MM,
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


# ---------- 对数曝光与特性曲线（单层） ----------
def _interp_uniform(x, x0, dx, fp):
    """均匀网格线性插值：语义等价 np.interp（两端夹紧），快约 5 倍。

    数据已统一插值到严格等距网格；内部用 float64 计算，
    与 np.interp 的数值差异 < 1e-6。
    """
    t = (x.astype(np.float64) - x0) / dx
    idx = np.floor(t).astype(np.int32)
    np.clip(idx, 0, len(fp) - 2, out=idx)
    frac = t - idx
    fpd = np.asarray(fp, dtype=np.float64)
    return (fpd[idx] + (fpd[idx + 1] - fpd[idx]) * frac).astype(np.float32)


def _to_log_exposure(e, exposure_offset):
    """线性亮度 → logH（对数曝光量工作区），含曝光偏移（档位）。"""
    e = np.maximum(np.asarray(e, dtype=np.float32), E_FLOOR)
    return np.log10(e) + REF_LOG_E + exposure_offset * LOG10_2


def _char_density(logh):
    """logH → 密度（均匀网格快速插值，两端夹紧）。"""
    char = _get_char()
    logE = char["logE"]
    x0, dx = float(logE[0]), float(logE[1] - logE[0])
    if char.get("logE_uniform", False):
        return _interp_uniform(logh, x0, dx, char["density"])
    return np.interp(logh, logE, char["density"]).astype(np.float32)


# ---------- 颗粒（单层，密度域，48µm 孔径标定） ----------
# ① 白噪声 → sign·χ² 伽马聚簇成形（化合物-泊松散粒噪声幅度模型）；
# ② 可分离高斯组合赋形（平滑 σ1 + unsharp 高频“咬合”）；
# ③ 可选 2x 超采样：高分辨率生成 → 块平均降采样；
# ④ Monte Carlo 做 48µm 孔径闭环标定，宏观 σ 严格回到 datasheet。
# 黑白 rms 颗粒度曲线 X 轴=净密度，颗粒 σ 直接按密度查表。
_GRAIN_SHAPE_EXPONENT = 2.0   # 伽马幅度成形：sign·|z|^p（p=2 即 χ²₁）
_GRAIN_KERNEL_SIGMA = 0.7     # 平滑赋形核 σ（px，显影扩散尺度）
_GRAIN_UNSHARP = 0.8          # unsharp 强度（高频“咬合”）
_GRAIN_UNSHARP_SIGMA = 2.0    # unsharp 模糊宽度（px）
_GRAIN_CALIB_SEED = 12345
_GRAIN_CALIB_SIZE = 768
_GRAIN_CACHE = {}             # ss -> (aperture, scale)
_GRAIN_CACHE_VERSION = 1
_GRAIN_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "grain_calib_hp5_ss{}.npz")


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
    """读取颗粒标定磁盘缓存；参数指纹不符返回 None（触发重新标定）。"""
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
    """颗粒成形后卷积：可分离高斯组合（唯一算法）。

    平滑赋形 + unsharp 在图像域合并为两个可分离高斯模糊：
    k_boost = (1+u)·g(σ1) − u·g(√(σ1²+σ2²))，DC 增益恒为 1；
    宏观 σ 由蒙特卡洛闭环标定保证。
    """
    s = float(ss)
    sigma1 = _GRAIN_KERNEL_SIGMA * s
    sigma2 = float(np.hypot(_GRAIN_KERNEL_SIGMA, _GRAIN_UNSHARP_SIGMA)) * s
    u = _GRAIN_UNSHARP
    return ((1.0 + u) * cv2.GaussianBlur(g, (0, 0), sigma1)
            - u * cv2.GaussianBlur(g, (0, 0), sigma2)).astype(np.float32)


def _grain_parts(ss=2):
    """按超采样级 ss 构建孔径核与标定系数（磁盘缓存）。"""
    ss = int(ss)
    if ss in _GRAIN_CACHE:
        return _GRAIN_CACHE[ss]
    cached = _load_grain_calib(ss)
    if cached is not None:
        _GRAIN_CACHE[ss] = cached
        return cached
    # 48µm 环形孔径 → 像素圆盘核（最终分辨率）
    radius = 24.0 * PX_PER_MM / 1000.0
    rr = int(np.ceil(radius))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    aperture = ((xx ** 2 + yy ** 2) <= radius ** 2).astype(np.float32)
    aperture = aperture / aperture.sum()
    # Monte Carlo 孔径闭环标定：ss× 生成 → 成形 → 卷积 → 块平均 → 孔径平均
    c = _GRAIN_CALIB_SIZE
    rng = np.random.default_rng(_GRAIN_CALIB_SEED)
    z = rng.standard_normal((c * ss, c * ss)).astype(np.float32)
    g = np.sign(z) * (np.abs(z) ** _GRAIN_SHAPE_EXPONENT)
    n = _grain_convolve(g, ss)
    n_d = n.reshape(c, ss, c, ss).mean(axis=(1, 3))
    ap = cv2.filter2D(n_d, cv2.CV_32F, aperture,
                      borderType=cv2.BORDER_REFLECT)
    scale = float(1.0 / ap.std())
    result = (aperture.astype(np.float32), scale)
    _GRAIN_CACHE[ss] = result
    _save_grain_calib(ss, aperture, scale)
    return result


def _grain_field(ss, scale, h, w):
    """颗粒场：按超采样级生成（ss×白噪声→成形→块平均降采样）。

    ss=1 直接在本分辨率生成；ss>1 分条带处理控制内存。
    """
    if ss == 1:
        z = np.random.randn(h, w).astype(np.float32)
        g = np.sign(z) * (np.abs(z) ** _GRAIN_SHAPE_EXPONENT)
        return _grain_convolve(g, ss) * scale

    hs, ws = h * ss, w * ss
    out = np.empty((h, w), dtype=np.float32)
    s = float(ss)
    sigma_max = float(np.hypot(_GRAIN_KERNEL_SIGMA, _GRAIN_UNSHARP_SIGMA)) * s
    margin = int(np.ceil(3.0 * sigma_max))
    stripe_out = max(1, 1024 // (ss * ss))
    for i0 in range(0, h, stripe_out):
        i1 = min(i0 + stripe_out, h)
        r0 = max(i0 * ss - margin, 0)
        r1 = min(i1 * ss + margin, hs)
        z = np.random.randn(r1 - r0, ws).astype(np.float32)
        g = np.sign(z) * (np.abs(z) ** _GRAIN_SHAPE_EXPONENT)
        n = _grain_convolve(g, ss)
        crop = n[(i0 * ss - r0):(i1 * ss - r0)]
        rows = crop.shape[0]
        out[i0:i1] = crop.reshape(rows // ss, ss, w, ss).mean(axis=(1, 3)) * scale
    return out


def _grain_sigma(density):
    """按密度查颗粒曲线，返回 σD（黑白 rms 曲线 X 轴=净密度）。"""
    gran = _get_gran()
    gden = gran["density"]
    x0, dx = float(gden[0]), float(gden[1] - gden[0])
    if gran.get("density_uniform", False):
        return _interp_uniform(density, x0, dx, gran["sigma_D"])
    return np.interp(density, gden, gran["sigma_D"]).astype(np.float32)


def _apply_grain(d, ss=2):
    """颗粒：伽马聚簇 + 可分离高斯赋形 + 可选 2x 超采样 + 孔径闭环。

    σ 按密度查表（HP5 rms 曲线 X 轴=密度），密度域直接叠加。
    """
    sigma = _grain_sigma(d)
    aperture, scale = _grain_parts(ss)
    n = _grain_field(ss, scale, d.shape[0], d.shape[1])
    return d + n * sigma


# ---------- 主入口（第二段 + 第三段组合） ----------
def process_scan(lux_r, lux_g, lux_b, grain_ss=2, exposure_offset=0.0):
    """第二段：线性光照度 → 密度（float32 单通道）。

    黑白无染料层，密度即"扫描读数"；处理链（0.3.4 邻界测试阶段）：
    光谱输入 → PSF（0.3.3 legacy）→ logH → 特性曲线 →
    [密度域邻界增强（开关）] → 密度域颗粒。
    """
    from Films import psf_config, psf_engine
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")
    if grain_ss not in (1, 2):
        raise ValueError("grain_ss 只支持 1 / 2")

    lux_r = np.asarray(lux_r, dtype=np.float32)
    lux_g = np.asarray(lux_g, dtype=np.float32)
    lux_b = np.asarray(lux_b, dtype=np.float32)

    # 1. 光谱输入（固定开启）：光谱重建 RGB→单通道曝光
    from Spectral import migrate_bw_primary
    e0 = migrate_bw_primary(lux_r, lux_g, lux_b, film="hp5")

    # 2. PSF 光扩散（单层物理核，Tri-X 400 MTF 拟合）
    e = _apply_psf(e0)

    # 3. 对数曝光工作区
    logh = _to_log_exposure(e, exposure_offset)

    # 4. 特性曲线 → 密度
    d = _char_density(logh)

    # 4.5 密度域邻域增强（复合核 FFT；零 DC，平场不变）
    if psf_config.get()["adjacency_strength"] > 0:
        params = psf_engine.apply_overrides(_get_psf_fit()["R"],
                                            psf_config.get())
        d = psf_engine.apply_adjacency(d, params, PX_PER_MM)

    # 5. 密度域颗粒（σ 按密度查表 + 可选 2x 超采样）
    return _apply_grain(d, ss=grain_ss)


def process(lux_r, lux_g, lux_b, grain_ss=2, exposure_offset=0.0,
            contrast=0.0):
    """便捷入口：第二段 + 第三段（0.3.3 三段式）。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（第一段 Input 输出）。
        grain_ss：颗粒超采样级（1=1x 不超采样，2=2x；默认 2）。
        exposure_offset：曝光偏移（档位，正=过曝）。
        contrast：风格对比度（±100%，0 线性；三锚点已保证曝光正确）。
    返回：
        (H,W) uint8 单通道成片。
    """
    density = process_scan(lux_r, lux_g, lux_b, grain_ss, exposure_offset)
    from Scanner import render
    return render(density, "hp5", contrast=contrast)
