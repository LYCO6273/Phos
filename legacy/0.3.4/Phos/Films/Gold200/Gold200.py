#赛博请神
# Gold200.py —— KODAK Gold 200（E-7022，C-41 入门彩负）光谱模型
# 结构与 5213 完全一致（彩负统一设计思路）：光谱输入 → 物理 PSF →
# 对数曝光 → 特性曲线 → 密度域颗粒 → 染料堆叠 + 客观扫描
# （第二段）→ 第三段 Scanner 三锚点平滑校正 + 风格对比度。
#
# 处理链：
#   线性 RGB
#     → 光谱输入（固定）光谱重建迁移 → 层曝光 E_R/E_G/E_B
#     → [PSF] 物理核（核心高斯 + K0 扩散晕 − 邻界高斯，MTF 拟合）
#     → 颗粒：密度域伽马聚簇+可分离高斯赋形+可选 2x 超采样
#     → log10 层曝光（统一对数工作区 + 曝光偏移）
#     → 特性曲线 → Status M 密度
#     → 染料光谱堆叠（固定开启）→ CIE-RGB 客观扫描（负片读数）
#     → [第三段 Scanner] 反转：三锚点平滑校正（黑/灰/白）→ 对比度 → sRGB
#
# 层映射约定：
#   lux_r = 红感光层（青染料）  lux_g = 绿感光层（品红染料）  lux_b = 蓝感光层（黄染料）

import numpy as np
import cv2
import os

from Spectral import cie_data

import Films.Gold200.Gold200_data as _data

# 模型目录 / 数据目录（磁盘缓存与数据文件所在）
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_MODEL_DIR, "data")


# ---------- 曝光标定常量 ----------
# 曝光轴已按静照测光常数 K=12.5、EI=200 标准化到绝对物理轴：
#   18% 灰的曝光量 H_gray = K / EI = 12.5/200 = 0.0625 lux·s
#   logH_gray = log10(0.0625) = -1.20412
# 旧相对轴（沿用 Gold200 老模型）把 18% 灰放在 1.582 处，故数据轴
# 整体平移 -2.78612，灰点密度不变（R≈0.77/G≈1.22/B≈1.54）。
REF_LOG_E = -1.20412
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


# ---------- 光扩散（PSF，线性物理模式，唯一路径） ----------
# 物理核 = 核心高斯 + K0 扩散晕 − 邻界高斯，由 datasheet MTF 逐层频域拟合：
#   K0(r/δ) 是三维扩散格林函数 e^(-r/δ)/r 沿膜厚的精确投影（重尾 1/r·e^(-r/δ)），
#   其频域闭式为洛伦兹 1/(1+(2πδf)²)；负邻界高斯负责 MTF 中频 >100% 的响应。
# 像素-毫米声明：1 图像像素 = 10µm 胶片平面（35mm 全幅 ≈ 3600px 宽）。
# MTF 拟合出的 σ/δ 单位是 mm，乘以 PX_PER_MM 即得像素；
# 颗粒的 48µm 孔径换算也使用同一声明。
PX_PER_MM = 100.0

# 物理核参数搜索范围（µm；δ 上限受 MTF 最低频 2.5cyc/mm 约束：δ ≲ 1/(2π·2.5) ≈ 64µm）

_PSF_KERNELS = {}
_PSF_FIT = None
_PSF_V1 = None


_PSF_V1_FILE = os.path.join(_DATA_DIR, "psf_params_gold200_v1.npz")
_PSF_ADJ_FILE = os.path.join(_DATA_DIR, "psf_params_gold200_adj.npz")


def _get_mtf_layers():
    mtf = _data.get_mtf()
    return {
        "R": (mtf["freq_R_raw"], mtf["resp_R_raw"]),
        "G": (mtf["freq_G_raw"], mtf["resp_G_raw"]),
        "B": (mtf["freq_B_raw"], mtf["resp_B_raw"]),
    }


def _get_psf_fit():
    """密度域邻界拟合参数（每层 a/σ窄/σ宽，基础核=legacy），磁盘缓存。"""
    global _PSF_FIT
    if _PSF_FIT is None:
        from Films import psf_engine
        _PSF_FIT = psf_engine.load_or_fit_adjacency(
            "gold200", _get_mtf_layers(), PX_PER_MM,
            _PSF_V1_FILE, _PSF_ADJ_FILE)
    return _PSF_FIT


def _get_psf_v1():
    """0.3.3 legacy 拟合参数（基础扩散核），只读缓存。"""
    global _PSF_V1
    if _PSF_V1 is None:
        from Films import psf_engine
        _PSF_V1 = psf_engine.load_legacy_v1(_PSF_V1_FILE)
    return _PSF_V1


def _get_psf_kernel(layer):
    """0.3.3 legacy 核（按层，与关闭开关时的 0.3.3 输出逐位一致）。"""
    key = ("legacy", layer)
    if key in _PSF_KERNELS:
        return _PSF_KERNELS[key]
    from Films import psf_engine
    kern = psf_engine.build_legacy_kernel(
        _get_psf_v1()[layer], PX_PER_MM)
    _PSF_KERNELS[key] = kern
    return kern


_PSF_STACK = {}


def _get_psf_stack(layer):
    """0.3.3 legacy 高斯堆（n_pos=8 + 负邻界），逐位一致。"""
    key = ("legacy", layer)
    if key in _PSF_STACK:
        return _PSF_STACK[key]
    from Films import psf_engine
    sigma_adj = _get_psf_v1()[layer][2] * PX_PER_MM
    _PSF_STACK[key] = psf_engine.gauss_decomp_v1(
        _get_psf_kernel(layer), sigma_adj=sigma_adj)
    return _PSF_STACK[key]


def _apply_psf(e_r, e_g, e_b):
    """0.3.3 legacy 高斯堆卷积（逐层，邻域开关 OFF 时即原样基线）。"""
    def stack_apply(e, layer):
        out = np.zeros_like(e)
        for sign, wi, s in _get_psf_stack(layer):
            b = cv2.GaussianBlur(e, (0, 0), s, borderType=cv2.BORDER_REFLECT)
            if sign == "+":
                out += (wi * b).astype(np.float32)
            else:
                out -= (wi * b).astype(np.float32)
        return out

    return (stack_apply(e_r, "R"), stack_apply(e_g, "G"), stack_apply(e_b, "B"))


# ---------- 颗粒（密度域，48µm 孔径标定） ----------
# ① 白噪声 → Cholesky 层间相关（ρ=0.5）；
# ② sign·χ² 伽马聚簇成形——化合物-泊松散粒噪声的幅度模型，
#    产生非高斯重尾团块（χ² 分布即散粒噪声幅度模型）；
# ③ 可分离高斯组合赋形（平滑 σ1 + unsharp 高频“咬合”）；
# ④ 可选 2x 超采样：高分辨率生成 → 块平均降采样，
#    模拟“细颗粒被像素积分”的物理过程，并消除网格采样伪影；
# ⑤ Monte Carlo 做 48µm 孔径闭环标定，宏观 σ 严格回到 datasheet。
_GRAIN_RHO = 0.5
_GRAIN_COV = None
# 颗粒参数
_GRAIN_SHAPE_EXPONENT = 2.0   # 伽马幅度成形：sign·|z|^p（p=2 即 χ²₁）
_GRAIN_KERNEL_SIGMA = 0.7     # 平滑赋形核 σ（px，显影扩散尺度）
_GRAIN_UNSHARP = 0.8          # unsharp 强度（高频“咬合”）
_GRAIN_UNSHARP_SIGMA = 2.0    # unsharp 模糊宽度（px）
_GRAIN_CALIB_SEED = 12345
_GRAIN_CALIB_SIZE = 768
_GRAIN_CACHE = {}             # ss -> (aperture, scale)
_GRAIN_CACHE_VERSION = 2
_GRAIN_CACHE_DIR = os.path.join(_DATA_DIR, "grain_calib_gold200_ss{}.npz")


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
    宏观 σ 由蒙特卡洛闭环标定保证，与 datasheet 严格一致。
    """
    s = float(ss)
    sigma1 = _GRAIN_KERNEL_SIGMA * s
    sigma2 = float(np.hypot(_GRAIN_KERNEL_SIGMA, _GRAIN_UNSHARP_SIGMA)) * s
    u = _GRAIN_UNSHARP
    return ((1.0 + u) * cv2.GaussianBlur(g, (0, 0), sigma1)
            - u * cv2.GaussianBlur(g, (0, 0), sigma2)).astype(np.float32)


def _grain_cholesky():
    """三层噪声的协方差 Cholesky 分解（ρ=0.5，亮度颗粒为主、含色度颗粒）。"""
    global _GRAIN_COV
    if _GRAIN_COV is None:
        rho = _GRAIN_RHO
        cov = np.array([[1.0, rho, rho], [rho, 1.0, rho], [rho, rho, 1.0]])
        _GRAIN_COV = np.linalg.cholesky(cov).astype(np.float32)
    return _GRAIN_COV


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


def _grain_shaped(corr):
    """伽马幅度成形：sign·|z|^p（p=2 即 χ²₁，散粒噪声幅度模型）。

    每层银晶体的染料云大小随机，密度涨落近似“化合物-泊松”
    散粒噪声，幅度分布常用伽马分布近似；sign·χ² 是零均值、
    重尾的实现——偶发强颗粒产生聚簇感，且没有点过程印章。
    """
    return np.sign(corr) * (np.abs(corr) ** _GRAIN_SHAPE_EXPONENT)


def _grain_parts(ss=1):
    """按超采样级 ss 构建孔径核与标定系数（磁盘缓存）。

    返回 (aperture, scale)；卷积统一走 _grain_convolve 的可分离高斯
    组合，标定闭环同路径，保证 scale 与运行时一致。
    """
    ss = int(ss)
    if ss in _GRAIN_CACHE:
        return _GRAIN_CACHE[ss]
    cached = _load_grain_calib(ss)
    if cached is not None:
        _GRAIN_CACHE[ss] = cached
        return cached

    s = float(ss)
    # 48µm 环形孔径 → 像素圆盘核（最终分辨率）
    radius = 24.0 * PX_PER_MM / 1000.0
    rr = int(np.ceil(radius))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    aperture = ((xx ** 2 + yy ** 2) <= radius ** 2).astype(np.float32)
    aperture = aperture / aperture.sum()
    # Monte Carlo 孔径闭环标定：ss× 生成 → 成形 → 块平均降采样 → 孔径平均
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
    """颗粒场：按超采样级生成（ss×白噪声→成形→块平均降采样）。

    ss=1 直接在本分辨率生成；ss>1 分条带处理控制内存，
    条带边界用反射余量保证卷积正确。
    """
    if ss == 1:
        z = np.random.randn(h, w, 3).astype(np.float32)
        corr = z @ _grain_cholesky().T
        g = _grain_shaped(corr)
        return _grain_convolve(g, ss) * scale

    hs, ws = h * ss, w * ss
    out = np.empty((h, w, 3), dtype=np.float32)
    # 可分离高斯组合的最大 3σ 截断半径（条带边界反射余量）
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


def _apply_grain(d_r, d_g, d_b, logh_r, logh_g, logh_b, ss=1):
    """颗粒：伽马聚簇 + 可分离高斯赋形 + 可选 2x 超采样 + 孔径闭环。"""
    sigma_r, sigma_g, sigma_b = _grain_sigmas(logh_r, logh_g, logh_b)
    aperture, scale = _grain_parts(ss)
    n = _grain_field(ss, scale, d_r.shape[0], d_r.shape[1])
    d_r = d_r + n[..., 0] * sigma_r
    d_g = d_g + n[..., 1] * sigma_g
    d_b = d_b + n[..., 2] * sigma_b
    return d_r, d_g, d_b


# ---------- 对数曝光与特性曲线 ----------
def _interp_uniform(x, x0, dx, fp):
    """均匀网格线性插值：语义等价 np.interp（两端夹紧），快约 5 倍。

    数据已统一插值到严格等距的 logE 轴（见 *_data.py）；x 允许超出
    网格范围，两端自动夹到 fp[0]/fp[-1]。内部用 float64 计算，
    与 np.interp 的数值差异 < 1e-6（远小于 8bit 量化步长）。
    """
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
    """logH → Status M 密度（三层各自查表，np.interp 自动夹两端）。"""
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


# ---------- 染料堆叠 + 客观 CIE-RGB 扫描（负片读数） ----------
_SCAN_WEIGHTS = None
_DYE_TILE = 512
_LN10 = 2.3025850929940455  # ln(10)，透光率 T=exp(-ln10·D) 用

def _scan_weights():
    """客观扫描仪的 CIE 权重：k·D65·CMF·Δλ，Y 归一化到 1。"""
    global _SCAN_WEIGHTS
    if _SCAN_WEIGHTS is None:
        k = 1.0 / (cie_data.D65_SPD * cie_data.CMF_1931_2[:, 1] * cie_data.DELTA_LAMBDA).sum()
        w = k * cie_data.D65_SPD[:, None] * cie_data.CMF_1931_2 * cie_data.DELTA_LAMBDA
        _SCAN_WEIGHTS = w.astype(np.float32)  # (61,3)：X/Y/Z
    return _SCAN_WEIGHTS


def _dye_stack_scan(d_r, d_g, d_b):
    """染料堆叠 + 客观扫描（分块，避免大图一次性生成 61 波段中间量）。

    D(λ) = (D_B-Dmin_B)·s_y + (D_G-Dmin_G)·s_m + (D_R-Dmin_R)·s_c + D_min(λ)
    T(λ) = 10^(-D(λ))
    扫描：XYZ = Σ T·D65·CMF·Δλ → 线性 sRGB（负片读数，不在这里反转）。
    """
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
            # 净染料量 = 层密度 - 该层 D-min（D-min 已经包含在光谱 D-min 里）
            d_b_t = np.maximum(d_b[y0:y1, x0:x1] - char["dmin_B"], 0.0)[..., None]
            d_g_t = np.maximum(d_g[y0:y1, x0:x1] - char["dmin_G"], 0.0)[..., None]
            d_r_t = np.maximum(d_r[y0:y1, x0:x1] - char["dmin_R"], 0.0)[..., None]
            d_lambda = d_b_t * s_y + d_g_t * s_m + d_r_t * s_c + dmin
            # exp(-ln10·D) 代替 10**(-D)：快约 1.8 倍，误差 ~6e-8（float32 精度内）
            t = np.exp(-_LN10 * d_lambda)
            # 一次矩阵乘代替三次逐列加权和：快约 10 倍，浮点差异 ~1e-5
            xyz = t @ weights  # (h,w,61) @ (61,3) → (h,w,3)
            lin_srgb = xyz @ srgb_to_xyz_inv
            out[y0:y1, x0:x1] = np.clip(lin_srgb, 0.0, None)
    return out


# ---------- 主入口（第二段 + 第三段组合） ----------
def process_scan(lux_r, lux_g, lux_b, grain_ss=2, exposure_offset=0.0):
    """第二段：线性光照度 → 未反转负片扫描（float32 透光率语义）。

    处理链（0.3.4）：光谱输入 → PSF（v2 引擎，逐层）→ [曝光自适应晕] →
    logH → 特性曲线 → [密度域邻界增强] → 密度域颗粒 → 染料堆叠 + 客观扫描。
    返回的 scan 即"扫描仪看到的底片"，供第三段渲染成片。
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
    e_r, e_g, e_b = migrate_rgb_primary(lux_r, lux_g, lux_b, film="gold200")

    # 2. PSF 光扩散（线性物理模式：核心高斯 + K0 扩散晕 − 邻界，MTF 拟合）
    e_r, e_g, e_b = _apply_psf(e_r, e_g, e_b)

    # 3. 对数曝光工作区
    logh_r = _to_log_exposure(e_r, exposure_offset)
    logh_g = _to_log_exposure(e_g, exposure_offset)
    logh_b = _to_log_exposure(e_b, exposure_offset)

    # 4. 特性曲线 → Status M 密度
    d_r, d_g, d_b = _char_density(logh_r, logh_g, logh_b)

    # 4.5 密度域邻域增强（复合核 FFT，逐层；零 DC，平场不变）
    if psf_config.get()["adjacency_strength"] > 0:
        fit = _get_psf_fit()
        d_r = psf_engine.apply_adjacency(
            d_r, psf_engine.apply_overrides(fit["R"], psf_config.get()),
            PX_PER_MM)
        d_g = psf_engine.apply_adjacency(
            d_g, psf_engine.apply_overrides(fit["G"], psf_config.get()),
            PX_PER_MM)
        d_b = psf_engine.apply_adjacency(
            d_b, psf_engine.apply_overrides(fit["B"], psf_config.get()),
            PX_PER_MM)

    # 5. 密度域颗粒（伽马聚簇+可分离高斯赋形+超采样+孔径闭环）
    d_r, d_g, d_b = _apply_grain(d_r, d_g, d_b, logh_r, logh_g, logh_b,
                                   ss=grain_ss)

    # 6. 输出端：染料光谱堆叠 + CIE-RGB 客观扫描（负片读数，不反转）
    return _dye_stack_scan(d_r, d_g, d_b)


def process(lux_r, lux_g, lux_b, grain_ss=2, exposure_offset=0.0,
            contrast=0.0, return_scan=False):
    """便捷入口：第二段 + 第三段（0.3.3 三段式）。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（第一段 Input 输出）。
        grain_ss：颗粒超采样级（1=1x 不超采样，2=2x；默认 2）。
        exposure_offset：曝光偏移（档位，正=过曝）。
        contrast：风格对比度（±100%，0 线性；三锚点已保证曝光正确）。
        return_scan：True 时同时返回 (成片 uint8, 未反转扫描 float32)。
    返回：
        (H,W,3) uint8（RGB）；return_scan=True 时返回 (uint8, scan)。
    """
    scan = process_scan(lux_r, lux_g, lux_b, grain_ss, exposure_offset)
    from Scanner import render
    out = render(scan, "gold200", contrast=contrast)
    if return_scan:
        return out, scan
    return out
