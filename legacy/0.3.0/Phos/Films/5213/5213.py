#赛博请神
# 5213.py —— Vision3 200T（5213/7213）光谱实验模型
# 这是 0.3.0 唯一承载新特性的胶片模型；HP5 / Gold200 保持不变。
#
# 处理链：
#   线性 RGB
#     → [开关:光谱输入，默认开] CIE-RGB 色匹配函数伪发射迁移 → 层曝光 E_R/E_G/E_B
#     → [开关:PSF] MTF 拟合核（new）/ 猜测双高斯（old）
#     → 颗粒：密度域伽马聚簇+平滑赋形+unsharp+可选超采样
#     → log10 层曝光（统一对数工作区 + 曝光偏移）
#     → 特性曲线 → Status M 密度
#     → 染料光谱堆叠（固定开启）→ CIE-RGB 客观扫描（负片读数）
#     → 反转：对数密度域三点锚定 + 小剂量中灰白平衡 + 全局斜率
#       + 趾肩 S 曲线 → 伽马 → sRGB 输出
#
# 层映射约定：
#   lux_r = 红感光层（青染料）  lux_g = 绿感光层（品红染料）  lux_b = 蓝感光层（黄染料）

import numpy as np
import cv2
import importlib

from Spectral import cie_data

# 数据模块名以数字开头，普通 import 语法不可用，这里用 importlib 加载
_data = importlib.import_module("Films.5213.5213_data")


# ---------- 曝光标定常量 ----------
# 特性曲线参考：0 档 = logE = -1.284 lux-seconds（3200K、1/50s）
REF_LOG_E = -1.284
# 默认曝光参考：18% 灰落在这条 logE 上
GRAY_E = 0.18
# logH = log10(E) + CAL，使得 log10(0.18) + CAL = REF_LOG_E
CAL = REF_LOG_E - np.log10(GRAY_E)
# 曝光量下限，防止 log(0)
E_FLOOR = 1e-4
# 曝光偏移（档位）→ log10 曝光单位
LOG10_2 = 0.3010299956639812
# 输出伽马（反转后叠的显示伽马）
OUT_GAMMA = 1.60


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


# ---------- 光扩散（PSF，线性物理模式） ----------
# 两种模式：
#   new：由 datasheet MTF 曲线逐层频域拟合（核心+翼-邻界，系数非负）；
#   old：0.3.0 早期猜测的双高斯（窄核+宽翼，翼权重按层不同）。
# 像素-毫米声明：1 图像像素 = 10µm 胶片平面（35mm 全幅 ≈ 3600px 宽）。
# MTF 拟合出的 σ 单位是 mm，乘以 PX_PER_MM 即得像素 σ；
# 颗粒的 48µm 孔径换算也使用同一声明。
PX_PER_MM = 100.0

_PSF_CORE_SIGMA = 1.2
_PSF_WING_SIGMA = 6.0
_PSF_WING_WEIGHT = {"R": 0.15, "G": 0.20, "B": 0.30}
_PSF_KERNELS = {}
_PSF_FIT = None


def _gaussian_kernel(sigma, size=None):
    """生成归一化的二维高斯核；size 缺省时按 sigma 自动取。"""
    if size is None:
        size = max(int(2 * np.ceil(3.0 * sigma)) + 1, 3)
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel = kernel_1d @ kernel_1d.T
    return kernel / kernel.sum()


def _solve_nonneg(A, b):
    """活动集式非负最小二乘（numpy 实现，无 scipy 依赖）。"""
    cols = list(range(A.shape[1]))
    coef = np.zeros(A.shape[1])
    while cols:
        x, *_ = np.linalg.lstsq(A[:, cols], b, rcond=None)
        if np.all(x >= -1e-12):
            coef[cols] = np.maximum(x, 0.0)
            return coef
        drop = cols[int(np.argmin(x))]
        cols.remove(drop)
    return coef


def _fit_mtf_layer(freq, resp):
    """单层 MTF → 空间域参数拟合（纯 numpy）。

    模型：MTF(f) = c0·exp(-2π²σc²f²) + c1·exp(-2π²σw²f²) - c2·exp(-2π²σa²f²)
    对应空间域核：c0·g(σc) + c1·g(σw) - c2·g(σa)（σ 单位 mm）。
    网格搜索 σc/σw/σa，对每个组合用非负最小二乘解系数，取残差最小。
    """
    twopi2 = 2.0 * np.pi ** 2
    best = None
    # 核心 0.8~6µm、翼 4~80µm、邻界 6~80µm（空间 σ，mm）
    s0s = np.logspace(np.log10(0.0008), np.log10(0.006), 24)
    s1s = np.logspace(np.log10(0.004), np.log10(0.080), 22)
    s2s = np.logspace(np.log10(0.006), np.log10(0.080), 18)
    for s0 in s0s:
        g0 = np.exp(-twopi2 * s0 ** 2 * freq ** 2)
        for s1 in s1s:
            g1 = np.exp(-twopi2 * s1 ** 2 * freq ** 2)
            for s2 in s2s:
                g2 = np.exp(-twopi2 * s2 ** 2 * freq ** 2)
                coef = _solve_nonneg(np.column_stack([g0, g1, -g2]), resp)
                resid = float(np.sum((g0 * coef[0] + g1 * coef[1] - g2 * coef[2] - resp) ** 2))
                if best is None or resid < best[0]:
                    best = (resid, s0, s1, s2, coef)
    return best


def _get_psf_fit():
    """加载 MTF 数据并逐层拟合（返回 σ_mm 与系数），缓存。"""
    global _PSF_FIT
    if _PSF_FIT is None:
        mtf = _data.get_mtf()
        layers = {
            "R": (mtf["freq_R_raw"], mtf["resp_R_raw"]),
            "G": (mtf["freq_G_raw"], mtf["resp_G_raw"]),
            "B": (mtf["freq_B_raw"], mtf["resp_B_raw"]),
        }
        _PSF_FIT = {}
        for layer, (freq, resp) in layers.items():
            _, s_c, s_w, s_a, coef = _fit_mtf_layer(freq, resp / 100.0)
            _PSF_FIT[layer] = (s_c, s_w, s_a, coef[0], coef[1], coef[2])
    return _PSF_FIT


def _get_psf_kernel(layer, mode="new"):
    """按层获取 PSF 核（new=MTF 拟合；old=猜测双高斯），缓存。"""
    key = (layer, mode)
    if key in _PSF_KERNELS:
        return _PSF_KERNELS[key]
    if mode == "old":
        # 两个高斯必须在同一尺寸上相加：统一用宽翼的尺寸
        wing_size = max(int(2 * np.ceil(3.0 * _PSF_WING_SIGMA)) + 1, 3)
        core = _gaussian_kernel(_PSF_CORE_SIGMA, size=wing_size)
        wing = _gaussian_kernel(_PSF_WING_SIGMA, size=wing_size)
        w = _PSF_WING_WEIGHT[layer]
        kernel = (1.0 - w) * core + w * wing
        kernel = kernel / kernel.sum()
    elif mode == "new":
        # MTF 拟合：核心+翼-邻界；σ_mm → σ_px（PX_PER_MM）
        s_c, s_w, s_a, c0, c1, c2 = _get_psf_fit()[layer]
        sigmas_px = [s_c * PX_PER_MM, s_w * PX_PER_MM, s_a * PX_PER_MM]
        size = max(int(2 * np.ceil(3.0 * max(sigmas_px))) + 1, 3)
        kernel = (
            c0 * _gaussian_kernel(sigmas_px[0], size)
            + c1 * _gaussian_kernel(sigmas_px[1], size)
            - c2 * _gaussian_kernel(sigmas_px[2], size)
        )
        kernel = kernel / np.maximum(kernel.sum(), 1e-6)
    else:
        raise ValueError("psf_mode 只支持 new / old")
    _PSF_KERNELS[key] = kernel.astype(np.float32)
    return _PSF_KERNELS[key]


def _apply_psf(e_r, e_g, e_b, mode="new"):
    """对线性层曝光做 PSF 卷积（物理上光在乳剂中先扩散、后显影）。"""
    e_r = cv2.filter2D(e_r, cv2.CV_32F, _get_psf_kernel("R", mode), borderType=cv2.BORDER_REFLECT)
    e_g = cv2.filter2D(e_g, cv2.CV_32F, _get_psf_kernel("G", mode), borderType=cv2.BORDER_REFLECT)
    e_b = cv2.filter2D(e_b, cv2.CV_32F, _get_psf_kernel("B", mode), borderType=cv2.BORDER_REFLECT)
    return e_r, e_g, e_b


# ---------- 颗粒（密度域，48µm 孔径标定） ----------
# ① 白噪声 → Cholesky 层间相关（ρ=0.5）；
# ② sign·χ² 伽马聚簇成形——化合物-泊松散粒噪声的幅度模型，
#    产生非高斯重尾团块（χ² 分布即散粒噪声幅度模型）；
# ③ 平滑高斯核赋形（显影扩散 h）+ unsharp 合并单核，给高频“咬合”；
# ④ 可选 2x/3x 超采样：高分辨率生成 → 块平均降采样，
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
_GRAIN_CACHE = {}             # 超采样级 ss -> (k_boost, aperture, scale)


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


def _grain_parts(ss=2):
    """按超采样级 ss 构建“赋形+unsharp”合并核、孔径核与标定系数。"""
    ss = int(ss)
    if ss in _GRAIN_CACHE:
        return _GRAIN_CACHE[ss]
    s = float(ss)
    # 平滑赋形核 + unsharp 线性合并成单个核（DC 增益=1）
    k_sharp = _gaussian_kernel(_GRAIN_KERNEL_SIGMA * s)
    k_blur = _gaussian_kernel(_GRAIN_UNSHARP_SIGMA * s)
    k_blurred = cv2.filter2D(k_sharp, cv2.CV_32F, k_blur,
                             borderType=cv2.BORDER_REFLECT)
    k_boost = ((1.0 + _GRAIN_UNSHARP) * k_sharp
               - _GRAIN_UNSHARP * k_blurred)
    k_boost = k_boost.astype(np.float32)
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
    n = np.empty_like(g)
    for ch in range(3):
        n[..., ch] = cv2.filter2D(g[..., ch], cv2.CV_32F, k_boost,
                                  borderType=cv2.BORDER_REFLECT)
    n_d = n.reshape(c, ss, c, ss, 3).mean(axis=(1, 3))
    ap = np.empty((c, c, 3), dtype=np.float32)
    for ch in range(3):
        ap[..., ch] = cv2.filter2D(n_d[..., ch], cv2.CV_32F, aperture,
                                   borderType=cv2.BORDER_REFLECT)
    scale = float(1.0 / ap.std())
    _GRAIN_CACHE[ss] = (k_boost, aperture.astype(np.float32), scale)
    return _GRAIN_CACHE[ss]


def _grain_field(ss, k_boost, scale, h, w):
    """颗粒场：按超采样级生成（ss×白噪声→成形→块平均降采样）。

    ss=1 直接在本分辨率生成；ss>1 分条带处理控制内存，
    条带边界用反射余量保证卷积正确。
    """
    if ss == 1:
        z = np.random.randn(h, w, 3).astype(np.float32)
        corr = z @ _grain_cholesky().T
        g = _grain_shaped(corr)
        n = np.empty_like(g)
        for ch in range(3):
            n[..., ch] = cv2.filter2D(g[..., ch], cv2.CV_32F, k_boost,
                                      borderType=cv2.BORDER_REFLECT)
        return n * scale

    hs, ws = h * ss, w * ss
    out = np.empty((h, w, 3), dtype=np.float32)
    margin = k_boost.shape[0] // 2
    stripe_out = max(1, 1024 // (ss * ss))
    chol = _grain_cholesky()
    for i0 in range(0, h, stripe_out):
        i1 = min(i0 + stripe_out, h)
        r0 = max(i0 * ss - margin, 0)
        r1 = min(i1 * ss + margin, hs)
        z = np.random.randn(r1 - r0, ws, 3).astype(np.float32)
        corr = z @ chol.T
        g = _grain_shaped(corr)
        n = np.empty_like(g)
        for ch in range(3):
            n[..., ch] = cv2.filter2D(g[..., ch], cv2.CV_32F, k_boost,
                                      borderType=cv2.BORDER_REFLECT)
        crop = n[(i0 * ss - r0):(i1 * ss - r0)]
        rows = crop.shape[0]
        n_d = crop.reshape(rows // ss, ss, w, ss, 3).mean(axis=(1, 3))
        out[i0:i1] = n_d * scale
    return out


def _apply_grain(d_r, d_g, d_b, logh_r, logh_g, logh_b, ss=2):
    """颗粒：伽马聚簇 + 平滑赋形 + unsharp + 可选超采样 + 孔径闭环。"""
    sigma_r, sigma_g, sigma_b = _grain_sigmas(logh_r, logh_g, logh_b)
    k_boost, aperture, scale = _grain_parts(ss)
    n = _grain_field(ss, k_boost, scale, d_r.shape[0], d_r.shape[1])
    d_r = d_r + n[..., 0] * sigma_r
    d_g = d_g + n[..., 1] * sigma_g
    d_b = d_b + n[..., 2] * sigma_b
    return d_r, d_g, d_b


# ---------- 对数曝光与特性曲线 ----------
def _to_log_exposure(e, exposure_offset):
    """线性曝光 → logH（对数曝光量工作区），含曝光偏移（档位）。"""
    e = np.maximum(np.asarray(e, dtype=np.float32), E_FLOOR)
    return np.log10(e) + CAL + exposure_offset * LOG10_2


def _char_density(logh_r, logh_g, logh_b):
    """logH → Status M 密度（三层各自查表，np.interp 自动夹两端）。"""
    char = _get_char()
    logE = char["logE"]
    d_r = np.interp(logh_r, logE, char["density_R"]).astype(np.float32)
    d_g = np.interp(logh_g, logE, char["density_G"]).astype(np.float32)
    d_b = np.interp(logh_b, logE, char["density_B"]).astype(np.float32)
    return d_r, d_g, d_b


# ---------- 染料堆叠 + 客观 CIE-RGB 扫描（负片读数） ----------
_SCAN_WEIGHTS = None
_DYE_TILE = 512
_SCAN_DMIN = None
_SCAN_DMAX = None


def _scan_weights():
    """客观扫描仪的 CIE 权重：k·D65·CMF·Δλ，Y 归一化到 1。"""
    global _SCAN_WEIGHTS
    if _SCAN_WEIGHTS is None:
        k = 1.0 / (cie_data.D65_SPD * cie_data.CMF_1931_2[:, 1] * cie_data.DELTA_LAMBDA).sum()
        w = k * cie_data.D65_SPD[:, None] * cie_data.CMF_1931_2 * cie_data.DELTA_LAMBDA
        _SCAN_WEIGHTS = w.astype(np.float32)  # (61,3)：X/Y/Z
    return _SCAN_WEIGHTS


def _scan_dmin():
    """D-min 状态走一遍扫描仪，得到每通道理论“白点”（静态截止用）。"""
    global _SCAN_DMIN
    if _SCAN_DMIN is None:
        dye = _get_dye()
        t_min = 10.0 ** (-dye["dmin"][:61].astype(np.float32))
        xyz_min = t_min @ _scan_weights()
        _SCAN_DMIN = np.maximum(
            xyz_min @ cie_data.XYZ_TO_SRGB.astype(np.float32).T, 1e-6
        ).astype(np.float32)
    return _SCAN_DMIN


def _scan_dmax():
    """三层都处于 D-max 状态走一遍扫描仪，得到每通道理论“黑点”（静态截止用）。"""
    global _SCAN_DMAX
    if _SCAN_DMAX is None:
        char = _get_char()
        dye = _get_dye()
        s_y = dye["yellow_norm"][:61].astype(np.float32)
        s_m = dye["magenta_norm"][:61].astype(np.float32)
        s_c = dye["cyan_norm"][:61].astype(np.float32)
        d_lambda = (
            (char["dmax_B"] - char["dmin_B"]) * s_y
            + (char["dmax_G"] - char["dmin_G"]) * s_m
            + (char["dmax_R"] - char["dmin_R"]) * s_c
            + dye["dmin"][:61].astype(np.float32)
        )
        t_max = 10.0 ** (-d_lambda)
        xyz_max = t_max @ _scan_weights()
        _SCAN_DMAX = np.maximum(
            xyz_max @ cie_data.XYZ_TO_SRGB.astype(np.float32).T, 1e-6
        ).astype(np.float32)
    return _SCAN_DMAX


def _scan_of_net(net_b, net_g, net_r):
    """用净染料量（已减 D-min）直接算单点客观扫描值 (3,)。"""
    dye = _get_dye()
    s_y = dye["yellow_norm"][:61].astype(np.float32)
    s_m = dye["magenta_norm"][:61].astype(np.float32)
    s_c = dye["cyan_norm"][:61].astype(np.float32)
    dmin = dye["dmin"][:61].astype(np.float32)
    weights = _scan_weights()
    srgb_inv = cie_data.XYZ_TO_SRGB.astype(np.float32).T
    d_lambda = net_b * s_y + net_g * s_m + net_r * s_c + dmin
    t = 10.0 ** (-d_lambda)
    xyz = np.stack([
        (t * weights[:, 0]).sum(),
        (t * weights[:, 1]).sum(),
        (t * weights[:, 2]).sum(),
    ])
    return np.clip(xyz @ srgb_inv, 0.0, None).astype(np.float32)


def _scan_gray(gray):
    """输入灰值（线性，白场=1）走完前段管线后的扫描值 (3,)。"""
    char = _get_char()
    logh = np.log10(max(float(gray), E_FLOOR)) + CAL
    logE = char["logE"]
    d_b = np.interp(logh, logE, char["density_B"]).astype(np.float32)
    d_g = np.interp(logh, logE, char["density_G"]).astype(np.float32)
    d_r = np.interp(logh, logE, char["density_R"]).astype(np.float32)
    return _scan_of_net(
        np.maximum(d_b - char["dmin_B"], 0.0),
        np.maximum(d_g - char["dmin_G"], 0.0),
        np.maximum(d_r - char["dmin_R"], 0.0),
    )


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
            t = 10.0 ** (-d_lambda)
            xyz = np.stack([
                (t * weights[:, 0]).sum(axis=-1),
                (t * weights[:, 1]).sum(axis=-1),
                (t * weights[:, 2]).sum(axis=-1),
            ], axis=-1)
            lin_srgb = xyz @ srgb_to_xyz_inv
            out[y0:y1, x0:x1] = np.clip(lin_srgb, 0.0, None)
    return out


# ---------- 反转：对数密度域三点锚定 + SOP（negfix8 × negadoctor 合流） ----------
def _invert(scan, gamma=OUT_GAMMA, clip_mode="dynamic",
              wb_strength=0.0, wb_gray=0.18, s_strength=0.10,
              slope=1.0):
    """对数密度域三点锚定 + slope/offset/power 反转。

    与方案原版的区别：反转在密度刻度（D=-log10(scan)）上进行，
    而不是在透光率刻度上进行：
    1. 密度化：D = -log10(scan)。色罩在各通道的差异在密度里是加法常数，
       在透光率里却被指数放大成倍率，因此密度域中灰落点更齐
       （实测 18% 灰的通道落点差从 0.118 缩到 0.039）；
    2. 两端锚点（dynamic/static 与旧方案同语义，搬到密度域）；
    3. 归一化反转 pos = (D - D_white)/(D_black - D_white)，不预裁切；
    4. 小剂量中灰白平衡：密度域的加法偏移（静态灰点标定，不随图变），
       剂量远小于线性域乘法增益，不推偏其它灰阶；强度 -1~+1，
       负值表示反向增强原有偏色（默认 0=不起用）；
    5. 全局斜率（可选）：围绕中灰目标点缩放，只改对比、不改色；
    6. 趾肩 S 曲线（可选）：pos + k·pos(1-pos)(pos-0.5)，端点不动、
       中亮部压一点、中暗部提一点，复刻手工较色时的经典 S 形；
    7. 伽马 → uint8。
    """
    eps = 1e-6
    static_white = _scan_dmin()
    static_black = _scan_dmax()

    d = -np.log10(np.maximum(np.asarray(scan, dtype=np.float32), eps))

    if clip_mode == "dynamic":
        black_s = np.percentile(scan, 1.0, axis=(0, 1)).astype(np.float32)
        white_s = np.percentile(scan, 99.0, axis=(0, 1)).astype(np.float32)
        # 均匀场保护：与旧方案同一判据（透光率空间锚点差 < 0.02）
        if (white_s - black_s).min() < 0.02:
            black_s = static_black
            white_s = static_white
        d_black = -np.log10(np.maximum(black_s, eps))
        d_white = -np.log10(np.maximum(white_s, eps))
    elif clip_mode == "static":
        d_black = -np.log10(np.maximum(static_black, eps))
        d_white = -np.log10(np.maximum(static_white, eps))
    else:
        raise ValueError("clip_mode 只支持 dynamic / static")

    # 归一化反转：场景高光（负片低透光、高密度 D≈d_black）→ pos=1，
    # 阴影（负片高透光、低密度 D≈d_white）→ pos=0
    pos = (d - d_white) / np.maximum(d_black - d_white, 1e-6)

    # 小剂量中灰白平衡（密度域加法偏移，静态标定）
    scan_gray = _scan_gray(wb_gray)
    d_gray = -np.log10(np.maximum(scan_gray, eps))
    d_black_s = -np.log10(np.maximum(static_black, eps))
    d_white_s = -np.log10(np.maximum(static_white, eps))
    pos_gray = (d_gray - d_white_s) / np.maximum(d_black_s - d_white_s, 1e-6)
    target = float(pos_gray.min())
    if wb_strength != 0.0:
        pos = pos + wb_strength * (target - pos_gray)

    # 全局斜率（可选，围绕中灰目标点，只改对比不改色）
    if slope != 1.0:
        pos = (pos - target) * slope + target

    # 趾肩 S 曲线（可选）
    if s_strength > 0.0:
        pos = pos + s_strength * pos * (1.0 - pos) * (pos - 0.5)

    pos = np.clip(pos, 0.0, 1.0)
    out = pos ** gamma
    return (out * 255.0 + 0.5).astype(np.uint8)
# ---------- 主入口 ----------
def process(lux_r, lux_g, lux_b, use_spectral_input=True, psf_mode="new",
            grain_ss=2,
            exposure_offset=0.0, gamma=OUT_GAMMA, clip_mode="dynamic",
            wb_strength=0.0, wb_gray=0.18, s_strength=0.10, slope=1.0):
    """5213 光谱实验模型主入口。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（与 Helpers 输出一致）。
        use_spectral_input：光谱输入开关。True=CIE-RGB 色匹配函数伪发射迁移
                            （默认）；False=直接使用 RGB 作为层曝光。
        psf_mode：光扩散模式。"new"=datasheet MTF 频域拟合核（默认）；
                  "old"=早期猜测的双高斯核。
        grain_ss：颗粒超采样级（1=关闭，2=2x，3=3x；默认 2）。
        exposure_offset：曝光偏移（档位，正=过曝）。
        gamma：反转后叠的显示伽马（默认 1.60）。
        clip_mode：两端截止模式，"dynamic"=1/99 分位动态截止（默认），
                   "static"=理论白/黑点静态截止（均匀场安全）。
        wb_strength：中灰白平衡强度（-1~+1，默认 0=不起用；
                     正=校平中灰，负=反向增强原有偏色）。
        wb_gray：白平衡参考灰点（线性输入灰值，默认 0.18）。
        s_strength：趾肩 S 曲线强度（0=关闭，默认 0.10）。
        slope：全局斜率（围绕中灰目标点，1.0=关闭）。
    返回：
        (H,W,3) uint8，RGB 顺序。
    """
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")
    if psf_mode not in ("new", "old"):
        raise ValueError("psf_mode 只支持 new / old")
    if grain_ss not in (1, 2, 3):
        raise ValueError("grain_ss 只支持 1 / 2 / 3")

    lux_r = np.asarray(lux_r, dtype=np.float32)
    lux_g = np.asarray(lux_g, dtype=np.float32)
    lux_b = np.asarray(lux_b, dtype=np.float32)

    # 1. 光谱输入（开关，默认开启）
    if use_spectral_input:
        # CIE-RGB 标准色匹配函数伪发射源 → 实测灵敏度吸收积分
        from Spectral import migrate_rgb_primary
        e_r, e_g, e_b = migrate_rgb_primary(lux_r, lux_g, lux_b, film="5213")
    else:
        e_r, e_g, e_b = lux_r.copy(), lux_g.copy(), lux_b.copy()

    # 2. PSF 光扩散（线性物理模式；new=MTF拟合，old=猜测双高斯）
    e_r, e_g, e_b = _apply_psf(e_r, e_g, e_b, psf_mode)

    # 3. 对数曝光工作区
    logh_r = _to_log_exposure(e_r, exposure_offset)
    logh_g = _to_log_exposure(e_g, exposure_offset)
    logh_b = _to_log_exposure(e_b, exposure_offset)

    # 4. 特性曲线 → Status M 密度
    d_r, d_g, d_b = _char_density(logh_r, logh_g, logh_b)

    # 5. 密度域颗粒（伽马聚簇+平滑赋形+unsharp+超采样+孔径闭环）
    d_r, d_g, d_b = _apply_grain(d_r, d_g, d_b, logh_r, logh_g, logh_b,
                                   ss=grain_ss)

    # 6. 输出端：染料光谱堆叠 + CIE-RGB 客观扫描（固定开启）→ 反转
    scan = _dye_stack_scan(d_r, d_g, d_b)
    return _invert(
        scan, gamma=gamma, clip_mode=clip_mode,
        wb_strength=wb_strength, wb_gray=wb_gray,
        s_strength=s_strength, slope=slope,
    )
