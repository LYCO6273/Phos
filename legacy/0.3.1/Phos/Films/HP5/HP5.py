#赛博请神
# HP5.py —— ILFORD HP5 Plus（ISO 400 黑白负片）光谱模型
# 黑白负片 = 彩负的单层简化（见 STANDARD_FLOW.md 第二部分）：
#   线性 RGB
#     → 光谱输入（固定）CIE-RGB 色匹配函数伪发射 → 实测灵敏度吸收积分（单通道）
#     → [PSF] 物理核（核心高斯 + K0 扩散晕 − 邻界高斯，MTF 拟合）
#     → 对数曝光（统一对数工作区 + 曝光偏移）
#     → 特性曲线 → 密度
#     → 密度域颗粒（σ 按密度查表，rms 曲线 X 轴=密度）
#     → 反转：单通道固定锚点（0,0,0→黑端，1,1,1→白端）→ 伽马 → uint8
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
# 默认曝光参考：18% 灰落在这条 logE 上
GRAY_E = 0.18
# logH = log10(E) + CAL，使得 log10(0.18) + CAL = REF_LOG_E
CAL = REF_LOG_E - np.log10(GRAY_E)
# 曝光量下限，防止 log(0)
E_FLOOR = 1e-4
# 曝光偏移（档位）→ log10 曝光单位
LOG10_2 = 0.3010299956639812
# 输出伽马（反转后叠的显示伽马，按胶片标定：HP5 默认 1.4）
OUT_GAMMA = 1.4


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


# ---------- 光扩散（PSF，线性物理模式，单层） ----------
# 物理核 = 核心高斯 + K0 扩散晕 − 邻界高斯，由 MTF 频域拟合：
#   K0(r/δ) 是三维扩散格林函数 e^(-r/δ)/r 沿膜厚的精确投影（重尾 1/r·e^(-r/δ)），
#   其频域闭式为洛伦兹 1/(1+(2πδf)²)；负邻界高斯负责 MTF 中频 >100% 的响应。
# 像素-毫米声明：1 图像像素 = 10µm 胶片平面（35mm 全幅 ≈ 3600px 宽）。
PX_PER_MM = 100.0

# 物理核参数搜索范围（µm）
_PSF_CORE_RANGE_UM = (0.8, 6.0)
_PSF_DELTA_RANGE_UM = (4.0, 60.0)
_PSF_ADJ_RANGE_UM = (6.0, 40.0)

_PSF_KERNEL = None
_PSF_STACK = None
_PSF_FIT = None

# PSF 拟合磁盘缓存：版本号 + MTF 原始点指纹，数据变更时自动失效重拟合
_PSF_FIT_VERSION = 1
_PSF_FIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "psf_params_hp5.npz")


def _gaussian_kernel(sigma, size=None):
    """生成归一化的二维高斯核；size 缺省时按 sigma 自动取。"""
    if size is None:
        size = max(int(2 * np.ceil(3.0 * sigma)) + 1, 3)
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel = kernel_1d @ kernel_1d.T
    return kernel / kernel.sum()


def _k0(x):
    """修正贝塞尔 K0（Abramowitz-Stegun 9.8，numpy 实现，无 scipy 依赖）。"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = x <= 2.0
    xs = x[small]
    t = xs / 2.0
    i0 = (1.0 + 0.25 * xs ** 2 + 0.015625 * xs ** 4
          + 0.0004340 * xs ** 6 + 0.00000678 * xs ** 8)
    out[small] = (-np.log(t) * i0 - 0.57721566
                  + 0.42278420 * t ** 2 + 0.23069756 * t ** 4
                  + 0.03488590 * t ** 6 + 0.00262698 * t ** 8
                  + 0.00010750 * t ** 10 + 0.00000740 * t ** 12)
    xl = x[~small]
    z = 2.0 / xl
    out[~small] = (np.exp(-xl) / np.sqrt(xl)) * (
        1.25331414 - 0.07832358 * z + 0.02189568 * z ** 2
        - 0.01062446 * z ** 3 + 0.00587872 * z ** 4
        - 0.00251540 * z ** 5 + 0.00053208 * z ** 6)
    return out


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


def _fit_mtf(freq, resp):
    """单层 MTF → 空间域参数拟合（纯 numpy）。

    模型：MTF(f) = w_c·exp(-2π²σc²f²) + w_d/(1+(2πδf)²) - w_a·exp(-2π²σa²f²)
    对应空间域核：w_c·g(σc) + (w_d/2πδ²)·K0(r/δ) - w_a·g(σa)（σ/δ 单位 mm）。
    网格搜索 σc/δ/σa，非负最小二乘解权重。
    """
    twopi2 = 2.0 * np.pi ** 2
    best = None
    s0s = np.logspace(np.log10(_PSF_CORE_RANGE_UM[0] * 1e-3),
                      np.log10(_PSF_CORE_RANGE_UM[1] * 1e-3), 20)
    dlt = np.logspace(np.log10(_PSF_DELTA_RANGE_UM[0] * 1e-3),
                      np.log10(_PSF_DELTA_RANGE_UM[1] * 1e-3), 24)
    s2s = np.logspace(np.log10(_PSF_ADJ_RANGE_UM[0] * 1e-3),
                      np.log10(_PSF_ADJ_RANGE_UM[1] * 1e-3), 16)
    for s0 in s0s:
        g0 = np.exp(-twopi2 * s0 ** 2 * freq ** 2)
        for dl in dlt:
            g1 = 1.0 / (1.0 + (2.0 * np.pi * dl * freq) ** 2)
            for s2 in s2s:
                g2 = np.exp(-twopi2 * s2 ** 2 * freq ** 2)
                coef = _solve_nonneg(np.column_stack([g0, g1, -g2]), resp)
                resid = float(np.sum((g0 * coef[0] + g1 * coef[1]
                                      - g2 * coef[2] - resp) ** 2))
                if best is None or resid < best[0]:
                    best = (resid, s0, dl, s2, coef)
    return best


def _load_psf_fit():
    """读取 PSF 拟合磁盘缓存；版本或 MTF 指纹不符返回 None（触发重拟合）。"""
    if not os.path.exists(_PSF_FIT_FILE):
        return None
    d = np.load(_PSF_FIT_FILE, allow_pickle=True)
    if "psf_cache_version" not in d.files:
        return None
    if int(d["psf_cache_version"]) != _PSF_FIT_VERSION:
        return None
    mtf = _data.get_mtf()
    if not (np.array_equal(d["mtf_freq_raw"], mtf["freq_raw"])
            and np.array_equal(d["mtf_resp_raw"], mtf["resp_raw"])):
        return None
    # 缓存字段按 µm 存储 σ/δ（与归档 psf_params_*.npz 语义一致），读取时还原 mm
    return (float(d["sigma_core_um"]) * 1e-3,
            float(d["delta_um"]) * 1e-3,
            float(d["sigma_adj_um"]) * 1e-3,
            float(d["w_core"]), float(d["w_halo"]), float(d["w_adj"]))


def _save_psf_fit(fit):
    """把拟合结果写回磁盘缓存（含 MTF 指纹与版本号；σ/δ 按 µm 存）。"""
    mtf = _data.get_mtf()
    s_c, dl, s_a, w_c, w_d, w_a = fit
    np.savez(_PSF_FIT_FILE,
             psf_cache_version=np.int32(_PSF_FIT_VERSION),
             sigma_core_um=np.float64(s_c * 1e3),
             delta_um=np.float64(dl * 1e3),
             sigma_adj_um=np.float64(s_a * 1e3),
             w_core=np.float64(w_c), w_halo=np.float64(w_d),
             w_adj=np.float64(w_a),
             mtf_freq_raw=mtf["freq_raw"], mtf_resp_raw=mtf["resp_raw"])


def _get_psf_fit():
    """物理核拟合：返回 (σc, δ, σa, w_c, w_d, w_a)（mm），磁盘缓存。"""
    global _PSF_FIT
    if _PSF_FIT is None:
        _PSF_FIT = _load_psf_fit()
        if _PSF_FIT is None:
            mtf = _data.get_mtf()
            _, s_c, dl, s_a, coef = _fit_mtf(mtf["freq_raw"], mtf["resp_raw"] / 100.0)
            _PSF_FIT = (s_c, dl, s_a, coef[0], coef[1], coef[2])
            _save_psf_fit(_PSF_FIT)
    return _PSF_FIT


def _get_psf_kernel():
    """物理 PSF 核（核心高斯 + K0 扩散晕 − 邻界高斯），缓存。"""
    global _PSF_KERNEL
    if _PSF_KERNEL is not None:
        return _PSF_KERNEL
    s_c, dl, s_a, w_c, w_d, w_a = _get_psf_fit()
    s_c_px = s_c * PX_PER_MM
    dl_px = dl * PX_PER_MM
    s_a_px = s_a * PX_PER_MM
    radius = max(s_c_px, dl_px, s_a_px)
    size = max(int(2 * np.ceil(3.0 * radius)) + 1, 3)
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(xx, yy)
    core = _gaussian_kernel(s_c_px, size)
    # K0 在 r=0 有对数奇点：中心用 ε=0.5px 截断，中心由核心高斯主导
    halo = _k0(np.maximum(rr, 0.5) / dl_px)
    halo = halo / halo.sum()
    adj = _gaussian_kernel(s_a_px, size)
    kernel = w_c * core + w_d * halo - w_a * adj
    kernel = kernel / np.maximum(kernel.sum(), 1e-6)
    _PSF_KERNEL = kernel.astype(np.float32)
    return _PSF_KERNEL


def _gauss_decomp(kernel, n_pos=8, sigma_adj=None):
    """把空间核分解成可分离高斯堆（正高斯和 + 可选负邻界高斯）。

    基底为网格归一化高斯（与 cv2.GaussianBlur 的归一化一致），
    保证堆叠后的 DC 增益恒为 1；与精确核卷积对比误差 <0.1%。
    物理依据：K0(r/δ) = ∫ e^{-t}·G(√(2t)·δ) dt/(2t) —— 扩散晕是高斯尺度混合。
    """
    k = kernel.astype(np.float64)
    H, W = k.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(xx - cx, yy - cy).ravel()
    rmax = float(r.max())
    sigs = np.logspace(np.log10(0.3), np.log10(max(rmax, 1.5)), n_pos)
    cols, names = [], []
    for s in sigs:
        g = np.exp(-r ** 2 / (2.0 * s ** 2))
        cols.append(g / g.sum())
        names.append(("+", float(s)))
    if sigma_adj is not None:
        g = np.exp(-r ** 2 / (2.0 * sigma_adj ** 2))
        cols.append(-g / g.sum())
        names.append(("-", float(sigma_adj)))
    w = _solve_nonneg(np.column_stack(cols), k.ravel())
    out = []
    for wi, (sign, s) in zip(w, names):
        if abs(wi) > 1e-6:
            out.append((sign, float(wi), s))
    return out


def _get_psf_stack():
    """快速卷积用的 (sign, weight, sigma_px) 高斯堆，缓存。"""
    global _PSF_STACK
    if _PSF_STACK is None:
        sigma_adj = _get_psf_fit()[2] * PX_PER_MM
        _PSF_STACK = _gauss_decomp(_get_psf_kernel(), sigma_adj=sigma_adj)
    return _PSF_STACK


def _apply_psf(e):
    """对线性亮度做 PSF 卷积（多尺度高斯堆，物理扩散）。"""
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
    return np.log10(e) + CAL + exposure_offset * LOG10_2


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


# ---------- 反转：单通道固定锚点 + 伽马 ----------
# 黑白点标定（特性曲线改用 0.2.3 手描版后重新标定，2026-08-28）：
# 黑端 = 0,0,0 输入（L=0 → logH≈-4.76 → D_min）的密度；
# 白端 = 1,1,1 输入（L=1 → logH=CAL=-0.76042 → D=1.270218）的密度；
# 灰点（0.18）D≈0.7898。
# 反转在密度域做两端截止：pos = (D - D黑端)/(D白端 - D黑端)。
_D_SHADOW = 0.175                  # 黑端锚点（输出黑）
_D_HIGHLIGHT = 1.2702176075104528  # 白端锚点（输出白）


def _invert(density, gamma=OUT_GAMMA):
    """密度域反转：两端截止 + 归一化 + 伽马 → uint8。"""
    pos = (density - _D_SHADOW) / (_D_HIGHLIGHT - _D_SHADOW)
    pos = np.clip(pos, 0.0, 1.0)
    out = pos ** gamma
    return (out * 255.0 + 0.5).astype(np.uint8)


# ---------- 主入口 ----------
def process(lux_r, lux_g, lux_b, grain_ss=2,
            exposure_offset=0.0, gamma=OUT_GAMMA):
    """HP5 黑白负片模型主入口（结构同彩负的单层简化）。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（与 Helpers 输出一致）。
        grain_ss：颗粒超采样级（1=关闭，2=2x；默认 2）。
        exposure_offset：曝光偏移（档位，正=过曝）。
        gamma：反转后叠的显示伽马（默认 1.4）。
    返回：
        (H,W) uint8 单通道成片。
    """
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")
    if grain_ss not in (1, 2):
        raise ValueError("grain_ss 只支持 1 / 2")

    lux_r = np.asarray(lux_r, dtype=np.float32)
    lux_g = np.asarray(lux_g, dtype=np.float32)
    lux_b = np.asarray(lux_b, dtype=np.float32)

    # 1. 光谱输入（固定开启）：CIE-RGB 色匹配函数伪发射源 → 实测灵敏度吸收积分
    from Spectral import migrate_bw_primary
    e0 = migrate_bw_primary(lux_r, lux_g, lux_b, film="hp5")

    # 2. PSF 光扩散（单层物理核，Tri-X 400 MTF 拟合）
    e = _apply_psf(e0)

    # 3. 对数曝光工作区
    logh = _to_log_exposure(e, exposure_offset)

    # 4. 特性曲线 → 密度
    d = _char_density(logh)

    # 5. 密度域颗粒（σ 按密度查表 + 可选 2x 超采样）
    d = _apply_grain(d, ss=grain_ss)

    # 6. 反转：单通道固定锚点 + 伽马
    return _invert(d, gamma=gamma)
