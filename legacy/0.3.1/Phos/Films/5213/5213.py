#赛博请神
# 5213.py —— Vision3 200T（5213/7213）光谱实验模型
# 这是 0.3.0 唯一承载新特性的胶片模型；HP5 / Gold200 保持不变。
#
# 处理链：
#   线性 RGB
#     → 光谱输入（固定）CIE-RGB 色匹配函数伪发射迁移 → 层曝光 E_R/E_G/E_B
#     → [PSF] 物理核（核心高斯 + K0 扩散晕 − 邻界高斯，MTF 拟合）
#     → 颗粒：密度域伽马聚簇+可分离高斯赋形+可选 2x 超采样
#     → log10 层曝光（统一对数工作区 + 曝光偏移）
#     → 特性曲线 → Status M 密度
#     → 染料光谱堆叠（固定开启）→ CIE-RGB 客观扫描（负片读数）
#     → 反转：对数密度域固定锚点（0,0,0→黑端，1,1,1→白端）→ 伽马 → sRGB 输出
#
# 层映射约定：
#   lux_r = 红感光层（青染料）  lux_g = 绿感光层（品红染料）  lux_b = 蓝感光层（黄染料）

import numpy as np
import cv2
import importlib
import os

from Spectral import cie_data

# 数据模块名以数字开头，普通 import 语法不可用，这里用 importlib 加载
_data = importlib.import_module("Films.5213.5213_data")

# 模型目录 / 数据目录（磁盘缓存与数据文件所在）
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_MODEL_DIR, "data")


# ---------- 曝光标定常量 ----------
# 曝光轴已按静照测光常数 K=12.5、EI=200 标准化到绝对物理轴：
#   18% 灰的曝光量 H_gray = K / EI = 12.5/200 = 0.0625 lux·s
#   logH_gray = log10(0.0625) = -1.20412
# 旧相对轴灰点原在 -1.284，数据轴整体平移 +0.07988 后落在 -1.20412。
# 注意：灰点从 -1.284 移到 -1.20412，等效同一输入下输出约亮 0.27 档
# （0.07988 / log10(2) ≈ 0.265 EV），这是统一到绝对标尺的预期变化。
REF_LOG_E = -1.20412
# 默认曝光参考：18% 灰落在这条 logE 上
GRAY_E = 0.18
# logH = log10(E) + CAL，使得 log10(0.18) + CAL = REF_LOG_E
CAL = REF_LOG_E - np.log10(GRAY_E)
# 曝光量下限，防止 log(0)
E_FLOOR = 1e-4
# 曝光偏移（档位）→ log10 曝光单位
LOG10_2 = 0.3010299956639812
# 输出伽马（反转后叠的显示伽马，按胶片标定：5213 默认 1.7）
OUT_GAMMA = 1.70


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
_PSF_CORE_RANGE_UM = (0.8, 6.0)
_PSF_DELTA_RANGE_UM = (4.0, 60.0)
_PSF_ADJ_RANGE_UM = (6.0, 40.0)

_PSF_KERNELS = {}
_PSF_FIT = None


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
    # 小参数：K0(x) = -ln(x/2)·I0(x) + 多项式
    xs = x[small]
    t = xs / 2.0
    i0 = (1.0 + 0.25 * xs ** 2 + 0.015625 * xs ** 4
          + 0.0004340 * xs ** 6 + 0.00000678 * xs ** 8)
    out[small] = (-np.log(t) * i0 - 0.57721566
                  + 0.42278420 * t ** 2 + 0.23069756 * t ** 4
                  + 0.03488590 * t ** 6 + 0.00262698 * t ** 8
                  + 0.00010750 * t ** 10 + 0.00000740 * t ** 12)
    # 大参数：K0(x) ≈ e^(-x)/√x · 多项式（误差 < 1.9e-7）
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


def _fit_mtf_layer(freq, resp):
    """物理核：单层 MTF → 空间域参数拟合（纯 numpy）。

    模型：MTF(f) = w_c·exp(-2π²σc²f²) + w_d/(1+(2πδf)²) - w_a·exp(-2π²σa²f²)
    对应空间域核：w_c·g(σc) + (w_d/2πδ²)·K0(r/δ) - w_a·g(σa)（σ/δ 单位 mm）。
    洛伦兹项是 K0 扩散晕的频域闭式；网格搜索 σc/δ/σa，非负最小二乘解权重。
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
                resid = float(np.sum((g0 * coef[0] + g1 * coef[1] - g2 * coef[2] - resp) ** 2))
                if best is None or resid < best[0]:
                    best = (resid, s0, dl, s2, coef)
    return best


# PSF 拟合磁盘缓存：版本号 + MTF 原始点指纹，数据变更时自动失效重拟合
_PSF_FIT_VERSION = 3
_PSF_FIT_FILE = os.path.join(_DATA_DIR, "psf_params_5213.npz")


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
    for tag, key in [("R", "freq_R_raw"), ("R", "resp_R_raw"),
                     ("G", "freq_G_raw"), ("G", "resp_G_raw"),
                     ("B", "freq_B_raw"), ("B", "resp_B_raw")]:
        if not np.array_equal(d[f"mtf_{key}"], mtf[key]):
            return None
    out = {}
    for i, layer in enumerate(d["layer"]):
        out[str(layer)] = (
            float(d["sigma_core_um"][i]) * 1e-3,
            float(d["delta_um"][i]) * 1e-3,
            float(d["sigma_adj_um"][i]) * 1e-3,
            float(d["w_core"][i]),
            float(d["w_halo"][i]),
            float(d["w_adj"][i]),
        )
    return out


def _save_psf_fit(fit):
    """把拟合结果写回磁盘缓存（含 MTF 指纹与版本号）。"""
    mtf = _data.get_mtf()
    # 注意：拟合返回的 σ/δ 单位是 mm，缓存字段按 µm 存储（与归档
    # psf_params_*.npz 语义一致），读取时再乘 1e-3 还原为 mm；
    # w_core/w_halo/w_adj 是无量纲权重，直接存原值。
    arr = {
        "sigma_core_um": np.array([fit[tag][0] * 1e3 for tag in ("R", "G", "B")]),
        "delta_um":      np.array([fit[tag][1] * 1e3 for tag in ("R", "G", "B")]),
        "sigma_adj_um":  np.array([fit[tag][2] * 1e3 for tag in ("R", "G", "B")]),
        "w_core":        np.array([fit[tag][3] for tag in ("R", "G", "B")]),
        "w_halo":        np.array([fit[tag][4] for tag in ("R", "G", "B")]),
        "w_adj":         np.array([fit[tag][5] for tag in ("R", "G", "B")]),
    }
    save = {
        "layer": np.array(["R", "G", "B"]),
        "psf_cache_version": np.int32(_PSF_FIT_VERSION),
        **arr,
        **{f"mtf_{k}": mtf[k] for k in
           ("freq_R_raw", "resp_R_raw", "freq_G_raw", "resp_G_raw",
            "freq_B_raw", "resp_B_raw")},
    }
    np.savez(_PSF_FIT_FILE, **save)


def _get_psf_fit():
    """物理核拟合：返回每层 (σc, δ, σa, w_c, w_d, w_a)（mm）。

    优先读磁盘缓存（跨进程复用，省去每次冷启动的网格搜索拟合）；
    MTF 数据或版本变化时自动重拟合并写回。
    """
    global _PSF_FIT
    if _PSF_FIT is None:
        _PSF_FIT = _load_psf_fit()
        if _PSF_FIT is None:
            mtf = _data.get_mtf()
            layers = {
                "R": (mtf["freq_R_raw"], mtf["resp_R_raw"]),
                "G": (mtf["freq_G_raw"], mtf["resp_G_raw"]),
                "B": (mtf["freq_B_raw"], mtf["resp_B_raw"]),
            }
            _PSF_FIT = {}
            for layer, (freq, resp) in layers.items():
                _, s_c, dl, s_a, coef = _fit_mtf_layer(freq, resp / 100.0)
                _PSF_FIT[layer] = (s_c, dl, s_a, coef[0], coef[1], coef[2])
            _save_psf_fit(_PSF_FIT)
    return _PSF_FIT


def _get_psf_kernel(layer):
    """按层获取物理 PSF 核（核心高斯 + K0 扩散晕 − 邻界高斯），缓存。"""
    key = layer
    if key in _PSF_KERNELS:
        return _PSF_KERNELS[key]
    # 物理核：核心高斯 + K0 扩散晕 − 邻界高斯；σ/δ_mm → σ/δ_px
    s_c, dl, s_a, w_c, w_d, w_a = _get_psf_fit()[layer]
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
    _PSF_KERNELS[key] = kernel.astype(np.float32)
    return _PSF_KERNELS[key]


_PSF_STACK = {}


def _gauss_decomp(kernel, n_pos=8, sigma_adj=None):
    """把空间核分解成可分离高斯堆（正高斯和 + 可选负邻界高斯）。

    基底为网格归一化高斯（与 cv2.GaussianBlur 的归一化一致），
    保证堆叠后的 DC 增益恒为 1；与精确核卷积对比误差 <0.1%。
    物理依据：K0(r/δ) = ∫ e^{-t}·G(√(2t)·δ) dt/(2t) —— 扩散晕是高斯尺度混合，
    即扩散方程的尺度空间解，用几个可分离高斯即可精确表达。
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


def _get_psf_stack(layer):
    """按层返回快速卷积用的 (sign, weight, sigma_px) 高斯堆，缓存。"""
    key = layer
    if key in _PSF_STACK:
        return _PSF_STACK[key]
    sigma_adj = _get_psf_fit()[layer][2] * PX_PER_MM
    stack = _gauss_decomp(_get_psf_kernel(layer), sigma_adj=sigma_adj)
    _PSF_STACK[key] = stack
    return stack


def _apply_psf(e_r, e_g, e_b):
    """对线性层曝光做 PSF 卷积（多尺度高斯堆，物理扩散）。

    扩散晕 K0(r/δ) 是高斯尺度混合（扩散方程的解），因此整条 PSF 可以
    用几个可分离高斯模糊求和实现；相比全图 FFT 快约两个数量级，
    与精确核卷积的误差 <0.1%。
    """
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
_GRAIN_CACHE_DIR = os.path.join(_DATA_DIR, "grain_calib_5213_ss{}.npz")


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
    return np.log10(e) + CAL + exposure_offset * LOG10_2


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

# 黑白点标定（曝光轴标准化后重新标定，2026-08-28）：
# 标定条件——K=12.5、EI=200 → logH_gray=-1.20412，曝光偏移 0；
# 黑端 = 0,0,0（片基/未曝光）输入走完
#        「CIE-RGB 光谱迁移 → 特性曲线 → 染料堆叠 → 客观扫描」后的扫描值；
# 白端 = 1,1,1（场景白）输入同管线的扫描值。
_SCAN_DMIN = np.array([0.641162276, 0.262283742, 0.137173802], np.float32)
_SCAN_WHITE = np.array([0.109555550, 0.019348590, 0.007591876], np.float32)


def _scan_weights():
    """客观扫描仪的 CIE 权重：k·D65·CMF·Δλ，Y 归一化到 1。"""
    global _SCAN_WEIGHTS
    if _SCAN_WEIGHTS is None:
        k = 1.0 / (cie_data.D65_SPD * cie_data.CMF_1931_2[:, 1] * cie_data.DELTA_LAMBDA).sum()
        w = k * cie_data.D65_SPD[:, None] * cie_data.CMF_1931_2 * cie_data.DELTA_LAMBDA
        _SCAN_WEIGHTS = w.astype(np.float32)  # (61,3)：X/Y/Z
    return _SCAN_WEIGHTS


def _scan_dmin():
    """黑端锚点：0,0,0（片基）扫描值（固定标定，见 _SCAN_DMIN）。"""
    return _SCAN_DMIN


def _scan_white():
    """白端锚点：1,1,1（场景白）扫描值（固定标定，见 _SCAN_WHITE）。"""
    return _SCAN_WHITE


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


# ---------- 反转：对数密度域固定锚点 + 伽马 ----------
def _invert(scan, gamma=OUT_GAMMA):
    """对数密度域反转（固定胶片锚点：0,0,0→黑端，1,1,1→白端）。

    反转在密度刻度（D=-log10(scan)）上进行，而不是在透光率刻度上：
    1. 密度化：D = -log10(scan)。色罩在各通道的差异在密度里是加法常数，
       在透光率里却被指数放大成倍率，因此密度域中灰落点更齐；
    2. 固定胶片锚点：黑端 = 0,0,0（片基/未曝光）的扫描密度，
       白端 = 1,1,1（场景白）的扫描密度——
       即“反转后上端截止”取 1,1,1 输入的输出，场景白正好落在输出白；
    3. 归一化反转 pos = (D - D_white)/(D_black - D_white)，不预裁切；
    4. 伽马 → uint8。
    """
    eps = 1e-6
    d = -np.log10(np.maximum(np.asarray(scan, dtype=np.float32), eps))
    d_white = -np.log10(np.maximum(_scan_dmin(), eps))    # 0,0,0 → 黑端
    d_black = -np.log10(np.maximum(_scan_white(), eps))   # 1,1,1 → 白端

    # 归一化反转：场景高光（负片低透光、高密度 D≈d_black）→ pos=1，
    # 阴影（负片高透光、低密度 D≈d_white）→ pos=0
    pos = (d - d_white) / np.maximum(d_black - d_white, 1e-6)

    pos = np.clip(pos, 0.0, 1.0)
    out = pos ** gamma
    return (out * 255.0 + 0.5).astype(np.uint8)
# ---------- 主入口 ----------
def process(lux_r, lux_g, lux_b, grain_ss=2,
            exposure_offset=0.0, gamma=OUT_GAMMA, return_scan=False):
    """5213 光谱实验模型主入口。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（与 Helpers 输出一致）。
        grain_ss：颗粒超采样级（1=关闭，2=2x；默认 2）。
        exposure_offset：曝光偏移（档位，正=过曝）。
        gamma：反转后叠的显示伽马（默认 1.70）。
        return_scan：是否同时返回未反转的负片扫描（线性）；True 时返回
                     (反转 uint8 图, 线性扫描 float32)。
    返回：
        (H,W,3) uint8（RGB）；return_scan=True 时返回 (uint8, scan)。
    """
    if not (lux_r.shape == lux_g.shape == lux_b.shape):
        raise ValueError("lux_r/lux_g/lux_b 的形状必须一致")
    if grain_ss not in (1, 2):
        raise ValueError("grain_ss 只支持 1 / 2")

    lux_r = np.asarray(lux_r, dtype=np.float32)
    lux_g = np.asarray(lux_g, dtype=np.float32)
    lux_b = np.asarray(lux_b, dtype=np.float32)

    # 1. 光谱输入（固定开启）：CIE-RGB 色匹配函数伪发射源 → 实测灵敏度吸收积分
    from Spectral import migrate_rgb_primary
    e_r, e_g, e_b = migrate_rgb_primary(lux_r, lux_g, lux_b, film="5213")

    # 2. PSF 光扩散（线性物理模式：核心高斯 + K0 扩散晕 − 邻界，MTF 拟合）
    e_r, e_g, e_b = _apply_psf(e_r, e_g, e_b)

    # 3. 对数曝光工作区
    logh_r = _to_log_exposure(e_r, exposure_offset)
    logh_g = _to_log_exposure(e_g, exposure_offset)
    logh_b = _to_log_exposure(e_b, exposure_offset)

    # 4. 特性曲线 → Status M 密度
    d_r, d_g, d_b = _char_density(logh_r, logh_g, logh_b)

    # 5. 密度域颗粒（伽马聚簇+可分离高斯赋形+超采样+孔径闭环）
    d_r, d_g, d_b = _apply_grain(d_r, d_g, d_b, logh_r, logh_g, logh_b,
                                   ss=grain_ss)

    # 6. 输出端：染料光谱堆叠 + CIE-RGB 客观扫描（固定开启）→ 反转
    scan = _dye_stack_scan(d_r, d_g, d_b)
    out = _invert(scan, gamma=gamma)
    if return_scan:
        return out, scan
    return out
