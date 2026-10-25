#赛博请神
# psf_engine.py —— 光扩散（PSF）与邻域增强引擎（0.3.4 定稿版）
#
# 管线语义：
#   OFF    ：0.3.3 legacy 核 + 高斯堆卷积（逐位一致）
#   A（邻域效应）：legacy 基础核不变，特性曲线后叠加密度域单 DoG 邻界
#   —— DoG = a·(G(σ窄) − G(σ宽))，零 DC；运行时合成复合核、每通道一次
#      FFT 卷积（替代两次高斯模糊，效果逐位等价、性能更好）。
#
# 标定（fit_adjacency_density）：固定 legacy 基础核，只优化 a/σ窄/σ宽，
# 目标 MTF_legacy × (1 + a·DoG) = datasheet（≤20 cyc/mm 主权重），
# 并带**过冲预算约束**：a ≤ 过冲预算 / 带通阶跃最大过冲 ——
# 大信号边缘不过冲（避免数码锐化感），小信号仍贴合 datasheet。

import os
import pickle
import math

import cv2
import numpy as np

from Cache import stage_cache

# ---------- 基础函数（numpy 实现，无 scipy） ----------


def k0(x):
    """修正贝塞尔 K0（Abramowitz-Stegun 9.8）。"""
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


def gaussian_kernel(sigma_px, size=None):
    """归一化二维高斯核（与 cv2.GaussianBlur 归一化一致）。"""
    if size is None:
        size = max(int(2 * np.ceil(3.0 * sigma_px)) + 1, 3)
    k1 = cv2.getGaussianKernel(size, sigma_px)
    return (k1 @ k1.T).astype(np.float64)


def solve_nonneg(A, b):
    """活动集式非负最小二乘。"""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
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


def kernel_mtf(kernel, freq, px_per_mm=100.0):
    """核实际 FFT → datasheet 频率上的 DC 归一 MTF。"""
    size = kernel.shape[0]
    n = 1
    while n < max(1024, 4 * size):
        n *= 2
    k = np.zeros((n, n), dtype=np.float64)
    cy = size // 2
    k[n // 2 - cy:n // 2 - cy + size, n // 2 - cy:n // 2 - cy + size] = kernel
    f = np.abs(np.fft.fftshift(np.fft.fft2(k)))
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - n // 2, yy - n // 2)
    freq_ring = np.arange(n // 2) * (px_per_mm / n)
    dc = f[n // 2, n // 2]
    out = []
    for ff in freq:
        ri = int(np.argmin(np.abs(freq_ring - ff)))
        ring = (r >= ri - 0.5) & (r < ri + 0.5)
        out.append(f[ring].mean() / dc if ring.any() else 1.0)
    return np.array(out, dtype=np.float64)


# ---------- 0.3.3 legacy 精确路径（OFF 逐位一致） ----------


def load_legacy_v1(path):
    """读取 0.3.3 旧 PSF 参数（µm→mm），返回 {"R"/…: (σc,δ,σa,w_c,w_d,w_a)}。"""
    d = np.load(path)
    um = 1e-3
    if "layer" in d.files:
        layers = list(d["layer"])
        return {str(layer): (float(d["sigma_core_um"][i]) * um,
                             float(d["delta_um"][i]) * um,
                             float(d["sigma_adj_um"][i]) * um,
                             float(d["w_core"][i]), float(d["w_halo"][i]),
                             float(d["w_adj"][i]))
                for i, layer in enumerate(layers)}
    return {"R": (float(d["sigma_core_um"]) * um, float(d["delta_um"]) * um,
                  float(d["sigma_adj_um"]) * um, float(d["w_core"]),
                  float(d["w_halo"]), float(d["w_adj"]))}


def build_legacy_kernel(v1, px_per_mm):
    """0.3.3 旧核公式（w_c·G + w_d·K0 − w_a·G_adj，归一化），逐位一致。"""
    s_c, dl, s_a, w_c, w_d, w_a = v1
    s_c_px = s_c * px_per_mm
    dl_px = dl * px_per_mm
    s_a_px = s_a * px_per_mm
    radius = max(s_c_px, dl_px, s_a_px)
    size = max(int(2 * np.ceil(3.0 * radius)) + 1, 3)
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(xx, yy)
    core = gaussian_kernel(s_c_px, size)
    halo = k0(np.maximum(rr, 0.5) / dl_px)
    halo = halo / halo.sum()
    adj = gaussian_kernel(s_a_px, size)
    kernel = w_c * core + w_d * halo - w_a * adj
    kernel = kernel / np.maximum(kernel.sum(), 1e-6)
    return kernel.astype(np.float32)


def gauss_decomp_v1(kernel, sigma_adj=None, n_pos=8):
    """0.3.3 旧高斯堆分解（n_pos=8 + 单个负邻界），逐位一致。"""
    k = kernel.astype(np.float64)
    h, w = k.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
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
    w = solve_nonneg(np.column_stack(cols), k.ravel())
    out = []
    for wi, (sign, s) in zip(w, names):
        if abs(wi) > 1e-6:
            out.append((sign, float(wi), s))
    return out


# ---------- 邻域效应（方案 A）：密度域单 DoG，过冲预算标定 ----------

_ADJ_VERSION = 3
_ADJ_SPECTRUM_CACHE = {}
_ADJ_SPECTRUM_CACHE_MAX = 32


def _step_overshoot(sn, sw):
    """带通 G(σn)−G(σw) 对理想阶跃(0→1)的最大过冲（一维闭式）。

    2D 高斯模糊阶跃 = 0.5·erf(x/(σ√2))；带通阶跃响应 = 0.5·(erf(x/(σn√2))
    − erf(x/(σw√2)))，其最大绝对值为过冲系数 M，a·M 即实际过冲。
    """
    x = np.linspace(-6.0, 6.0, 2001)
    erf = np.vectorize(math.erf)
    resp = 0.5 * (erf(x / (max(sn, 1e-9) * np.sqrt(2)))
                  - erf(x / (sw * np.sqrt(2))))
    return float(np.max(np.abs(resp)))


def _clamp_adj_a(a, sn, sw, f, base, twopi2, budget, ceiling):
    """把增益 a 压到两个物理约束内：
    ① 阶跃响应过冲 ≤ budget（防数码锐化感的振铃）；
    ② 组合 MTF 峰值 ≤ ceiling（datasheet 峰值，任何频率的增强
       不超过胶片的实测标定增强）。
    """
    a = min(a, budget / max(_step_overshoot(sn, sw), 1e-9))
    dog = np.exp(-twopi2 * sn ** 2 * f ** 2) \
        - np.exp(-twopi2 * sw ** 2 * f ** 2)
    pos = dog > 1e-12
    if np.any(pos):
        denom = base[pos] * dog[pos]
        numer = ceiling - base[pos]
        cand = np.full(int(np.sum(pos)), np.inf)
        np.divide(numer, denom, out=cand, where=denom > 1e-12)
        a = min(a, max(float(cand.min()), 0.0))
    return max(a, 0.0)


def fit_adjacency_density(freq, resp, v1, px_per_mm=100.0,
                          overshoot_budget=0.15, mtf_ceiling=None,
                          verbose=False):
    """按密度域组合目标拟合邻界参数（a/σ窄/σ宽）。

    目标：MTF_legacy × (1 + a·DoG) = datasheet，≤20 cyc/mm 主权重；
    约束①：阶跃响应过冲 ≤ overshoot_budget（默认 15%）；
    约束②：组合 MTF 峰值 ≤ mtf_ceiling（默认取 datasheet 响应峰值）。
    返回 {"sigmas_mm": {sn, sw}, "weights": {a}, "overshoot": 实际过冲}；
    若无可行正增益则返回零增益（保持 legacy 基线，不硬增强）。
    """
    f = np.asarray(freq, dtype=np.float64)
    y = np.asarray(resp, dtype=np.float64)
    mask = f <= 0.8 * px_per_mm / 2.0
    f = f[mask]
    y = y[mask]
    wgt = np.where(f <= 20.0, 1.0, 0.5)
    base = kernel_mtf(build_legacy_kernel(v1, px_per_mm), f, px_per_mm)
    ceiling = float(np.max(y)) if mtf_ceiling is None else float(mtf_ceiling)
    twopi2 = 2.0 * np.pi ** 2

    def a_optimal(sn, sw):
        dog = np.exp(-twopi2 * sn ** 2 * f ** 2) \
            - np.exp(-twopi2 * sw ** 2 * f ** 2)
        ww = wgt * base * dog
        num = float(np.sum(ww * (y - base)))
        den = float(np.sum(ww ** 2))
        return (num / den) if den > 1e-12 else 0.0

    best = None
    for sn in np.logspace(np.log10(1.5e-3), np.log10(15.0e-3), 9):
        for sw in np.logspace(np.log10(30.0e-3), np.log10(300.0e-3), 9):
            a = _clamp_adj_a(a_optimal(sn, sw), sn, sw, f, base,
                             twopi2, overshoot_budget, ceiling)
            if a <= 0:
                continue
            dog = np.exp(-twopi2 * sn ** 2 * f ** 2) \
                - np.exp(-twopi2 * sw ** 2 * f ** 2)
            err = float(np.mean(((base * (1.0 + a * dog) - y) * wgt) ** 2))
            cand = (err, sn, sw, a)
            if best is None or err < best[0]:
                best = cand
    if best is None:
        return {"sigmas_mm": {"sn": 1.0e-3, "sw": 30.0e-3},
                "weights": {"a": 0.0}, "overshoot": 0.0}
    _, sn, sw, a = best

    # 坐标下降细化（对数空间，a 每次重解并受两个约束）
    step = 0.25
    while step > 0.01:
        improved = False
        for name, cur in (("sn", sn), ("sw", sw)):
            lo, hi = (1.0e-3, 20.0e-3) if name == "sn" else (20.0e-3, 500.0e-3)
            for dd in (-step, step):
                val = float(np.clip(cur * 10.0 ** dd, lo, hi))
                a2 = _clamp_adj_a(
                    a_optimal(sn if name == "sw" else val,
                              val if name == "sw" else sw),
                    sn if name == "sw" else val,
                    val if name == "sw" else sw,
                    f, base, twopi2, overshoot_budget, ceiling)
                dog = np.exp(-twopi2 * (sn if name == "sw" else val) ** 2 * f ** 2) \
                    - np.exp(-twopi2 * (val if name == "sw" else sw) ** 2 * f ** 2)
                err = float(np.mean(
                    ((base * (1.0 + a2 * dog) - y) * wgt) ** 2))
                if err < best[0] - 1e-12:
                    best = (err, sn if name == "sw" else val,
                            val if name == "sw" else sw, a2)
                    sn, sw, a = best[1], best[2], best[3]
                    improved = True
        if not improved:
            step *= 0.5

    _, sn, sw, a = best
    actual_os = a * _step_overshoot(sn, sw)
    if verbose:
        dog = np.exp(-twopi2 * sn ** 2 * f ** 2) \
            - np.exp(-twopi2 * sw ** 2 * f ** 2)
        comb = base * (1.0 + a * dog)
        print(f"  adj: a={a:.3f} σ窄={sn*1e3:.2f}µm σ宽={sw*1e3:.1f}µm "
              f"过冲={actual_os:.4f} "
              f"组合峰={float(np.max(comb)):.4f}/限={ceiling:.4f} "
              f"误差≤20={np.abs(comb[f<=20]-y[f<=20]).max():.4f}")
    return {"sigmas_mm": {"sn": float(sn), "sw": float(sw)},
            "weights": {"a": float(a)}, "overshoot": float(actual_os)}


def load_or_fit_adjacency(film_key, mtf_layers, px_per_mm, v1_file, cache_file,
                          overshoot_budget=0.15):
    """按层拟合/加载邻域效应参数（MTF 指纹 + v1 参数 + 预算变化失效）。"""
    v1 = load_legacy_v1(v1_file)
    key = {"film": film_key,
           "mtf": {k: (list(f), list(r)) for k, (f, r) in mtf_layers.items()},
           "v1": {k: list(t) for k, t in v1.items()},
           "os": float(overshoot_budget)}
    if os.path.exists(cache_file):
        try:
            d = np.load(cache_file, allow_pickle=True)
            if int(d["adj_version"]) == _ADJ_VERSION \
                    and bool(d["mtf_fp"] == stage_cache.key_hash(key)):
                return pickle_load(d)
        except Exception:
            pass
    result = {}
    for layer, (freq, resp) in mtf_layers.items():
        print(f"  [psf_engine] 邻域效应拟合 {film_key}/{layer} "
              f"(过冲预算 {overshoot_budget:.2f}) ...")
        result[layer] = fit_adjacency_density(
            np.asarray(freq), np.asarray(resp) / 100.0, v1[layer],
            px_per_mm, overshoot_budget=overshoot_budget, verbose=True)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez(cache_file, adj_version=np.int32(_ADJ_VERSION),
             mtf_fp=stage_cache.key_hash(key),
             payload=np.frombuffer(pickle.dumps(result), dtype=np.uint8))
    return result


def apply_overrides(params, opts):
    """把运行时选项应用到拟合参数：override 数值覆盖 + 强度缩放。"""
    ov = (opts or {}).get("override") or {}
    out = {"sigmas_mm": dict(params["sigmas_mm"]),
           "weights": dict(params["weights"])}
    if ov:
        for k, v in ov.items():
            if k in out["sigmas_mm"]:
                out["sigmas_mm"][k] = float(v)
            elif k in out["weights"]:
                out["weights"][k] = float(v)
    # 邻域强度：0.5 = 增益减半（观感保守档），0 = 关闭
    strength = float((opts or {}).get("adjacency_strength", 1.0))
    out["weights"]["a"] = out["weights"]["a"] * strength
    return out


def build_adjacency_kernel(fit, px_per_mm):
    """复合核：C = δ + a·(G(σ窄) − G(σ宽))（零 DC、可正可负、ΣC=1）。"""
    a = fit["weights"]["a"]
    if a <= 0:
        return None
    sn = fit["sigmas_mm"]["sn"] * px_per_mm
    sw = fit["sigmas_mm"]["sw"] * px_per_mm
    size = max(int(2 * np.ceil(3.0 * sw)) + 1, 3)
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(xx, yy)
    k = np.zeros((size, size), np.float64)
    k[half, half] = 1.0
    gn = np.exp(-rr ** 2 / (2.0 * sn ** 2))
    k = k + a * (gn / gn.sum())
    gw = np.exp(-rr ** 2 / (2.0 * sw ** 2))
    k = k - a * (gw / gw.sum())
    return k.astype(np.float32)


def _adj_spectrum(fit, px_per_mm, fh, fw):
    """复合核的 rfft2 频谱，按（拟合参数、像素尺度、FFT 尺寸）缓存跨通道复用。"""
    a = fit["weights"]["a"]
    sn = fit["sigmas_mm"]["sn"] * px_per_mm
    sw = fit["sigmas_mm"]["sw"] * px_per_mm
    key = (round(float(a), 10), round(float(sn), 6), round(float(sw), 6),
           int(fh), int(fw))
    spec = _ADJ_SPECTRUM_CACHE.get(key)
    if spec is not None:
        return spec
    kern = build_adjacency_kernel(fit, px_per_mm)
    if kern is None:
        return None
    kh, kw = kern.shape
    k_p = np.zeros((fh, fw), np.float32)
    k_p[:kh, :kw] = kern[::-1, ::-1]
    spec = np.fft.rfft2(k_p)
    if len(_ADJ_SPECTRUM_CACHE) >= _ADJ_SPECTRUM_CACHE_MAX:
        _ADJ_SPECTRUM_CACHE.clear()
    _ADJ_SPECTRUM_CACHE[key] = spec
    return spec


def apply_adjacency(d, fit, px_per_mm):
    """密度域邻域增强：D′ = C ⊛ D，一次 rfft2 卷积（核频谱缓存，替代两次模糊）。

    与 cv2.dft 全复变换的偏差 ~1e-6（BORDER_REFLECT 补边一致），
    单通道 3024×4536 约 0.31s（cv2 版约 1.2s），三通道共享核频谱再省 1/3。
    """
    kern = build_adjacency_kernel(fit, px_per_mm)
    if kern is None:
        return d
    h, w = d.shape
    kh, kw = kern.shape
    ph, pw = kh // 2, kw // 2
    img_b = cv2.copyMakeBorder(d, ph, kh - 1 - ph, pw, kw - 1 - pw,
                               borderType=cv2.BORDER_REFLECT)
    fh = cv2.getOptimalDFTSize(h + kh - 1)
    fw = cv2.getOptimalDFTSize(w + kw - 1)
    img_p = np.zeros((fh, fw), np.float32)
    img_p[:h + kh - 1, :w + kw - 1] = img_b
    spec = _adj_spectrum(fit, px_per_mm, fh, fw)
    out = np.fft.irfft2(np.fft.rfft2(img_p) * spec, s=(fh, fw))
    return out[kh - 1:kh - 1 + h, kw - 1:kw - 1 + w].astype(np.float32)


def pickle_load(d):
    return pickle.loads(d["payload"].tobytes())
