#赛博请神
# Scanner/noritsu.py —— Noritsu LS-600 风格反转（逆向移植）
# 移植自 noritsu-tool（MIT License，noritsu-tool 仓库），按用户要求
# "原封不动"搬入本管线，仅新增与 Phos 的对接接口（render）。
# 与 Scanner/scanner.py 的"翻拍（基线）"平行并立：
#   基线 = 无风格倾向的理想扫描仪（三锚点物理校正，调试/科研用）；
#   本模块 = 真实冲扫机性格（密度域反转 + 灰世界自动白平衡 +
#   Noritsu 控制点色调曲线），模仿真实胶片玩家经手扫描仪后的观感。
#
# 处理链（与 2026-09-01 桌面对比样张所用的 noritsu-tool 默认参数一致）：
#   透光率扫描 → 密度 D = -log10(T)
#     → 自动 Dmin（片基/色罩：边框优先，否则低百分位）/ Dmax（高百分位）
#     → (D - Dmin)/(Dmax - Dmin) 归一化（互补色反转随密度域一并完成）
#     → 广义灰世界自动白平衡（Shades of Gray，Minkowski 幂平均，在色调曲线之前）
#     → Noritsu 控制点 S 曲线（contrast 1.05 / toe 0.02 / gamma 0.98）
#     → 亮度保护饱和度（sat=1 默认不动）
#     → 风格对比度（与基线共用 0.5 锚 S 曲线）→ uint8
import numpy as np

EPS = 1e-6


# ---------- 密度域反转（忠实移植 noritsu.invert） ----------
def to_density(linear, black=0.0):
    """线性透光率 [0,1] → 密度。black = 扫描仪黑电平（默认 0）。"""
    v = np.clip(linear - black, EPS, None)
    return -np.log10(v)


def film_valid_mask(linear, black_th=0.005, clear_th=0.9):
    """底片上有效像素掩码：排除扫描黑边（≈0）和透明边距（≈1）。"""
    mx = linear.max(axis=2)
    return (mx < clear_th) & (mx > black_th)


def border_mask(shape, frac=0.03):
    """外圈（胶片边缘/片基边框）掩码。"""
    h, w = shape[0], shape[1]
    m = np.zeros((h, w), dtype=bool)
    bh = max(1, int(h * frac))
    bw = max(1, int(w * frac))
    m[:bh, :] = True
    m[-bh:, :] = True
    m[:, :bw] = True
    m[:, -bw:] = True
    return m


def estimate_dmin_border(density, frac=0.03):
    """边框若是均匀橙罩（B>G>R、低方差），用其中位数作 Dmin；否则 None。"""
    m = border_mask(density.shape, frac)
    d = density[m]
    med = np.median(d, axis=0)
    std = d.std(axis=0)
    if med[2] > med[1] >= med[0] - 1e-3 and med.mean() > 0.1 and std.mean() < 0.15:
        return med.astype(np.float32)
    return None


def estimate_percentile(density, pct, mask=None):
    d = density if mask is None else density[mask]
    return np.percentile(d.reshape(-1, 3), pct, axis=0)


def invert_negative(linear, dmin_pct=0.5, dmax_pct=99.8, use_border=True):
    """线性 C-41 负片扫描 → 密度归一化正像 [0,1]（高值 = 亮场景）。

    色罩（Dmin）从边框或低百分位估计，Dmax 取高百分位；
    (D - Dmin)/(Dmax - Dmin) 同时完成互补色反转与影调归一化。
    """
    density = to_density(linear)
    mask_dmin = film_valid_mask(linear, black_th=0.0)
    mask_dmax = film_valid_mask(linear, black_th=0.0005)

    dmin = None
    if use_border:
        dmin = estimate_dmin_border(density)
    if dmin is None:
        dmin = estimate_percentile(density, dmin_pct, mask_dmin)
    dmax = estimate_percentile(density, dmax_pct, mask_dmax)

    rng = dmax - dmin
    rng = np.where(rng < 0.05, 0.05, rng)
    positive = (density - dmin[None, None, :]) / rng[None, None, :]
    return np.clip(positive, 0.0, 1.0).astype(np.float32)


# ---------- 广义灰世界自动白平衡（Shades of Gray） ----------
def auto_wb_gains(positive, p=4.0):
    """广义灰世界：Minkowski 幂平均中性化。

    经典灰世界是算术平均（p=1）；Shades of Gray 用幂平均
    mu_c = (mean(x_c^p))^(1/p) 估计每通道"灰水平"，再让三通道对齐：
    target = 三通道灰水平之均值，gain_c = target / mu_c。
    p>1 时较亮（更接近白点）的像素权重更高，对高饱和大色块、
    阴影色罩残留更稳健，同时不像纯白点法那样被过曝高光绑架。
    默认 p=4：胶片高光肩部带染色，p 不宜太大（白点法风险）。
    """
    x = np.clip(np.asarray(positive, dtype=np.float32), 0.0, None)
    mu = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)
    mu = np.clip(mu, 1e-4, None)
    target = mu.mean()
    return tuple(float(target / mu[c]) for c in range(3))


def white_balance(img, gains):
    g = np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(img * g, 0.0, 1.0)


# ---------- Noritsu 控制点色调曲线（忠实移植 noritsu.render） ----------
def noritsu_tone_curve(x, contrast=1.05, toe=0.02, gamma=0.98):
    """Noritsu 风格 S 曲线：微抬趾部、长直线中段、柔和肩部。

    control points 按 pcb_MakeGammaCurve2 的 20 点三段式思想缩成 7 点；
    与原工具默认完全一致（对比样张同款参数）。
    """
    x = np.clip(x, 0.0, 1.0)
    k = contrast
    pts_x = np.array([0.0, 0.08, 0.25, 0.5, 0.75, 0.92, 1.0], dtype=np.float32)
    mid = 0.5
    pts_y = mid + (pts_x - mid) * k
    pts_y = pts_y + toe * (1.0 - pts_x) * np.exp(-pts_x * 4.0)
    pts_y = np.clip(pts_y, 0.0, 1.0)
    for i in range(1, len(pts_y)):
        if pts_y[i] < pts_y[i - 1]:
            pts_y[i] = pts_y[i - 1]
    out = np.interp(x, pts_x, pts_y)
    if gamma != 1.0:
        out = out ** gamma
    return out.astype(np.float32)


# ---------- 亮度保护饱和度（忠实移植 noritsu.render） ----------
def adjust_saturation(img, sat):
    y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    cb = (img[..., 2] - y) * 0.564
    cr = (img[..., 0] - y) * 0.713
    cb *= sat
    cr *= sat
    b = y + cb / 0.564
    r = y + cr / 0.713
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


# ---------- 与 Phos 的对接接口 ----------
def render(scan, contrast=0.0, dmin_pct=0.5, dmax_pct=99.8, use_border=True,
           tone_contrast=1.05, tone_toe=0.02, tone_gamma=0.98,
           saturation=1.0, auto_wb=True, wb_p=4.0):
    """Noritsu 扫描仪反转：未处理负片扫描 → uint8 成片。

    scan：第二段输出的线性透光率扫描（彩负语义）。
    contrast：风格对比度（与基线共用，0.5 锚 S 曲线，0 线性）。
    wb_p：广义灰世界幂指数（p=1 即经典灰世界；默认 4）。
    其余参数为 Noritsu 性格配方，默认与 noritsu-tool CLI 默认一致。
    """
    from .scanner import _encode_display
    positive = invert_negative(scan, dmin_pct=dmin_pct, dmax_pct=dmax_pct,
                               use_border=use_border)
    if auto_wb:
        positive = white_balance(positive, auto_wb_gains(positive, p=wb_p))
    pos = np.stack(
        [noritsu_tone_curve(positive[..., c], tone_contrast, tone_toe,
                            tone_gamma) for c in range(3)], axis=-1)
    if abs(saturation - 1.0) > 1e-4:
        pos = adjust_saturation(pos, saturation)
    return _encode_display(pos, contrast)
