#赛博请神
# Scanner/scanner.py —— 第三段（独立扫描仪模块）：扫描 → 成片
# 职责：把第二段（胶片模拟）输出的"未处理扫描"渲染成最终显示成片。
# 三锚点平滑校正（0.3.3 纯物理版）：
#   每通道（黑/灰/白）三个密度锚点构成一条 PCHIP 平滑曲线（PS 拉曲线式），
#   保证输入黑点→输出黑、输入灰点（18% 灰）→输出中灰（sRGB 118）、
#   输入白点→输出白；中间影调只受三点约束，顺其自然（保留胶片特性）。
# 锚点由第二段标定（0,0,0 / E_mid=1 / E_mid=6.24 输入走管线实测），
# 存于 FILM_PROFILES；风格化对比度（0.5 锚点 S 曲线）叠加在校正之后。
# 锚点正确时曝光由三锚点保证，无需 gamma 编码。
import numpy as np


# ---------- 胶片扫描元数据（三锚点，第二段标定，2026-08-29 纯物理版） ----------
# 均为密度域锚点 D = -log10(扫描值)（黑白为特性曲线密度本身）：
#   black：0,0,0 输入（片基/未曝光）→ 输出黑
#   gray： E_mid=1（18% 灰）输入 → 输出中灰（sRGB 118）
#   white：胶片自身高光白点（特性曲线高曝光端，对正片即 Dmin）→ 输出白
#          （标定输入 E_mid=6.24，+2.64 档）
# 注意：传感器饱和在 E_mid 中的位置随 BaselineExposure 变化
# （无 BE 时 ≈7.80，+2.97 档），白锚点是胶片参考而非传感器参考，
# 因此保持静态；正片（provia100f）密度随曝光下降，
# black > gray > white，曲线自动排序。
FILM_PROFILES = {
    "5213": {
        "kind": "color_negative",
        "black": np.array([0.19303, 0.58123, 0.86273], np.float32),
        "gray": np.array([0.70097, 1.33770, 1.67244], np.float32),
        "white": np.array([0.97792, 1.73950, 2.14993], np.float32),
    },
    "gold200": {
        "kind": "color_negative",
        "black": np.array([0.23908, 0.62761, 0.96540], np.float32),
        "gray": np.array([0.53693, 1.15090, 1.55571], np.float32),
        "white": np.array([0.78872, 1.55689, 2.13059], np.float32),
    },
    "provia100f": {
        "kind": "color_positive",
        "black": np.array([3.43603, 3.80905, 4.02428], np.float32),
        "gray": np.array([0.93828, 1.01484, 1.10056], np.float32),
        "white": np.array([0.13816, 0.14014, 0.14528], np.float32),
    },
    "hp5": {
        "kind": "bw_negative",
        "black": 0.12670,
        "gray": 0.78990,
        "white": 1.30260,
    },
}

# 中灰输出参考：sRGB 118 / 255（gamma=1.0 语义下 18% 灰 → 输出 118）
_MID = 118.0 / 255.0


# ---------- 三锚点平滑校正（Fritsch–Carlson 单调保形三次插值，PCHIP 同款） ----------
def _monotone_cubic(xs, ys, x):
    """Fritsch–Carlson 单调保形三次插值（PCHIP 同款，PS 拉曲线式）。

    锚点 (xs, ys) 必须按 xs 严格递增；对 x 向量插值，边界外推由调用方 clip。
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(xs)
    if n < 2:
        raise ValueError("锚点至少需要 2 个")
    h = np.diff(xs)
    delta = np.diff(ys) / h
    # 内部点导数：保单调（异号→0，同号→调和平均）
    m = np.empty(n)
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            m[i] = 2.0 / (1.0 / delta[i - 1] + 1.0 / delta[i])
    # 端点导数（三点差分）＋单调裁剪
    m[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    m[n - 1] = ((2.0 * h[n - 2] + h[n - 3]) * delta[n - 2]
                - h[n - 2] * delta[n - 3]) / (h[n - 2] + h[n - 3])
    if m[0] * delta[0] <= 0.0:
        m[0] = 0.0
    elif abs(m[0]) > 3.0 * abs(delta[0]):
        m[0] = 3.0 * delta[0] * np.sign(m[0])
    if m[n - 1] * delta[n - 2] <= 0.0:
        m[n - 1] = 0.0
    elif abs(m[n - 1]) > 3.0 * abs(delta[n - 2]):
        m[n - 1] = 3.0 * delta[n - 2] * np.sign(m[n - 1])
    # 分段三次 Hermite（向量化）
    idx = np.searchsorted(xs, x, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    t = (x - xs[idx]) / h[idx]
    h0 = h[idx]
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t ** 2 * (3.0 - 2.0 * t)
    h11 = t ** 2 * (t - 1.0)
    return (ys[idx] * h00 + m[idx] * h0 * h10
            + ys[idx + 1] * h01 + m[idx + 1] * h0 * h11)


def _three_anchor_curve(d, black, gray, white, mid=_MID):
    """每通道三锚点（黑→0、灰→mid、白→1）→ 平滑校正曲线（自动按 D 排序）。

    边界截断：超出锚点范围的密度直接取端点输出（黑/白），
    避免三次 Hermite 外推产生异常值（例如扫描≈0 → 密度极大 →
    多项式外推错误上扬，造成色相硬跳变）。
    """
    pairs = sorted(zip([black, gray, white], [0.0, mid, 1.0]))
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)
    d_clip = np.clip(np.asarray(d, dtype=np.float64), xs[0], xs[-1])
    pos = _monotone_cubic(xs, ys, d_clip)
    return np.clip(pos, 0.0, 1.0)


# ---------- 高光收口说明 ----------
# uint8 的 255 上限下，过曝一侧由两处负责：
#   - 特性曲线肩部（高光密度趋平，灰锚与白锚之间已有渐变）；
#   - 三锚点 PCHIP 白端截断（超过白锚密度 → 输出白）。
# 完整过曝梯度走 16bit 原片导出（scan 为密度域 HDR，不 clip）。


def _encode_display(pos, contrast=0.0):
    """校正后 pos → uint8（风格对比度 S 曲线 + 最终夹取）。

    对比度以 0.5 为锚叠正反 S 曲线（±100%）：
      u = 2(pos−0.5)，u' = sign(u)·|u|^p，p = 1/(1+contrast)
      contrast=0 线性；>0 增对比（暗更暗亮更亮）；<0 减对比（压向中灰）。
    曝光正确性由三锚点保证，此处纯为风格层。
    """
    pos = np.asarray(pos, dtype=np.float32)
    c = float(np.clip(contrast, -0.99, 1.0))
    p = 1.0 / (1.0 + c)
    u = 2.0 * (pos - 0.5)
    u2 = np.sign(u) * np.abs(u) ** p
    v = (u2 + 1.0) / 2.0
    return (np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def get_profile(film):
    """返回某胶片的扫描元数据字典。"""
    if film not in FILM_PROFILES:
        raise ValueError(
            f"未知胶片：{film}（扫描仪模块支持 "
            f"{sorted(FILM_PROFILES)}）")
    return FILM_PROFILES[film]


def render_negative(scan, black, gray, white, contrast=0.0):
    """彩负：对数密度域三锚点平滑校正 + 风格对比度。

    1. 密度化：D = -log10(scan)（色罩在密度域是加法常数）；
    2. 三锚点 PCHIP 曲线：黑端(0,0,0)→0、中灰(18% 灰)→118/255、
       白端(1,1,1)→1，每通道独立（黑端去色罩、中灰校色平衡）；
    3. 中间影调只受三点约束，响应曲线的暗部青/高光暖自然保留。
    """
    d = -np.log10(np.maximum(np.asarray(scan, dtype=np.float32), 1e-6))
    pos = np.empty_like(d)
    for c in range(3):
        pos[..., c] = _three_anchor_curve(
            d[..., c], float(black[c]), float(gray[c]), float(white[c]))
    return _encode_display(pos, contrast)


def render_positive(scan, black, gray, white, contrast=0.0):
    """正片：无反转，扫描值即亮度；三锚点平滑校正 + 风格对比度。

    正片密度随曝光下降（black > gray > white），锚点自动按 D 排序，
    数学上与彩负统一：0,0,0 → 黑、18% 灰 → 中灰、1,1,1 → 白。
    """
    d = -np.log10(np.maximum(np.asarray(scan, dtype=np.float32), 1e-6))
    pos = np.empty_like(d)
    for c in range(3):
        pos[..., c] = _three_anchor_curve(
            d[..., c], float(black[c]), float(gray[c]), float(white[c]))
    return _encode_display(pos, contrast)


def render_bw(density, black, gray, white, contrast=0.0):
    """黑白：密度域三锚点平滑校正 + 风格对比度（单通道）。"""
    pos = _three_anchor_curve(np.asarray(density, dtype=np.float32),
                              float(black), float(gray), float(white))
    return _encode_display(pos, contrast)


def render(scan, film, contrast=0.0):
    """第三段主入口：未处理扫描 → 成片。

    参数：
        scan：第二段输出的扫描（float32；彩负/正片为透光率语义，
              黑白为密度域单通道）。
        film：胶片键名（"5213"/"gold200"/"provia100f"/"hp5"）。
        contrast：风格对比度（±100%，0 线性）。
    返回：
        (H,W) 或 (H,W,3) uint8 成片。
    """
    profile = get_profile(film)
    kind = profile["kind"]
    if kind == "color_negative":
        return render_negative(scan, profile["black"], profile["gray"],
                               profile["white"], contrast)
    if kind == "color_positive":
        return render_positive(scan, profile["black"], profile["gray"],
                               profile["white"], contrast)
    if kind == "bw_negative":
        return render_bw(scan, profile["black"], profile["gray"],
                         profile["white"], contrast)
    raise ValueError(f"未知扫描类型：{kind}（胶片 {film}）")
