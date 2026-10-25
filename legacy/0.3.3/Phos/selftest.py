#赛博请神
# selftest.py —— 0.3.3 自检脚本
# 运行方式：在 Phos/0.3.3/Phos 目录下执行  python selftest.py
# 检查内容：
#   1. CIE 常量与白点
#   2. 5213 模型固定管线与参数响应
#   3. MTF 数据与 PSF 拟合
#   4. 颗粒：孔径闭环与超采样
#   5. HP5 旧模型不变性 + Gold200 新结构烟测
#   6. Provia 100F 正片模型验证（无反转、显示编码）
#   7. 光谱重建迁移（RGB→曝光统计映射，唯一路径）与色卡验证
#   8. 三段式一致性：Input → process_scan → Scanner.render
#      与便捷入口 process() 输出逐位一致（0.3.3 架构验证）

import importlib
import os
import sys

import cv2
import numpy as np

# 允许直接以脚本方式运行（当前目录即 Phos 包根目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# numpy 2.2.6 在部分平台会对含零矩阵的 matmul 误报，统一屏蔽
np.seterr(over="ignore", divide="ignore", invalid="ignore")

from Spectral import cie_data  # noqa: E402

# 输入契约（0.3.3 纯物理版）：E_mid 相对中灰（中灰=1.0，饱和=7.80）
E_MID_WHITE = 1.0 / (10.0 / 78.0)   # ≈ 7.80（DNG 饱和，stop +2.97）
E_MID_GRAY = 1.0                    # 18% 灰（stop 0）


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


def test_cie_data():
    print("\n== 1. CIE 常量 ==")
    ok = True
    ok &= check("波长数量为 61", len(cie_data.WAVELENGTHS) == 61)
    ok &= check("白点 XYZ 与 sRGB 标准白点一致", cie_data.verify_constants(),
                detail=f"实际 {np.round(cie_data.WHITE_XYZ, 4)}")
    ok &= check("CIE RGB 色匹配函数形状 (61,3)",
                cie_data.RGB_CMF_1931_2.shape == (61, 3))
    ok &= check("r̄ 含负瓣（440~530nm 为负）",
                cie_data.RGB_CMF_1931_2[:, 0].min() < -0.05)
    i600 = int(np.argmin(np.abs(cie_data.WAVELENGTHS - 600.0)))
    ok &= check("r̄(600)≈0.344（标准值）",
                abs(cie_data.RGB_CMF_1931_2[i600, 0] - 0.3443) < 0.01,
                detail=f"{cie_data.RGB_CMF_1931_2[i600, 0]:.4f}")
    return ok


def test_5213_model():
    print("\n== 2. 5213 模型（固定管线） ==")
    model = importlib.import_module("Films.5213.5213")
    ok = True

    # 合成测试图：白/中灰/纯红/纯绿/纯蓝/黑
    # 每块 64 像素宽、96 像素高，避免 PSF 核把相邻色块糊在一起
    patches = {
        "白": (E_MID_WHITE, E_MID_WHITE, E_MID_WHITE),
        "中灰": (E_MID_GRAY, E_MID_GRAY, E_MID_GRAY),
        "纯红": (E_MID_WHITE, 0.0, 0.0),
        "纯绿": (0.0, E_MID_WHITE, 0.0),
        "纯蓝": (0.0, 0.0, E_MID_WHITE),
        "黑": (0.0, 0.0, 0.0),
    }
    block_w = 64
    img = np.zeros((96, block_w * len(patches), 3), dtype=np.float32)
    for i, (name, rgb) in enumerate(patches.items()):
        img[:, i * block_w:(i + 1) * block_w] = rgb
    r, g, b = [img[:, :, i] for i in range(3)]

    # 4.1 默认调用（光谱输入/染料堆叠/客观扫描/反转固定开启）
    film = model.process(r, g, b)
    ok &= check("默认输出形状正确 uint8",
                film.shape == img.shape[:2] + (3,) and film.dtype == np.uint8)
    ok &= check("默认输出有限", np.isfinite(film).all())

    # 默认调用应等价于显式参数（光谱输入开、颗粒、曝光/伽马默认）
    np.random.seed(0)
    film_default = model.process(r, g, b)
    np.random.seed(0)
    film_ref = model.process(
        r, g, b, grain_ss=2,
        exposure_offset=0.0, contrast=0.0,
    )
    ok &= check("默认调用等价于显式参数", np.array_equal(film_default, film_ref))

    # 光谱输入开启时，纯红/纯蓝应分别驱动红/蓝感光层
    np.random.seed(0)
    film_spec = model.process(r, g, b)
    red = film_spec[48, 2 * block_w + block_w // 2].astype(float)
    blue = film_spec[48, 4 * block_w + block_w // 2].astype(float)
    ok &= check("光谱输入开启：纯红输出红通道最高",
                red[0] > red[1] and red[0] > red[2],
                detail=f"{red}")
    ok &= check("光谱输入开启：纯蓝输出蓝通道最高",
                blue[2] > blue[1] and blue[2] > blue[0],
                detail=f"{blue}")
    # 光谱重建映射：矩阵有限、白场（1,1,1）三层曝光=1
    from Spectral import reconstruct as rc
    mat = rc._get_map("5213")
    white_exp = np.array([1.0, 1.0, 1.0, 0.0]) @ mat  # 行 R/G/B/1，列 B/G/R
    ok &= check("光谱重建映射矩阵数值有限",
                bool(np.isfinite(mat).all()),
                detail=f"{np.round(mat, 3)}")
    ok &= check("光谱重建映射白场三层曝光=1",
                bool(np.allclose(white_exp, 1.0, atol=1e-4)),
                detail=f"{np.round(white_exp, 5)}")

    # 4.2 正片方向与色彩指向（默认参数）
    film = model.process(r, g, b)
    white_px = film[48, block_w // 2].astype(float)
    black_px = film[48, 5 * block_w + block_w // 2].astype(float)
    ok &= check("白块亮、黑块暗（正片方向）",
                white_px.mean() > 150 and black_px.mean() < 80,
                detail=f"白 {white_px} / 黑 {black_px}")
    red_dye = film[48, 2 * block_w + block_w // 2].astype(float)
    blue_dye = film[48, 4 * block_w + block_w // 2].astype(float)
    ok &= check("纯红输出红通道高于绿通道",
                red_dye[0] > red_dye[1],
                detail=f"{red_dye}")
    ok &= check("纯蓝输出蓝通道最高",
                blue_dye[2] > blue_dye[1] and blue_dye[2] > blue_dye[0],
                detail=f"{blue_dye}")
    # 无白平衡：中灰保留响应曲线带来的自然漂移（上限放宽）
    gray_dye = film[48, block_w + block_w // 2].astype(float)
    ok &= check("中灰通道差 < 120（白平衡关闭，保留自然漂移）",
                gray_dye.max() - gray_dye.min() < 120,
                detail=f"{gray_dye}")

    # 4.3 曝光偏移应改变输出（非全黑全白时）
    film_a = model.process(r, g, b, exposure_offset=0.0)
    film_b = model.process(r, g, b, exposure_offset=1.0)
    ok &= check("曝光偏移 +1 档会改变输出", not np.array_equal(film_a, film_b))

    # 4.4 风格对比度应改变输出
    film_g1 = model.process(r, g, b, contrast=0.6)
    film_g2 = model.process(r, g, b, contrast=-0.6)
    ok &= check("对比度改变输出", not np.array_equal(film_g1, film_g2))
    return ok


def test_mtf_psf():
    print("\n== 3. MTF 数据与 PSF 拟合 ==")
    model = importlib.import_module("Films.5213.5213")
    data_mod = importlib.import_module("Films.5213.5213_data")
    ok = True

    mtf = data_mod.get_mtf()
    dense = np.concatenate([
        mtf["response_R"], mtf["response_G"], mtf["response_B"]])
    ok &= check("MTF 稠密网格有限", bool(np.isfinite(dense).all()))
    ok &= check("MTF 响应在 [0,1.2]",
                bool(dense.min() >= 0.0 and dense.max() <= 1.2),
                detail=f"[{dense.min():.3f}, {dense.max():.3f}]")
    ok &= check("MTF 频率单调递增",
                bool(np.all(np.diff(mtf["frequency"]) > 0)))

    # 逐层拟合残差（物理核模型 vs 原始描点）
    fit = model._get_psf_fit()
    twopi2 = 2.0 * np.pi ** 2
    for layer in ("R", "G", "B"):
        f = mtf[f"freq_{layer}_raw"]
        y = mtf[f"resp_{layer}_raw"] / 100.0
        s_c, delta, s_a, w_c, w_d, w_a = fit[layer]
        pred = (w_c * np.exp(-twopi2 * s_c ** 2 * f ** 2)
                + w_d / (1.0 + (2.0 * np.pi * delta * f) ** 2)
                - w_a * np.exp(-twopi2 * s_a ** 2 * f ** 2))
        maxerr = float(np.max(np.abs(pred - y)))
        ok &= check(f"MTF 拟合 {layer} 最大误差 < 6%", maxerr < 0.06,
                    detail=f"{maxerr * 100:.2f}%")

    # 物理 PSF 核
    for layer in "RGB":
        kn = model._get_psf_kernel(layer)
        ok &= check(f"PSF {layer} 物理核有限且和=1",
                    bool(np.isfinite(kn).all())
                    and abs(float(kn.sum()) - 1.0) < 1e-3)
    return ok


def test_grain():
    print("\n== 4. 颗粒：孔径闭环与超采样 ==")
    model = importlib.import_module("Films.5213.5213")
    ok = True

    # 颗粒孔径闭环：两个超采样级都应回到 σ=1（另取种子验证）
    for ss in (1, 2):
        aper, scale = model._grain_parts(ss)
        c = 384
        rng = np.random.default_rng(1000 + ss)
        z = rng.standard_normal((c * ss, c * ss, 3)).astype(np.float32)
        corr = z @ model._grain_cholesky().T
        g = np.sign(corr) * (np.abs(corr) ** model._GRAIN_SHAPE_EXPONENT)
        n = model._grain_convolve(g, ss)
        nd = n.reshape(c, ss, c, ss, 3).mean(axis=(1, 3))
        ap = np.empty_like(nd)
        for ch in range(3):
            ap[..., ch] = cv2.filter2D(nd[..., ch], cv2.CV_32F, aper,
                                       borderType=cv2.BORDER_REFLECT)
        ok &= check(f"颗粒 ss={ss} 孔径闭环误差 < 5%",
                    abs(float(ap.std() * scale) - 1.0) < 0.05,
                    detail=f"孔径σ={ap.std() * scale:.4f}（目标1.0）")

    # 输出噪声：平坦中灰（E_mid=1），两次运行取差 /√2
    h = w = 512
    flat = np.full((h, w, 3), E_MID_GRAY, dtype=np.float32)

    def noise_sigma(ss):
        outs = []
        for seed in (1, 2):
            np.random.seed(seed)
            outs.append(model.process(flat[:, :, 0], flat[:, :, 1],
                                      flat[:, :, 2], grain_ss=ss))
        return (outs[0].astype(float) - outs[1].astype(float)
                ).std(axis=(0, 1)) / np.sqrt(2)

    s_grain = noise_sigma(2)
    ok &= check("颗粒输出噪声可见（平均 σ>2.0 灰度）",
                float(s_grain.mean()) > 2.0, detail=f"{s_grain.round(2)}")
    # 超采样开关：1x 与 2x 输出应不同
    np.random.seed(0)
    f_ss1 = model.process(flat[:, :, 0], flat[:, :, 1], flat[:, :, 2],
                         grain_ss=1)
    np.random.seed(0)
    f_ss2 = model.process(flat[:, :, 0], flat[:, :, 1], flat[:, :, 2],
                         grain_ss=2)
    ok &= check("颗粒超采样 1x/2x 输出不同",
                not np.array_equal(f_ss1, f_ss2))

    # 颗粒曝光轴复核：中灰（E_mid=1 → logH=REF_LOG_E）查表 σ 应处于峰值的一定比例内
    gran = model._get_gran()
    logh = model.REF_LOG_E
    for k in ("R", "G", "B"):
        s_gray = float(np.interp(logh, gran["logE"], gran[f"sigma_{k}"]))
        ratio = s_gray / float(gran[f"sigma_{k}"].max())
        ok &= check(f"颗粒曝光轴 {k}：灰点σ/峰值σ 在 [0.3,1.2]",
                    0.3 <= ratio <= 1.2, detail=f"{ratio:.2f}")
    return ok


def test_old_models():
    print("\n== 5. HP5 旧模型 + Gold200 新结构烟测 ==")
    ok = True
    rng = np.random.default_rng(7)
    lux = rng.random((64, 64, 3), dtype=np.float32) * 0.8
    r, g, b = [lux[:, :, i] for i in range(3)]

    import Films.HP5.HP5 as HP5
    film = HP5.process(r, g, b)
    ok &= check("HP5 输出 (64,64) uint8",
                film.shape == (64, 64) and film.dtype == np.uint8)
    blk = HP5.process(np.zeros((64, 64), np.float32),
                      np.zeros((64, 64), np.float32),
                      np.zeros((64, 64), np.float32))
    wht = HP5.process(np.full((64, 64), E_MID_WHITE, np.float32),
                      np.full((64, 64), E_MID_WHITE, np.float32),
                      np.full((64, 64), E_MID_WHITE, np.float32))
    ok &= check("HP5 0,0,0→黑 饱和→白",
                blk.mean() < 10 and wht.mean() > 240,
                detail=f"黑 {blk.mean():.1f} / 白 {wht.mean():.1f}")
    np.random.seed(0)
    h1 = HP5.process(r, g, b)
    np.random.seed(0)
    h2 = HP5.process(r, g, b, exposure_offset=1.0)
    ok &= check("HP5 曝光偏移改变输出", not np.array_equal(h1, h2))

    import Films.Gold200.Gold200 as Gold200
    np.random.seed(0)
    film = Gold200.process(r, g, b)
    ok &= check("Gold200 输出 (64,64,3) uint8",
                film.shape == (64, 64, 3) and film.dtype == np.uint8)
    np.random.seed(0)
    film_g1 = Gold200.process(r, g, b, contrast=0.6)
    np.random.seed(0)
    film_g2 = Gold200.process(r, g, b, contrast=-0.6)
    ok &= check("Gold200 对比度改变输出",
                not np.array_equal(film_g1, film_g2))
    return ok


def test_provia_model():
    print("\n== 6. Provia 100F 正片模型（无反转验证） ==")
    model = importlib.import_module("Films.Provia100F.Provia100F")
    ok = True

    patches = {
        "白": (E_MID_WHITE, E_MID_WHITE, E_MID_WHITE),
        "中灰": (E_MID_GRAY, E_MID_GRAY, E_MID_GRAY),
        "纯红": (E_MID_WHITE, 0.0, 0.0),
        "纯绿": (0.0, E_MID_WHITE, 0.0),
        "纯蓝": (0.0, 0.0, E_MID_WHITE),
        "黑": (0.0, 0.0, 0.0),
    }
    block_w = 64
    img = np.zeros((96, block_w * len(patches), 3), dtype=np.float32)
    for i, (name, rgb) in enumerate(patches.items()):
        img[:, i * block_w:(i + 1) * block_w] = rgb
    r, g, b = [img[:, :, i] for i in range(3)]

    # 默认调用（光谱输入/染料堆叠/客观扫描/显示编码固定）
    film = model.process(r, g, b)
    ok &= check("Provia 默认输出形状 uint8",
                film.shape == img.shape[:2] + (3,) and film.dtype == np.uint8)
    ok &= check("Provia 默认输出有限", np.isfinite(film).all())

    # 默认调用等价显式参数
    np.random.seed(0)
    film_default = model.process(r, g, b)
    np.random.seed(0)
    film_ref = model.process(r, g, b, grain_ss=2,
                             exposure_offset=0.0, contrast=0.0)
    ok &= check("Provia 默认调用等价于显式参数",
                np.array_equal(film_default, film_ref))

    # 三锚点：0,0,0 → 黑，饱和（E_mid=7.80）→ 白，正像无反转
    blk = model.process(np.zeros((64, 64), np.float32),
                        np.zeros((64, 64), np.float32),
                        np.zeros((64, 64), np.float32))
    wht = model.process(np.full((64, 64), E_MID_WHITE, np.float32),
                        np.full((64, 64), E_MID_WHITE, np.float32),
                        np.full((64, 64), E_MID_WHITE, np.float32))
    ok &= check("Provia 0,0,0→黑 饱和→白（三锚点）",
                blk.mean() < 15 and wht.mean() > 200,
                detail=f"黑 {blk.mean():.1f} / 白 {wht.mean():.1f}")
    np.random.seed(0)

    # 正像方向：纯红/纯蓝输入直接驱动对应通道（无反转）
    np.random.seed(0)
    film_spec = model.process(r, g, b)
    red = film_spec[48, 2 * block_w + block_w // 2].astype(float)
    blue = film_spec[48, 4 * block_w + block_w // 2].astype(float)
    ok &= check("Provia 纯红输出红通道最高",
                red[0] > red[1] and red[0] > red[2], detail=f"{red}")
    ok &= check("Provia 纯蓝输出蓝通道最高",
                blue[2] > blue[1] and blue[2] > blue[0], detail=f"{blue}")

    # 参数响应：曝光偏移 / 显示伽马
    np.random.seed(0)
    a = model.process(r, g, b)
    np.random.seed(0)
    b2 = model.process(r, g, b, exposure_offset=1.0)
    ok &= check("Provia 曝光偏移改变输出", not np.array_equal(a, b2))
    np.random.seed(0)
    c1 = model.process(r, g, b, contrast=0.5)
    np.random.seed(0)
    c2 = model.process(r, g, b, contrast=-0.5)
    ok &= check("Provia 对比度改变输出", not np.array_equal(c1, c2))

    # 特性曲线负斜率（正像：密度随曝光下降）
    logh = model._to_log_exposure(
        np.linspace(0.02, 1.0, 128, dtype=np.float32), 0.0)
    dr, dg, db = model._char_density(logh, logh, logh)
    ok &= check("Provia 特性曲线负斜率（密度随曝光下降）",
                bool(np.all(np.diff(dr) <= 0) and np.all(np.diff(dg) <= 0)
                     and np.all(np.diff(db) <= 0)),
                detail=f"dR {dr[0]:.2f}→{dr[-1]:.2f}")

    # MTF 拟合残差
    mtf = model._data.get_mtf()
    fit = model._get_psf_fit()
    twopi2 = 2.0 * np.pi ** 2
    f, y = mtf["freq_raw"], mtf["resp_raw"] / 100.0
    s_c, dl, s_a, w_c, w_d, w_a = fit
    pred = (w_c * np.exp(-twopi2 * s_c ** 2 * f ** 2)
            + w_d / (1.0 + (2.0 * np.pi * dl * f) ** 2)
            - w_a * np.exp(-twopi2 * s_a ** 2 * f ** 2))
    maxerr = float(np.abs(pred - y).max())
    ok &= check("Provia MTF 拟合误差 < 6%", maxerr < 0.06,
                detail=f"{maxerr * 100:.2f}%")
    return ok


def test_spectral_reconstruct():
    print("\n== 7. 光谱重建迁移（唯一路径 + 色卡验证） ==")
    from Spectral import reconstruct as rc
    ok = True

    # 合成光谱生成器物理性：平滑、非负、反射率 ≤1
    rng = np.random.default_rng(0)
    spec = rc._synth_spectrum(rng)
    ok &= check("合成光谱形状 (61,)",
                spec.shape == (61,))
    ok &= check("合成光谱非负且 ≤1（物理反射率）",
                bool(np.all(spec >= 0.0) and np.all(spec <= 1.0)))
    # 平滑上限 0.20/5nm：真实光谱峰附近斜率可较陡（色卡 red 红边约
    # 0.07/5nm，双峰叠加的极端样本可达 ~0.19），但不会出现振荡
    ok &= check("合成光谱平滑（相邻波长差 < 0.20）",
                float(np.abs(np.diff(spec)).max()) < 0.20,
                detail=f"最大步进 {np.abs(np.diff(spec)).max():.3f}")

    # 映射矩阵与白场精确归一
    for film in ["gold200", "5213", "provia100f", "hp5"]:
        mat = rc._get_map(film)
        if film == "hp5":
            ok &= check(f"{film} 映射形状 (4,1)", mat.shape == (4, 1))
            white = float((np.array([1, 1, 1, 1.0]) @ mat).item())
            ok &= check(f"{film} 白场曝光=1", abs(white - 1.0) < 1e-4,
                        detail=f"{white:.5f}")
        else:
            ok &= check(f"{film} 映射形状 (4,3)", mat.shape == (4, 3))
            white = np.array([1, 1, 1, 1.0]) @ mat
            ok &= check(f"{film} 白场三层曝光=1",
                        bool(np.all(np.abs(white - 1.0) < 1e-4)),
                        detail=f"{np.round(white, 5)}")

    # 色卡 24 色独立验证（真实光谱不参与训练）
    for film in ["gold200", "5213", "provia100f", "hp5"]:
        mae, mx = rc.verify_on_colorchecker(film)
        ok &= check(f"{film} 色卡验证曝光 MAE < 0.04",
                    mae < 0.04, detail=f"MAE={mae:.4f} max={mx:.4f}")

    # 纯红曝光合理性：红层主导，绿层串扰小（5213）
    mat = rc._get_map("5213")
    red_exp = np.array([1, 0, 0, 1.0]) @ mat  # B/G/R
    ok &= check("5213 纯红：红层曝光 > 0.8",
                red_exp[2] > 0.8, detail=f"R={red_exp[2]:.3f}")
    ok &= check("5213 纯红：绿层串扰 < 0.08（重建先验修正）",
                red_exp[1] < 0.08, detail=f"G={red_exp[1]:.3f}")

    # 偏置项清零：暗部（RGB→0）曝光必须为 0，避免"红雾"
    for film in ["gold200", "5213", "provia100f", "hp5"]:
        mat = rc._get_map(film)
        bias = mat[3]
        ok &= check(f"{film} 映射偏置项已清零（暗部=黑，无红雾）",
                    bool(np.all(np.abs(bias) < 1e-5)),
                    detail=f"偏置={np.round(bias, 6)}")

    # 无颗粒暗部灰阶：重建映射 RGB=0 → 曝光=0 → 输出黑；
    # 灰阶暗部（0.03-0.05）不偏红。偏置项清零的意义就是"无光=黑"——
    # 若暗部被注入 R 层曝光（红雾），成片 R 会显著高于 G/B，
    # 这里在成片层面验证无此现象（Gold200 中性暗部自然偏 G/B 属物理特性）。
    model = importlib.import_module("Films.Gold200.Gold200")
    from Scanner import render
    mr = rc._get_map("gold200")

    def run_plain(e_bgr):
        e_r = np.full((8, 8), e_bgr[2], np.float32)
        e_g = np.full((8, 8), e_bgr[1], np.float32)
        e_b = np.full((8, 8), e_bgr[0], np.float32)
        logh = model._to_log_exposure(e_r, 0), model._to_log_exposure(e_g, 0), \
            model._to_log_exposure(e_b, 0)
        d = model._char_density(*logh)
        return render(model._dye_stack_scan(*d), "gold200", contrast=0.0)[0, 0]

    dark_ok = True
    for level in [0.0, 0.01, 0.03, 0.05]:
        rgb = np.array([level] * 3, np.float64)
        o = run_plain(np.array([*rgb, 1.0]) @ mr)
        dark_ok &= bool(float(o[0]) <= float(o[1]) + 2.0)  # R 不显著高于 G
        if level < 0.02:
            dark_ok &= bool(float(o.max()) <= 2.0)          # 近黑区纯黑
    ok &= check("暗部不偏红、近黑区纯黑（无红雾、无颗粒）", dark_ok)

    # 纯红成片方向：唯一路径（光谱重建）下红通道最高、绿通道显著低于红
    model = importlib.import_module("Films.5213.5213")
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[:, :] = [1.0, 0.0, 0.0]
    np.random.seed(0)
    out_rec = model.process(img[..., 0], img[..., 1], img[..., 2], grain_ss=1)
    r_px = float(out_rec[32, 32, 0])
    g_px = float(out_rec[32, 32, 1])
    ok &= check("光谱重建纯红成片红通道最高", r_px > g_px,
                detail=f"R={r_px:.0f} G={g_px:.0f}")
    ok &= check("光谱重建纯红成片绿通道 < 红通道70%（偏橙修正）",
                g_px < 0.7 * r_px,
                detail=f"G/R={g_px / max(r_px, 1e-6):.2f}")
    return ok


def test_three_stage():
    """0.3.3 架构验证：三段式（Input → process_scan → Scanner.render）
    与便捷入口 process() 输出逐位一致；Input 解码烟测。"""
    print("\n== 8. 三段式一致性（0.3.3 架构） ==")
    ok = True

    # 8.1 第二段 + 第三段 == process() 便捷入口（逐位，含颗粒随机路径）
    from Scanner import render
    cases = [
        ("Films.HP5.HP5", "hp5"),
        ("Films.Gold200.Gold200", "gold200"),
        ("Films.5213.5213", "5213"),
        ("Films.Provia100F.Provia100F", "provia100f"),
    ]
    rng = np.random.default_rng(11)
    lux = rng.random((48, 64, 3), dtype=np.float32) * 0.8
    r, g, b = [lux[..., i] for i in range(3)]
    for modname, key in cases:
        model = importlib.import_module(modname)
        # 同一随机种子下，process() 与 process_scan()+render() 应逐位一致
        np.random.seed(5)
        out_direct = model.process(r, g, b)
        np.random.seed(5)
        scan = model.process_scan(r, g, b)
        out_stage = render(scan, key)
        ok &= check(f"{key} 三段式输出 == process() 便捷入口",
                    np.array_equal(out_direct, out_stage))
        # 自定义对比度也应一致
        if key != "hp5":
            np.random.seed(6)
            out_direct_g = model.process(r, g, b, contrast=0.5)
            np.random.seed(6)
            scan_g = model.process_scan(r, g, b)
            out_stage_g = render(scan_g, key, contrast=0.5)
            ok &= check(f"{key} 自定义对比度三段式一致",
                        np.array_equal(out_direct_g, out_stage_g))

    # 8.2 第一段 Input：JPEG → 线性光照度烟测
    img_bgr = np.full((128, 128, 3), 128, np.uint8)
    ok_enc, buf = cv2.imencode(".jpg", img_bgr)
    ok &= check("Input 测试图 JPEG 编码成功", ok_enc)
    if ok_enc:
        from Input import decode_to_lux
        lux_r, lux_g, lux_b = decode_to_lux(buf.tobytes(), "test.jpg")
        # 原片解析的 standardize 会把短边统一缩放到 3024（与 DNG 路径一致）
        ok &= check("Input 解码 JPEG → 光照度形状（短边=3024）",
                    lux_r.shape == (3024, 3024) and np.isfinite(lux_r).all(),
                    detail=f"shape={lux_r.shape}")
        spread = float(np.abs(lux_r - lux_b).max())
        ok &= check("Input 解码灰场三通道一致", spread < 1e-6,
                    detail=f"最大通道差 {spread:.2e}")
    return ok


def test_anchor_chain():
    """9. 三锚点链（0.3.3）：输入锚点 → 第二段密度 → 扫描锚点全程一致。

    约定：原片黑点/白点/中灰点、0-1 输入的黑/灰/白、扫描锚定的黑/灰/白
    在三个过程中始终统一：
      输入锚点：JPEG 0→E_mid 0、sRGB118→E_mid≈1、255→E_mid≈5.56；
               DNG 黑端→0、ISO 12232 灰点→1、饱和→7.80·2^BE。
      扫描锚点：E_mid=0→输出 0、E_mid=1→输出 118、白锚→输出 255，
               且第二段实测密度与 FILM_PROFILES 锚点密度一致（防漂移）。
    """
    print("\n== 9. 三锚点链 ==")
    ok = True

    # 9.1 JPEG 数转胶输入锚点：0→0、118→中灰≈1、255→白场≈5.56
    import cv2
    from Input import decode_to_lux
    for val, expect in [(0, 0.0), (118, 1.0), (255, 1.0 / 0.18)]:
        img = np.full((8, 8, 3), val, np.uint8)
        ok_enc, buf = cv2.imencode(".jpg", img)
        lr, lg, lb = decode_to_lux(buf.tobytes(), "t.jpg")
        got = float(lr[4, 4])
        tol = 0.02 if val else 1e-6
        ok &= check(f"JPEG 输入锚点 {val} → E_mid ≈{expect:.3f}",
                    abs(got - expect) < tol, detail=f"{got:.4f}")

    # 9.2/9.3 每卷：黑/灰/白锚输入 → 输出 0/118/255；实测密度 vs 锚点密度
    from Scanner import FILM_PROFILES, render
    films = [
        ("hp5", "Films.HP5.HP5", "hp5"),
        ("gold200", "Films.Gold200.Gold200", "gold200"),
        ("5213", "Films.5213.5213", "5213"),
        ("provia100f", "Films.Provia100F.Provia100F", "provia100f"),
    ]
    for fname, fmod, key in films:
        mod = importlib.import_module(fmod)
        prof = FILM_PROFILES[key]
        for label, e in [("黑", 0.0), ("灰", 1.0), ("白", 6.24)]:
            a = np.full((64, 64), e, np.float32)
            scan = mod.process_scan(a, a, a, grain_ss=1, exposure_offset=0.0)
            out = render(scan, key, contrast=0.0).astype(np.float32)
            med = np.median(out, axis=(0, 1)) if out.ndim == 3 else np.median(out)
            expect_out = {"黑": 0.0, "灰": 118.0, "白": 255.0}[label]
            ok &= check(f"{key} {label}锚输入 → 输出{expect_out:.0f}",
                        float(np.max(np.abs(med - expect_out))) < 2.0,
                        detail=f"{np.round(med, 1)}")
            # 第二段实测密度必须等于扫描锚点密度（锚点链不漂移）
            anchor_key = {"黑": "black", "灰": "gray", "白": "white"}[label]
            if key == "hp5":
                d = float(np.median(scan))
            else:
                d = -np.log10(np.clip(np.median(scan, axis=(0, 1)), 1e-6, 1.0))
            err = float(np.max(np.abs(d - prof[anchor_key])))
            ok &= check(f"{key} {label}密度 ≈ 扫描锚点",
                        err < 0.01, detail=f"最大差 {err:.4f}")
    return ok


def test_input_contract():
    """0.3.3 输入契约与三锚点：E_mid 中灰归一化、显示编码、三锚点。"""
    print("\n== 10. 输入契约与三锚点 ==")
    ok = True

    # 10.1 JPEG 数转胶契约：255 → E_mid≈5.56（白场），118 → ≈1.0（中灰）
    import cv2
    from Input import decode_to_lux
    for val, expect in [(255, 1.0 / 0.18), (118, 1.0)]:
        img = np.full((8, 8, 3), val, np.uint8)
        ok_enc, buf = cv2.imencode(".jpg", img)
        lux_r, lux_g, lux_b = decode_to_lux(buf.tobytes(), "t.jpg")
        got = float(lux_r[4, 4])
        ok &= check(f"JPEG 像素 {val} → 线性 ≈{expect}",
                    abs(got - expect) < 0.02, detail=f"{got:.4f}")

    # 10.2 DNG 中灰常量（ISO 12232：H_gray/H_sat = 10/78 = 18%/√2）
    from Input import raw as raw_mod
    ok &= check("ISO 灰点分数 ≈0.1282",
                abs(raw_mod._GRAY_FRAC - 10.0 / 78.0) < 1e-9,
                detail=f"{raw_mod._GRAY_FRAC:.5f}")

    # 10.3 显示编码：pos=1 → 255，pos>1 不超白、不低于白（单调收口）
    from Scanner.scanner import FILM_PROFILES, _encode_display
    pos = np.array([0.0, 0.5, 0.9, 1.0, 1.2, 2.0], np.float32)
    out = _encode_display(pos, 0.6)
    ok &= check("pos=0 → 输出黑（黑端硬切）", int(out[0]) == 0)
    ok &= check("pos=1 → 输出白 255", int(out[3]) == 255)
    ok &= check("过曝输出不超白", bool((out[4:] <= 255).all()))
    ok &= check("过曝输出不低于场景白（单调）", bool((out[4:] >= out[3]).all()))

    # 10.4 三锚点核心：黑→0、18% 灰→118、饱和→255（gamma=1.0）
    from Scanner.scanner import _three_anchor_curve
    # 用 Gold200 的密度锚点构造典型曲线（黑<灰<白）
    black = np.array([0.23908, 0.62761, 0.96540])
    gray = np.array([0.53693, 1.15090, 1.55571])
    white = np.array([0.78872, 1.55689, 2.13059])
    d_test = np.array([black, gray, white])  # (3 通道, 3 锚点)
    pos3 = np.zeros((3, 3))
    for c in range(3):
        pos3[c] = _three_anchor_curve(
            d_test[:, c], float(black[c]), float(gray[c]), float(white[c]))
    ok &= check("三锚点：黑密度→0", bool(np.all(np.abs(pos3[:, 0]) < 1e-6)))
    ok &= check("三锚点：中灰密度→118/255",
                bool(np.all(np.abs(pos3[:, 1] - 118.0/255.0) < 1e-6)))
    ok &= check("三锚点：白密度→1",
                bool(np.all(np.abs(pos3[:, 2] - 1.0) < 1e-6)))

    # 10.5 PCHIP 边界截断：超出锚点范围的密度取端点值（防三次外推异常）
    pos_edge = _three_anchor_curve(
        np.array([0.0, 0.01, 5.0, 10.0]),
        float(black[0]), float(gray[0]), float(white[0]))
    ok &= check("PCHIP 边界截断：超暗→0、超亮→1",
                pos_edge[0] == 0.0 and pos_edge[1] == 0.0
                and pos_edge[2] == 1.0 and pos_edge[3] == 1.0,
                detail=f"{np.round(pos_edge, 3)}")
    # 正片（black > white）：密度超出黑端（更暗）→ 0，超出白端（更亮）→ 1
    pos_edge_p = _three_anchor_curve(
        np.array([6.0, 0.01]),
        float(FILM_PROFILES["provia100f"]["black"][0]),
        float(FILM_PROFILES["provia100f"]["gray"][0]),
        float(FILM_PROFILES["provia100f"]["white"][0]))
    ok &= check("正片边界截断：超黑端→0、超白端→1",
                pos_edge_p[0] == 0.0 and pos_edge_p[1] == 1.0,
                detail=f"{np.round(pos_edge_p, 3)}")
    return ok


def test_optical():
    """11. 镜头光学修正（DNG OpcodeList3）：GainMap / FixVignetteRadial。"""
    print("\n== 11. 镜头光学修正 ==")
    ok = True
    from Input import optical as opt
    import struct

    def build_opcode_list(oid, params):
        head = struct.pack(">IIII", oid, 0x01030000, 0, len(params))
        return struct.pack(">I", 1) + head + params

    # 11.1 opcode 列表解析（大端格式）
    blob = build_opcode_list(9, b"\x00" * 64)
    parsed = opt.parse_opcode_list(blob)
    ok &= check("opcode 列表解析：单条 GainMap",
                len(parsed) == 1 and parsed[0][0] == 9,
                detail=f"{[(p[0], p[3]) for p in parsed]}")

    # 11.2 GainMap：均匀增益 2.0 → 全图×2；且只影响声明平面
    def gm_params(h, w, gain, plane=0, planes=3, mapv=2, maph=2, mplanes=3):
        p = struct.pack(">10I", 0, 0, h, w, plane, planes, 1, 1,
                        mapv, maph)
        sp = 1.0 / (mapv - 1) if mapv > 1 else 1.0
        p += struct.pack(">4d", sp, sp, 0.0, 0.0)
        p += struct.pack(">I", mplanes)
        p += struct.pack(">%df" % (mapv * maph * mplanes),
                         *([gain] * (mapv * maph * mplanes)))
        return p

    img = np.ones((8, 8, 3), np.float32)
    blob = build_opcode_list(9, gm_params(8, 8, 2.0))
    out = opt.apply_opcode_list3(img, blob)
    ok &= check("GainMap 均匀增益 ×2（三平面）",
                np.allclose(out, 2.0), detail=f"{out[0, 0]} 期望 2.0")

    # 只影响 plane=1（绿）通道
    blob = build_opcode_list(9, gm_params(8, 8, 2.0, plane=1, planes=1,
                                          mapv=2, maph=2, mplanes=1))
    out = opt.apply_opcode_list3(img, blob)
    ok &= check("GainMap 平面选择性（仅 G 通道 ×2）",
                np.allclose(out[..., 0], 1.0) and np.allclose(out[..., 1], 2.0)
                and np.allclose(out[..., 2], 1.0),
                detail=f"{out[0, 0]}")

    # 11.3 FixVignetteRadial：k=0 恒等；k0>0 时四角 > 中心
    k_ident = struct.pack(">7d", 0, 0, 0, 0, 0, 0.5, 0.5)
    out = opt.apply_opcode_list3(img, build_opcode_list(3, k_ident))
    ok &= check("FixVignetteRadial k=0 恒等", np.allclose(out, img))
    k_rad = struct.pack(">7d", 0.5, 0, 0, 0, 0, 0.5, 0.5)
    out = opt.apply_opcode_list3(img, build_opcode_list(3, k_rad))
    corner = float(out[0, 0, 0]); center = float(out[4, 4, 0])
    ok &= check("FixVignetteRadial 四角增益 > 中心",
                corner > center and corner > 1.0,
                detail=f"角 {corner:.3f} 中心 {center:.3f}")

    # 11.4 无 opcode / 空列表 → 原样返回
    ok &= check("空列表恒等", np.allclose(opt.apply_opcode_list3(img, b""), img))
    return ok


def main():
    print("Phos 0.3.3 自检开始")
    results = [
        test_cie_data(),
        test_5213_model(),
        test_mtf_psf(),
        test_grain(),
        test_old_models(),
        test_provia_model(),
        test_spectral_reconstruct(),
        test_three_stage(),
        test_anchor_chain(),
        test_input_contract(),
        test_optical(),
    ]
    print("\n==========================================")
    if all(results):
        print("全部检查通过 ✔ 可以正常使用")
        return 0
    print("存在未通过的检查 ✘ 请查看上方 FAIL 项")
    return 1


if __name__ == "__main__":
    sys.exit(main())
