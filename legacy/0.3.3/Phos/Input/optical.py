#赛博请神
# Input/optical.py —— 第一段（原片解析）：DNG OpcodeList 镜头光学修正
# DNG 规范 1.7.0.0 第 7 章：三个 opcode 列表分别在三处执行——
#   OpcodeList1（0xC740）：原始数据刚读出（未线性化）；
#   OpcodeList2（0xC741）：刚映射到线性参考值（未去马赛克）；
#   OpcodeList3（0xC74E）：刚去马赛克之后（线性 RGB）。
# 暗角/畸变相关的 opcode：
#   FixVignetteRadial（ID 3）：径向多项式增益 g = 1+Σk_i·r^(2i+2)；
#   GainMap（ID 9）：子采样 2D 浮点增益表（双线性插值 + 边缘复制）；
#   WarpRectilinear / WarpFisheye / WarpRectilinear2（ID 1/2/14）：畸变与
#   横向色差（本模块暂不实现，当前样张均无；遇到时跳过并告警）。
# rawpy/LibRaw 不执行任何 opcode，因此第一段必须在线性解码后自行应用
# OpcodeList3（部分相机的 GainMap 暗角修正就在这里）。
#
# 约定：与 DNG 规范不同，我们不把 opcode 结果 clip 到 [0,1]——
# 线性参考值允许 >1（过曝余量），胶片肩部需要它；乘法修正不受影响。
import struct

import cv2
import numpy as np

# ---------- opcode 常量 ----------
_OP_FIX_VIGNETTE_RADIAL = 3
_OP_GAIN_MAP = 9
_OP_WARP = {1, 2, 14}          # WarpRectilinear / WarpFisheye / WarpRectilinear2
_FLAG_OPTIONAL = 0x01


def parse_opcode_list(blob):
    """按 DNG 规范解析 opcode 列表（列表永远大端存储）。

    返回 [(opcode_id, dng_version, flags, params_bytes), ...]。
    """
    if not blob or len(blob) < 4:
        return []
    count = struct.unpack(">I", blob[:4])[0]
    out = []
    off = 4
    for _ in range(count):
        if off + 16 > len(blob):
            break
        oid, version, flags, plen = struct.unpack(">IIII", blob[off:off + 16])
        off += 16
        params = blob[off:off + plen]
        off += plen
        out.append((oid, version, flags, params))
    return out


def parse_gain_map(params):
    """解析 GainMap（ID 9）参数，返回字段字典 + gains(maph, mapv, mplanes)。"""
    o = 0

    def u32():
        nonlocal o
        v = struct.unpack(">I", params[o:o + 4])[0]
        o += 4
        return v

    def f64():
        nonlocal o
        v = struct.unpack(">d", params[o:o + 8])[0]
        o += 8
        return v

    def f32():
        nonlocal o
        v = struct.unpack(">f", params[o:o + 4])[0]
        o += 4
        return v

    top, left, bottom, right = u32(), u32(), u32(), u32()
    plane, planes = u32(), u32()
    rowpitch, colpitch = u32(), u32()
    mapv, maph = u32(), u32()
    spv, sph = f64(), f64()
    orv, orh = f64(), f64()
    mplanes = u32()
    gains = np.array(
        [f32() for _ in range(mapv * maph * mplanes)],
        dtype=np.float32).reshape(mapv, maph, mplanes)
    return dict(top=top, left=left, bottom=bottom, right=right,
                plane=plane, planes=planes,
                rowpitch=rowpitch, colpitch=colpitch,
                mapv=mapv, maph=maph, spv=spv, sph=sph,
                orv=orv, orh=orh, mplanes=mplanes, gains=gains)


def parse_fix_vignette_radial(params):
    """解析 FixVignetteRadial（ID 3）：5 个径向系数 + 归一化光心 (ĉx, ĉy)。"""
    k = struct.unpack(">5d", params[:40])
    cx, cy = struct.unpack(">2d", params[40:56])
    return np.asarray(k, dtype=np.float64), float(cx), float(cy)


def _full_gain_map(rgb, gm):
    """把子采样增益表插值到全分辨率（双线性 + 边缘复制，规范第 7 章）。

    采样坐标：像素相对位置 (rel − origin)/spacing，其中 rel 以被修改图像
    的宽高为 1.0；网格外按边缘复制。返回 (h, w, 通道数) 增益：
    仅受影响的平面取对应增益（planes > mplanes 时复用最后一层），
    其余平面为 1.0（保持不变）。
    """
    h, w = rgb.shape[:2]
    height = gm["bottom"] - gm["top"]
    width = gm["right"] - gm["left"]
    # 网格坐标（v=行, h=列）
    rel_v = (np.arange(h, dtype=np.float32) - gm["top"]) / height
    rel_h = (np.arange(w, dtype=np.float32) - gm["left"]) / width
    map_v = (rel_v - gm["orv"]) / gm["spv"]     # 行方向网格坐标
    map_h = (rel_h - gm["orh"]) / gm["sph"]     # 列方向网格坐标
    map_x = np.repeat(map_h[None, :], h, axis=0)   # (h, w) 源列坐标
    map_y = np.repeat(map_v[:, None], w, axis=1)   # (h, w) 源行坐标
    gains = gm["gains"]                            # (mapv, maph, mplanes)
    full = cv2.remap(gains, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)   # (h, w, mplanes)
    nch = rgb.shape[2]
    out = np.ones((h, w, nch), dtype=np.float32)
    start = gm["plane"]
    end = min(start + gm["planes"], nch)
    for c in range(start, end):
        src = min(c - start, gm["mplanes"] - 1)
        out[..., c] = full[..., src]
    return out


def _apply_gain_map(rgb, gm):
    """GainMap：只修正 bounds 内且 rowpitch/colpitch 命中的像素。"""
    full = _full_gain_map(rgb, gm)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    y0, y1 = gm["top"], gm["bottom"]
    x0, x1 = gm["left"], gm["right"]
    if y1 > y0 and x1 > x0:
        rows = np.arange(max(y0, 0), min(y1, h))
        cols = np.arange(max(x0, 0), min(x1, w))
        rows = rows[(rows - gm["top"]) % gm["rowpitch"] == 0]
        cols = cols[(cols - gm["left"]) % gm["colpitch"] == 0]
        mask[np.ix_(rows, cols)] = True
    gain = np.where(mask[..., None], full, 1.0)
    return rgb.astype(np.float32, copy=True) * gain


def _apply_fix_vignette_radial(rgb, k, cx, cy):
    """FixVignetteRadial：g = 1 + k0·r² + k1·r⁴ + … + k4·r¹⁰，逐像素乘。"""
    h, w = rgb.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cxp = 0.0 + cx * (w - 1.0)
    cyp = 0.0 + cy * (h - 1.0)
    mx = max(abs(0.0 - cxp), abs(w - 1.0 - cxp))
    my = max(abs(0.0 - cyp), abs(h - 1.0 - cyp))
    m = float(np.hypot(mx, my))
    r2 = ((x - cxp) ** 2 + (y - cyp) ** 2) / (m * m)
    g = 1.0 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3 \
        + k[3] * r2 ** 4 + k[4] * r2 ** 5
    return rgb.astype(np.float32, copy=True) * g[..., None].astype(np.float32)


def apply_opcode_list3(rgb, blob):
    """应用 OpcodeList3（去马赛克后的线性 RGB 上执行）。

    当前实现暗角修正（GainMap / FixVignetteRadial）；畸变类 opcode
    （WarpRectilinear 等）暂未实现——遇到时跳过（optional 直接跳，
    非 optional 也跳过并保留原图，避免错误修正），待后续补全。
    """
    if blob is None or len(blob) < 4:
        return rgb
    out = np.asarray(rgb, dtype=np.float32)
    for oid, version, flags, params in parse_opcode_list(blob):
        if oid == _OP_GAIN_MAP:
            gm = parse_gain_map(params)
            out = _apply_gain_map(out, gm)
        elif oid == _OP_FIX_VIGNETTE_RADIAL:
            k, cx, cy = parse_fix_vignette_radial(params)
            if np.any(k):
                out = _apply_fix_vignette_radial(out, k, cx, cy)
        elif oid in _OP_WARP:
            # 畸变修正尚未实现：跳过（当前样张均无 warp opcode）
            continue
        # 其余未知/未实现 opcode：optional 跳过；非 optional 也保守跳过
    return out
