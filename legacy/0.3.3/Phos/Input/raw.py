#赛博请神
# Input/raw.py —— 第一段（原片解析）：DNG/RAW → 线性光照度
# 输入契约（0.3.3 科学读图规范）：
#   线性值 E = 相对饱和曝光（bright=1.0，白电平归一化），
#   1.0 = 传感器饱和 / 白场，允许 >1（过曝余量）。
#   中灰按 ISO 12232 锚定：H_gray/H_sat = 10/78 ≈ 0.1282
#   （即 18%/√2，SOS 与饱和速度之比；100% 反射白 ≈ 70.8% 饱和，
#   留 1/2 档高光余量，与 JPEG 白场 E_mid≈5.56 一致），
#   再乘 2^BaselineExposure 补偿相机自身的中灰偏移（DNG 规范 0xC62A）：
#   相机定义的"正确曝光"（ACR/预览 Exposure=0）恒对应 E_mid = 1.0。
#   输出统一为相对中灰 E_mid = E/0.1282·2^BE（中灰 = 1.0，
#   饱和 E_mid ≈ 7.80·2^BE，随相机 BE 变化），
#   第二段 stop = log2(E_mid)，中灰恒在 stop 0。
import cv2
import numpy as np
import rawpy
import struct

# ---------- 输入契约常量 ----------
# ISO 12232：饱和曝光 H_sat = 78/ISO；标准输出灵敏度（SOS）灰点
# H_gray = 10/ISO（"10" 内含 1/2 档高光余量，即 18%/√2）。
# 中灰相对饱和 = (10/ISO)/(78/ISO) = 10/78 ≈ 0.1282（+2.97 档），
# 与相机 ISO 无关（ISO 在分子分母抵消）。
# 注意：测光常数 K=12.5 不是焦点平面曝光比例；12.5/78=0.1603 是
# 把 K 误当成 H_gray 得到的错误比值（对应 +2.64 档，偏暗 0.32 EV）。
_GRAY_FRAC = 10.0 / 78.0            # ≈ 0.1282（18%/√2）


# ---------- DNG 标签读取（纯 Python，零新依赖） ----------
# DNG 规范第 4 章：BaselineExposure 0xC62A（SRATIONAL，EV），
# BaselineExposureOffset 0xC7A5（RATIONAL，EV，随相机配置文件叠加）。
# 渲染时实际默认曝光 = BaselineExposure + BaselineExposureOffset。
_TAG_BASELINE_EXPOSURE = 0xC62A
_TAG_BASELINE_EXPOSURE_OFFSET = 0xC7A5
_TAG_OPCODE_LIST3 = 0xC74E
_TYPE_RATIONAL = 5
_TYPE_SRATIONAL = 10
_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
               8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _read_dng_ifd0(stream):
    """解析 DNG IFD0 的全部标签，返回 (endian, {tag: (type, count, bytes)})。

    IFD0 承载 DNG 规范要求的渲染元数据（BaselineExposure、OpcodeList 等）；
    读取后恢复流位置，便于后续 rawpy 从头解码。任何解析异常都回退空表。
    """
    saved = 0
    try:
        saved = stream.tell()
        stream.seek(0)
        header = stream.read(8)
        if len(header) < 8:
            return None, {}
        endian = "<" if header[:2] == b"II" else ">" if header[:2] == b"MM" else None
        if endian is None:
            return None, {}
        ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
        stream.seek(ifd_offset)
        count = struct.unpack(endian + "H", stream.read(2))[0]
        entries = stream.read(12 * count)
        if len(entries) < 12 * count:
            return endian, {}
        tags = {}
        for i in range(count):
            tag, typ, cnt = struct.unpack(
                endian + "HHI", entries[12 * i:12 * i + 8])
            nbytes = _TYPE_SIZES.get(typ, 1) * cnt
            value_field = entries[12 * i + 8:12 * i + 12]
            if nbytes <= 4:
                data = value_field[:nbytes]
            else:
                offset = struct.unpack(endian + "I", value_field)[0]
                stream.seek(offset)
                data = stream.read(nbytes)
            tags[tag] = (typ, cnt, data)
        return endian, tags
    except Exception:
        return None, {}
    finally:
        try:
            stream.seek(saved)
        except Exception:
            pass


def _baseline_ev_from_tags(endian, tags):
    """从已解析的 IFD0 标签计算 BaselineExposure（+Offset）总 EV；缺失 0.0。

    只处理 IFD0（DNG 规范要求本标签位于 IFD 0；相机配置文件的
    BaselineExposureOffset 通常不内嵌在文件里，缺失即 0）。
    """
    total_ev = 0.0
    for tag in (_TAG_BASELINE_EXPOSURE, _TAG_BASELINE_EXPOSURE_OFFSET):
        item = tags.get(tag)
        if item is None:
            continue
        typ, cnt, data = item
        if cnt != 1 or len(data) < 8:
            continue
        if typ == _TYPE_RATIONAL:
            num, den = struct.unpack(endian + "II", data[:8])
        elif typ == _TYPE_SRATIONAL:
            num, den = struct.unpack(endian + "ii", data[:8])
        else:
            continue
        if den:
            total_ev += num / den
    return total_ev


def standardize(image):
    """标准化图像尺寸"""
    min_size = 3024
    height, width = image.shape[:2]
    if height < width:
        scale_factor = min_size / height
        new_height = min_size
        new_width = int(width * scale_factor)
    else:
        scale_factor = min_size / width
        new_width = min_size
        new_height = int(height * scale_factor)

    new_width = new_width + 1 if new_width % 2 != 0 else new_width
    new_height = new_height + 1 if new_height % 2 != 0 else new_height
    interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (new_width, new_height),
                       interpolation=interpolation)
    return image


def split_linear(bgr_image):
    """从 BGR 图像分离通道，返回线性 RGB 浮点值 (r, g, b)"""
    b, g, r = cv2.split(bgr_image)
    return r, g, b


def raw_to_lux(uploaded_image, optical=True):
    """将上传的 DNG/RAW 图像转换为相对中灰 E_mid（按输入契约）。

    管线：rawpy 线性解码（bright=1.0 → 相对饱和）→ 中灰归一化
    （÷0.1282，ISO 12232 灰点 → E_mid=1）→ 应用相机 BaselineExposure
    （×2^BE，DNG 规范 0xC62A）→ 输出 float32，允许 E_mid>1（过曝）。
    optical=True 时应用 OpcodeList3 镜头光学修正（GainMap/FixVignetteRadial）。
    白平衡：use_camera_wb（DNG 有 AsShotNeutral 标注用标注；无标注时
    保持原始通道比，不做 AWB，避免把片基偏色当成场景色温修正）。
    """
    # DNG 规范：BaselineExposure 描述相机"正确曝光"相对 ISO 12232
    # 理论中灰的偏移（EV），正值 = 默认渲染更亮。ACR/预览等渲染器
    # 会在 Exposure=0 时应用它，因此我们必须在进入胶片曲线前同样
    # 折算进 E_mid，否则相机的曝光定义与 ISO 12232 理论值不一致时
    # 全部画面都会系统性偏暗/偏亮（各相机由文件内 BE 决定）。
    # 从上传字节流直接解析 IFD0（Streamlit 上传对象只有文件名没有真实路径，
    # 不能依赖文件系统）；解析失败保守回退（无 BE / 无 opcode 的 DNG 不变）。
    endian, dng_tags = _read_dng_ifd0(uploaded_image)
    baseline_ev = _baseline_ev_from_tags(endian, dng_tags)
    opcode3 = dng_tags.get(_TAG_OPCODE_LIST3)

    try:
        with rawpy.imread(uploaded_image) as raw:
            # postprocess to get linear RGB
            # gamma=(1,1) -> Linear (no gamma correction)
            # no_auto_bright=True -> Do not automatically adjust brightness
            # output_bps=16 -> 16-bit depth for higher dynamic range
            # bright=1.0：相对饱和（白电平归一化），不做显示亮度增益
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=True,
                bright=1.0,
                output_bps=16,
            )
    except Exception as e:
        raise RuntimeError(f"读取 RAW 图像失败: {str(e)}")

    # 16 位整数 → 相对饱和浮点（线性参考值，允许 >1）
    rgb_float = rgb.astype(np.float32) / 65535.0
    # OpcodeList3 镜头光学修正（去马赛克后的线性 RGB 上执行，规范第 7 章）
    if optical and opcode3 is not None:
        from Input.optical import apply_opcode_list3
        rgb_float = apply_opcode_list3(rgb_float, opcode3[2])
    # 中灰归一化（中灰 E_mid=1，允许 >1）
    # 再 ×2^BaselineExposure：把相机的曝光定义对齐到 ISO 12232 中灰锚点
    rgb_float = rgb_float / _GRAY_FRAC
    if baseline_ev:
        rgb_float *= float(2.0 ** baseline_ev)

    # 转换为 BGR 顺序，以复用 standardize 函数（它期望 BGR）
    bgr = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2BGR)

    # 标准化尺寸
    bgr_std = standardize(bgr)

    # 分离通道得到线性 lux 值
    lux_r, lux_g, lux_b = split_linear(bgr_std)
    # 物理约束：线性光照度无负值。重采样振铃（LANCZOS）会在黑位附近
    # 产生负振荡，这里按黑点钳制到 0（负值只有数学意义，胶片无法响应）。
    lux_r = np.maximum(lux_r, 0.0)
    lux_g = np.maximum(lux_g, 0.0)
    lux_b = np.maximum(lux_b, 0.0)
    return lux_r, lux_g, lux_b
