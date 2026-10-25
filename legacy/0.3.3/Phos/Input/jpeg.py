#赛博请神
# Input/jpeg.py —— 第一段（原片解析）：JPEG/PNG → 线性光照度
# 输入契约（0.3.3 科学读图规范）：
#   JPEG 像素 /255 → sRGB EOTF 线性化（数转胶假设：像素值=场景反射率），
#   白场（255）= 1.0，sRGB 中灰（≈118）= 0.18。
#   输出统一为相对中灰 E_mid = E/0.18（中灰 = 1.0，白场 ≈ 5.56），
#   与 DNG 路径（E_mid = E/0.1282×2^BE）同轴：stop = log2(E_mid)，中灰恒 0。
#   可承载动态范围 = 8bit 极限约 8 档（1/255 到 1）。
import cv2
import numpy as np

# 中灰归一化除数（sRGB 中灰线性值）
_GRAY_E = 0.18


def standardize(image):
    """标准化图像尺寸"""
    # 确定短边尺寸
    min_size = 3024  # 与 DNG 路径统一（原 3000 为笔误）

    # 获取原始尺寸
    height, width = image.shape[:2]
    # 确定缩放比例
    if height < width:
        # 竖图 - 高度为短边
        scale_factor = min_size / height
        new_height = min_size
        new_width = int(width * scale_factor)
    else:
        # 横图 - 宽度为短边
        scale_factor = min_size / width
        new_width = min_size
        new_height = int(height * scale_factor)

    # 确保新尺寸为偶数（避免某些处理问题）
    new_width = new_width + 1 if new_width % 2 != 0 else new_width
    new_height = new_height + 1 if new_height % 2 != 0 else new_height
    interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (new_width, new_height),
                       interpolation=interpolation)
    return image


def split(image):
    """分离 RGB 通道并做 sRGB → 线性化 → 中灰归一化，返回 E_mid 三通道。"""
    b, g, r = cv2.split(image)

    # 像素 → [0,1]（白场 255 → 1.0）→ sRGB 线性化 → ÷中灰（中灰 E_mid=1）
    b_norm = b.astype(np.float32) / 255.0
    g_norm = g.astype(np.float32) / 255.0
    r_norm = r.astype(np.float32) / 255.0

    # 定义 sRGB 到线性 RGB 的转换函数（向量化操作）
    def srgb_to_linear(c):
        linear = np.where(c <= 0.04045,
                          c / 12.92,
                          ((c + 0.055) / 1.055) ** 2.4)
        return linear

    lux_b = srgb_to_linear(b_norm)
    lux_g = srgb_to_linear(g_norm)
    lux_r = srgb_to_linear(r_norm)
    lux_r = lux_r / _GRAY_E
    lux_g = lux_g / _GRAY_E
    lux_b = lux_b / _GRAY_E
    return lux_r, lux_g, lux_b


def jpeg_to_lux(uploaded_image):
    """将上传的 JPEG/PNG 图像转换为相对中灰 E_mid（按输入契约）。"""
    try:
        # 读取上传的文件
        image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # 标准化图像尺寸
        standardized_image = standardize(image)
        # 计算光照度图像
        lux_r, lux_g, lux_b = split(standardized_image)
        return lux_r, lux_g, lux_b
    except Exception as e:
        raise RuntimeError(f"处理上传图像时发生错误: {str(e)}")
