#赛博请神
import cv2
import numpy as np
import rawpy
import io

def standardize(image):
    """标准化图像尺寸"""
    min_size = 3000
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
    image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return image

def split_linear(bgr_image):
    """从 BGR 图像分离通道，返回线性 RGB 浮点值 (r, g, b)"""
    b, g, r = cv2.split(bgr_image)
    return r, g, b

def RAW_to_lux(uploaded_image):
    """
    将上传的 DNG/RAW 图像转换为线性 RGB 亮度图像
    输入：uploaded_image 是 Streamlit 上传的文件对象
    输出：lux_r, lux_g, lux_b —— 线性浮点亮度值，范围 0～1
    """
    try:
        # 读取上传文件的字节
        file_bytes = uploaded_image.read()
        uploaded_image.seek(0)  # 重置指针，方便后续可能再次读取

        # 使用 rawpy 处理 RAW 数据
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            # 后期处理：输出线性 sRGB（16 位，无伽马校正）
            rgb = raw.postprocess(
                use_camera_wb=True,          # 使用相机白平衡
                output_bps=16,                # 输出 16 位整数
                output_color=rawpy.ColorSpace.sRGB,  # 转换到 sRGB 色彩空间
                gamma=(1, 1),                  # 禁用伽马校正，得到线性数据
                no_auto_bright=True,           # 不自动调整亮度
                bright=1.0                     # 亮度系数保持 1
            )
        # 将 16 位整数转换为 0～1 浮点数
        rgb_float = rgb.astype(np.float32) / 65535.0

        # 转换为 BGR 顺序，以复用 standardize 函数（它期望 BGR）
        bgr = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2BGR)

        # 标准化尺寸
        bgr_std = standardize(bgr)

        # 分离通道得到线性 lux 值
        lux_r, lux_g, lux_b = split_linear(bgr_std)

        return lux_r, lux_g, lux_b

    except Exception as e:
        raise RuntimeError(f"无法处理 RAW 图像: {str(e)}")