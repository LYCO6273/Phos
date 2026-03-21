#赛博请神
import cv2
import numpy as np

def standardize(image):
    """标准化图像尺寸"""
    
    #确定短边尺寸
    min_size=3000

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
    image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    interpolation = None

    return image

def split(image):
    # 分离RGB通道
    b, g, r = cv2.split(image)
    
    # 转换为浮点数,并反转伽马校正（输入图像是sRGB）
    lux_b = (b.astype(np.float32) / 255.0) ** (1/2.2)
    lux_g = (g.astype(np.float32) / 255.0) ** (1/2.2)
    lux_r = (r.astype(np.float32) / 255.0) ** (1/2.2)

    return lux_r, lux_g, lux_b

def Jpeg_to_lux(uploaded_image):
    """将上传的JPEG图像转换为标准化的亮度图像"""
    try:
    
        # 读取上传的文件
        image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
        uploaded_image = None
        
        # 标准化图像尺寸
        standardized_image = standardize(image)
        
        # 计算亮度图像
        lux_r, lux_g, lux_b = split(standardized_image)
        
        return lux_r, lux_g, lux_b
    
    except Exception as e:
        raise RuntimeError(f"处理上传图像时发生错误: {str(e)}")