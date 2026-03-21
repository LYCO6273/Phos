#赛博请神
import cv2
import numpy as np


def luminance(lux_r, lux_g, lux_b):
    """进行线性分光"""

    # 按比例计算不同频段亮度
    lux_r = 0.95 * lux_r + 0.07 * lux_g
    lux_g = 0.05 * lux_r + 0.85 * lux_g + 0.10 * lux_b
    lux_b = 0.03 * lux_g + 1.00 * lux_b
    lux_total = 0.2 * lux_r + 0.35 * lux_g + 0.4 * lux_b

    return lux_r,lux_g,lux_b,lux_total

def average(lux_total):
    """计算图像的平均亮度 (0-1)"""
    # 计算平均亮度
    avg_lux = np.mean(lux_total)
    avg_lux = np.clip(avg_lux, 0, 1)
    return avg_lux

def grain(lux):
    """基于加权随机的颗粒模拟"""

    avrl = average(lux)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.75 + 0.10
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens, 0.35, 0.65)
    #对敏感度进行裁切

    noise = np.random.normal(0, 1, lux.shape).astype(np.float32)
    noise = noise ** 2
    noise = noise * (np.random.choice([-1, 1], lux.shape))
    weights = (0.5 - np.abs(lux - 0.5)) * 2
    weights = np.clip(weights, 0.05, 0.9)
    sens_grain = np.clip(sens, 0.4, 0.6)
    weighted_noise = noise * weights * sens_grain
    noise = None
    weights = None
    weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
    weighted_noise = np.clip(weighted_noise, -1, 1)

    return weighted_noise

def opt_h (lux_total):
    """光学扩散模型；为了简化计算，暂且使用基于加权高斯模糊的方式进行快速近似"""

    avrl = average(lux_total)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.75 + 0.10
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens, 0.35, 0.7)
    #对敏感度进行裁切
    strg = 23 * sens**2
    #确定扩散强度
    rads = np.clip(int(100 * (sens**2) ), 1, 100)
    #确定扩散半径

    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    #确保核尺寸为奇数
    lux_total = np.clip(lux_total-0.8, 0, 1) * 5
    weights = (lux_total**5) * sens
    #根据点光源强度确定权重图
    weights = np.clip(weights, 0, 1)
    #裁切权重图以防超限
    bloom_base = cv2.GaussianBlur(lux_total * weights, (ksize * 3, ksize * 3), sens * 35)
    #应用高斯模糊
    bloom_effect = bloom_base * weights * strg
    #应用权重

    lux_h =bloom_effect * 0.15
    #将光晕效果叠加回底片

    return lux_h

def opt_r (lux_r):
    """光学扩散模型；为了简化计算，暂且使用基于加权高斯模糊的方式进行快速近似"""

    avrl = average(lux_r)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.75 + 0.10
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens, 0.35, 0.7)
    #对敏感度进行裁切
    strg = 23 * sens**2
    #确定扩散强度
    rads = np.clip(int(55 * (sens**2) ), 1, 50)
    #确定扩散半径

    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    #确保核尺寸为奇数

    weights = (lux_r**3.52) * sens
    #根据点光源强度确定权重图
    weights = np.clip(weights, 0, 1)
    #裁切权重图以防超限
    bloom_base = cv2.GaussianBlur(lux_r * weights, (ksize * 3, ksize * 3), sens * 35)
    #应用高斯模糊
    bloom_effect = bloom_base * weights * strg
    #应用权重

    lux_r = lux_r + bloom_effect * 0.15 - weights * 0.15
    #将光晕效果叠加回底片

    return lux_r

def opt_g (lux_g):
    """光学扩散模型；为了简化计算，暂且使用基于加权高斯模糊的方式进行快速近似"""

    avrl = average(lux_g)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.75 + 0.10
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens, 0.35, 0.7)
    #对敏感度进行裁切
    strg = 23 * sens**2
    #确定扩散强度
    rads = np.clip(int(45 * (sens**2) ), 1, 50)
    #确定扩散半径

    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    #确保核尺寸为奇数

    weights = (lux_g**3.52) * sens
    #根据点光源强度确定权重图
    weights = np.clip(weights, 0, 1)
    #裁切权重图以防超限
    bloom_base = cv2.GaussianBlur(lux_g * weights, (ksize * 3, ksize * 3), sens * 35)
    #应用高斯模糊
    bloom_effect = bloom_base * weights * strg
    #应用权重

    lux_g = lux_g + bloom_effect * 0.15 - weights * 0.15
    #将光晕效果叠加回底片

    return lux_g

def opt_b (lux_b):
    """光学扩散模型；为了简化计算，暂且使用基于加权高斯模糊的方式进行快速近似"""

    avrl = average(lux_b)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.75 + 0.10
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens, 0.35, 0.7)
    #对敏感度进行裁切
    strg = 23 * sens**2
    #确定扩散强度
    rads = np.clip(int(35 * (sens**2) ), 1, 50)
    #确定扩散半径

    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    #确保核尺寸为奇数

    weights = (lux_b**3.52) * sens
    #根据点光源强度确定权重图
    weights = np.clip(weights, 0, 1)
    #裁切权重图以防超限
    bloom_base = cv2.GaussianBlur(lux_b * weights, (ksize * 3, ksize * 3), sens * 35)
    #应用高斯模糊
    bloom_effect = bloom_base * weights * strg
    #应用权重

    lux_b = lux_b + bloom_effect * 0.15 - weights * 0.15
    #将光晕效果叠加回底片

    return lux_b




def pos_tone_red(lux_r):
    """基于正片成像规律的映射"""

    xp = np.array([0.0000, 0.5053, 0.7477, 0.9428, 1.1167, 1.3750, 1.5964, 1.8178,
                    2.0286, 2.2500, 2.4556, 2.6875, 2.8193, 2.9827, 3.1250, 3.3833, 4.0000], dtype=np.float32)
    fp = np.array([3.2950, 3.2909, 3.2606, 3.2000, 3.1030, 2.8242, 2.3394, 1.8545,
                    1.4121, 0.9939, 0.6667, 0.3636, 0.2485, 0.1636, 0.1212, 0.0920, 0.0900], dtype=np.float32)
    #用于确定曲线的控制点,来自Provia 100F的技术文档
    
    relative_log = 3.33 * (0.247190 * np.log10(5.555556 * lux_r + 0.072272) + 0.385537)
    relative_log = np.clip(relative_log, 0, 4)  # 裁切到 xp 的对应范围
    density = np.interp(relative_log, xp, fp) #利用插值曲线计算密度
    Pt = 10.0 ** (-density) #根据密度计算透光率
    result = np.clip((Pt-0.0004) * 61.00 , 0, 1) #将透光率映射回亮度，并裁切到0-1范围
    result = result ** (1/2.7) #应用伽马校正
    return result

def pos_tone_green(lux_g):
    """基于正片成像规律的映射"""

    xp = np.array([0.0000, 0.5053, 0.7583, 0.9428, 1.1431, 1.2907, 1.3803, 1.5648, 1.8178,
                    2.0286, 2.2500, 2.4556, 2.6875, 2.8193, 2.9827, 3.1250, 3.3833, 4.0000], dtype=np.float32)
    fp = np.array([3.4303, 3.4182, 3.4121, 3.3697, 3.2485, 3.0788, 2.9091, 2.4970, 1.8606,
                    1.4121, 0.9939, 0.6667, 0.3636, 0.2485, 0.1636, 0.1212, 0.0920, 0.0900], dtype=np.float32)
    #用于确定曲线的控制点，来自Provia 100F的技术文档
    
    relative_log = 3.33 * (0.247190 * np.log10(5.555556 * lux_g + 0.072272) + 0.385537)
    relative_log = np.clip(relative_log, 0, 4)  # 裁切到 xp 的对应范围
    density = np.interp(relative_log, xp, fp) #利用插值曲线计算密度
    Pt = 10.0 ** (-density) #根据密度计算透光率
    result = np.clip((Pt-0.0004) * 63.00 , 0, 1) #将透光率映射回亮度，并裁切到0-1范围
    result = result ** (1/2.7) #应用伽马校正
    return result

def pos_tone_blue(lux_b):
    """基于正片成像规律的映射"""

    xp = np.array([0.0000, 0.5053, 0.6739, 0.8479, 0.9428, 1.1009, 1.2748, 1.3750, 1.5964, 1.8178,
                    2.0286, 2.2500, 2.4556, 2.6875, 2.8193, 2.9827, 3.1250, 3.3833, 4.0000], dtype=np.float32)
    fp = np.array([3.3394, 3.3333, 3.3273, 3.2909, 3.2545, 3.1576, 2.9636, 2.8242, 2.3394, 1.8545,
                    1.4121, 0.9939, 0.6667, 0.3636, 0.2485, 0.1636, 0.1212, 0.0920, 0.0900], dtype=np.float32)
    #用于确定曲线的控制点,来自Provia 100F的技术文档
    
    relative_log = 3.33 * (0.247190 * np.log10(5.555556 * lux_b + 0.072272) + 0.385537)
    relative_log = np.clip(relative_log, 0, 4)  # 裁切到 xp 的对应范围
    density = np.interp(relative_log, xp, fp) #利用插值曲线计算密度
    Pt = 10.0 ** (-density) #根据密度计算透光率
    result = np.clip((Pt-0.0004) * 65.00 , 0, 1) #将透光率映射回亮度，并裁切到0-1范围
    result = result ** (1/2.7) #应用伽马校正
    return result

def process(lux_r,lux_g,lux_b,grain_style):
    """主处理函数"""
   
    lux_r,lux_g,lux_b,lux_total = luminance(lux_r, lux_g, lux_b)

    # 应用光学扩散
    lux_h = opt_h(lux_total)
    lux_r = opt_r(lux_r) + lux_h * 0.15

    lux_g = opt_g(lux_g)

    lux_b = opt_b(lux_b)
    
    # 应用颗粒效果
    if grain_style == "较粗":
        noise_r = grain(lux_r) * 1.5
        noise_g = grain(lux_g) * 1.5
        noise_b = grain(lux_b) * 1.5
    elif grain_style == "柔和":
        noise_r = grain(lux_r) * 0.5
        noise_g = grain(lux_g) * 0.5
        noise_b = grain(lux_b) * 0.5
    elif grain_style == "不使用":
        noise_r = np.zeros_like(lux_r)
        noise_g = np.zeros_like(lux_g)
        noise_b = np.zeros_like(lux_b)
    else:
        noise_r = grain(lux_r)
        noise_g = grain(lux_g)
        noise_b = grain(lux_b)
        
    lux_r = np.clip(lux_r + noise_r * 0.06 + noise_b * 0.02 + noise_g * 0.02, 0, 1)
    lux_g = np.clip(lux_g + noise_g * 0.06 + noise_r * 0.02 + noise_b * 0.02, 0, 1)
    lux_b = np.clip(lux_b + noise_b * 0.06 + noise_r * 0.02 + noise_g * 0.02, 0, 1)

    # 应用负片映射
    result_r = pos_tone_red(lux_r)
    result_g = pos_tone_green(lux_g)
    result_b = pos_tone_blue(lux_b)

    # 转换回8位图像
    film_r = (result_r * 255).astype(np.uint8)
    film_g = (result_g * 255).astype(np.uint8)
    film_b = (result_b * 255).astype(np.uint8)
    film = cv2.merge((film_r, film_g, film_b))

    return film
