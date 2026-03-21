#赛博请神
import cv2
import numpy as np


def luminance(lux_r, lux_g, lux_b):
    """进行线性分光"""

    # 按比例计算不同频段亮度
    lux_r = 0.84 * lux_r + 0.07 * lux_g + 0.09 * lux_b
    lux_g = 0.04 * lux_r + 0.89 * lux_g + 0.07 * lux_b
    lux_b = 0.03 * lux_r + 0.07 * lux_g + 0.90 * lux_b
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

    lux_r = lux_r * 0.88 + bloom_effect * 0.35
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

    lux_g = lux_g * 0.88 + bloom_effect * 0.22
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

    lux_b = lux_b * 0.88 + bloom_effect * 0.15
    #将光晕效果叠加回底片

    return lux_b




def neg_tone_cyan(lux_r):
    """基于负片成像规律的映射(青色层，感红光)"""

    xp = np.array([0.0000, 0.2574, 0.4950, 0.7525, 1.0099, 1.2574, 1.5050, 1.7525, 2.0000,
               2.2475, 2.5050, 2.7525, 3.0099, 3.2673, 3.5050, 3.7525, 4.0000], dtype=np.float32)
    fp = np.array([0.2550, 0.2550, 0.2848, 0.3543, 0.4636, 0.5927, 0.7318, 0.8510, 1.0000,
               1.1291, 1.2682, 1.4172, 1.5563, 1.6457, 1.7252, 1.7947, 1.8146], dtype=np.float32)
    #用于确定曲线的控制点，从Gold200技术文档里扒下来的（笑,而且据说Gold200和富士C200其实是同一个生产线
    
    relative_log = 5.32 * np.log10(lux_r + 0.32) + 1.72   # 对数化并校准偏移量
    relative_log = np.clip(relative_log, 0, 4)  # 裁切到 xp 的对应范围
    density = np.interp(relative_log, xp, fp) #利用插值曲线计算密度
    Pt = 10.0 ** (-density) #根据密度计算透光率
    result = np.clip((0.556 - Pt)*1.792 * 1.11 , 0, 1) #将透光率映射回亮度，反转并裁切到0-1范围
    result = result ** 2.2 #应用伽马校正
    return result

def neg_tone_magenta(lux_g):
    """基于负片成像规律的映射(品红层，感绿光)"""

    xp = np.array([0.0000, 0.2277, 0.5050, 0.7525, 1.0099, 1.2475, 1.4554, 1.7525, 2.0000,
               2.2574, 2.5050, 2.7525, 3.0000, 3.2574, 3.5050, 3.7525, 4.0000], dtype=np.float32)
    fp = np.array([0.6623, 0.6625, 0.7020, 0.7914, 0.9106, 1.0397, 1.1490, 1.3278, 1.4669,
               1.6159, 1.7550, 1.8940, 2.0232, 2.1126, 2.1921, 2.2616, 2.2914], dtype=np.float32)
    #用于确定曲线的控制点，从Gold200技术文档里扒下来的（笑,而且据说Gold200和富士C200其实是同一个生产线
    
    relative_log = 5.32 * np.log10(lux_g + 0.32) + 1.72   # 对数化并校准偏移量
    relative_log = np.clip(relative_log, 0, 4)  # 裁切到 xp 的对应范围
    density = np.interp(relative_log, xp, fp) #利用插值曲线计算密度
    Pt = 10.0 ** (-density) #根据密度计算透光率
    result = np.clip((0.219 - Pt)*4.522 * 1.08 , 0, 1) #将透光率映射回亮度，反转并裁切到0-1范围
    result = result ** 2.2 #应用伽马校正
    return result

def neg_tone_yellow(lux_b):
    """基于负片成像规律的映射(黄色层，感蓝光)"""

    xp = np.array([0.0000, 0.2376, 0.4455, 0.6733, 0.9208, 1.2574, 1.5050, 1.7624, 2.0000,
               2.2574, 2.5050, 2.7525, 2.9901, 3.2574, 3.5050, 3.7525, 4.0000], dtype=np.float32)
    fp = np.array([0.9901, 0.9702, 0.9702, 1.0298, 1.1391, 1.3377, 1.4868, 1.6755, 1.8146,
               1.9636, 2.1126, 2.2715, 2.4106, 2.5298, 2.6093, 2.6689, 2.6887], dtype=np.float32)
    #用于确定曲线的控制点，从Gold200技术文档里扒下来的（笑,而且据说Gold200和富士C200其实是同一个生产线
    
    relative_log = 5.32 * np.log10(lux_b + 0.32) + 1.72   # 对数化并调整偏移量
    relative_log = np.clip(relative_log, 0, 4)  # 裁切到 xp 的对应范围
    density = np.interp(relative_log, xp, fp) #利用插值曲线计算密度
    Pt = 10.0 ** (-density) #根据密度计算透光率
    result = np.clip((0.103 - Pt)*9.672 * 1.10 , 0, 1) #将透光率映射回亮度，反转并裁切到0-1范围
    result = result ** 2.2 #应用伽马校正
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
        
    lux_r = np.clip(lux_r + noise_r * 0.1 + noise_b * 0.03 + noise_g * 0.03, 0, 1)
    lux_g = np.clip(lux_g + noise_g * 0.1 + noise_r * 0.03 + noise_b * 0.03, 0, 1)
    lux_b = np.clip(lux_b + noise_b * 0.1 + noise_r * 0.03 + noise_g * 0.03, 0, 1)

    # 应用负片映射
    result_r = neg_tone_cyan(lux_r)
    result_g = neg_tone_magenta(lux_g)
    result_b = neg_tone_yellow(lux_b)

    # 转换回8位图像
    film_r = (result_r * 255).astype(np.uint8)
    film_g = (result_g * 255).astype(np.uint8)
    film_b = (result_b * 255).astype(np.uint8)
    film = cv2.merge((film_r, film_g, film_b))

    return film
