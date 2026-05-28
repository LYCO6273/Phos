#赛博请神
import cv2
import numpy as np


def luminance(lux_r, lux_g, lux_b):
    """进行线性分光"""

    lux = 0.18 * lux_r + 0.44 * lux_g + 0.38 * lux_b

    return lux

def average(lux):
    """计算图像的平均亮度 (0-1)"""
    # 计算平均亮度
    avg_lux = np.mean(lux)
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

def opt (lux):
    """光学扩散模型；为了简化计算，暂且使用基于加权高斯模糊的方式进行快速近似"""

    avrl = average(lux)
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

    weights = (lux**3.24) * sens
    #根据点光源强度确定权重图
    weights = np.clip(weights, 0, 1)
    #裁切权重图以防超限
    bloom_base = cv2.GaussianBlur(lux * weights, (ksize * 3, ksize * 3), sens * 35)
    #应用高斯模糊
    bloom_effect = bloom_base * weights * strg
    #应用权重

    lux = lux * 0.88 + bloom_effect * 0.35
    #将光晕效果叠加回底片

    return lux

def neg_tone(lux):
    """基于负片成像规律的映射
    写一下思路吧：
    由于胶片的曝光是对数性的（而且还是常用对数），先要将线性的曝光量对数化并进行相应的校准，然后根据
    从技术文档里找到的密度曲线进行映射，再去模拟扫描和翻拍的规律，最后做反转和其他调整得到成片。"""

    xp = np.array([0.000, 0.242, 0.503, 0.758, 0.993, 1.255, 1.497, 1.745, 1.993,
                   2.248, 2.490, 2.745, 3.000, 3.261, 3.490, 3.739, 4.000, 4.235,
                   4.490, 4.758, 5.000], dtype=np.float32)
    fp = np.array([0.175, 0.181, 0.188, 0.208, 0.261, 0.341, 0.467, 0.633, 0.792,
                   0.958, 1.117, 1.277, 1.442, 1.608, 1.754, 1.914, 2.080, 2.199,
                   2.272, 2.312, 2.338], dtype=np.float32)
    #用于确定曲线的控制点，从技术文档里扒下来的（笑
    relative_log = 5 * (0.247190 * np.log10(5.555556 * lux + 0.072272) + 0.385537)
    density = np.interp(relative_log, xp, fp) #利用插值曲线计算密度
    Pt = 10.0 ** (-density) #根据密度计算透光率
    result = np.clip((0.669 - Pt)*1.55, 0, 1) #将透光率映射回亮度，反转并裁切到0-1范围
    result = result ** 2.0 #应用伽马校正
    return result

def process(lux_r,lux_g,lux_b,grain_style):
    """主处理函数"""
   
    lux = luminance(lux_r, lux_g, lux_b)

    # 应用光学扩散
    lux = opt(lux)
    
    # 应用颗粒效果
    if grain_style == "较粗":
        noise = grain(lux) * 1.5
    elif grain_style == "柔和":
        noise = grain(lux) * 0.5
    elif grain_style == "不使用":
        noise = np.zeros_like(lux)
    else:
        noise = grain(lux)
    lux = np.clip(lux + noise * 0.12 , 0, 1)

    # 应用负片映射
    result = neg_tone(lux)

    # 转换回8位图像
    film = (result * 255).astype(np.uint8)

    return film



