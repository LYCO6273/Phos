"""
"No LUTs, we calculate LUX."

你说的对，但是 Phos. 是基于「计算光学」概念的胶片模拟。
通过计算光在底片上的行为，复现自然、柔美、立体的胶片质感。

这是一个原理验证demo，图像处理部分基于opencv，交互基于
streamlit平台制作，部分代码使用了AI辅助生成。

如果您发现了项目中的问题，或是有更好的想法想要分享，还请
通过邮箱 lyco_p@163.com 与我联系，我将不胜感激。

Hello! Phos. is a film simulation app based on 
the idea of "Computational optical imaging“. 
By calculating the optical effects on the film,
we could recurrent the natural, soft, and elegant
tone of these classical films.

This is a demo for idea testing. The image processing
part is based on OpenCV, and the interaction is built
on the Streamlit platform. Some of the code was
generated with the assistance of AI.

If you find any issues in the project or have better
ideas you would like to share, please contact me via
email at lyco_p@163.com. I would be very grateful.

"""
import streamlit as st

# 设置页面配置 
st.set_page_config(
    page_title="Phos. 胶片模拟",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#赛博请神
import cv2
import numpy as np
import time
from PIL import Image
import io

def film_choose(film_type):
    if film_type == ("NC200"):
        r_r = 0.77 #红色感光层吸收的红光
        r_g = 0.12 #红色感光层吸收的绿光
        r_b = 0.18 #红色感光层吸收的蓝光
        g_r = 0.08 #绿色感光层吸收的红光
        g_g = 0.85 #绿色感光层吸收的绿光
        g_b = 0.23 #绿色感光层吸收的蓝光
        b_r = 0.08 #蓝色感光层吸收的红光
        b_g = 0.09 #蓝色感光层吸收的绿光
        b_b = 0.92 #蓝色感光层吸收的蓝光
        t_r = 0.25 #全色感光层吸收的红光
        t_g = 0.35 #全色感光层吸收的绿光
        t_b = 0.35 #全色感光层吸收的蓝光
        color_type = ("color") #色彩类型
        sens_factor = 1.15 #高光敏感系数
        d_r = 0.46 #红色感光层接受的散射光
        l_r = 0.27 #红色感光层接受的直射光
        x_r = 1.18 #红色感光层的响应系数
        d_g = 0.42 #绿色感光层接受的散射光
        l_g = 0.23 #绿色感光层接受的直射光
        x_g = 1.02 #绿色感光层的响应系数
        d_b = 0.39 #蓝色感光层接受的散射光
        l_b = 0.27 #蓝色感光层接受的直射光
        x_b = 0.78 #蓝色感光层的响应系数
        d_l = None #全色感光层接受的散射光
        l_l = None #全色感光层接受的直射光
        x_l = None #全色感光层的响应系数
        contrast=2.9 #contrast: 对比度系数，控制曲线陡峭程度
        center=0.3 #center: 曲线中心点位置，控制整体曝光偏移
        grainy = 0.08 #颗粒度
    elif film_type == ("FS200"):
        r_r = 0 #红色感光层吸收的红光
        r_g = 0 #红色感光层吸收的绿光
        r_b = 0 #红色感光层吸收的蓝光
        g_r = 0 #绿色感光层吸收的红光
        g_g = 0 #绿色感光层吸收的绿光
        g_b = 0 #绿色感光层吸收的蓝光
        b_r = 0 #蓝色感光层吸收的红光
        b_g = 0 #蓝色感光层吸收的绿光
        b_b = 0 #蓝色感光层吸收的蓝光
        t_r = 0.15 #全色感光层吸收的红光
        t_g = 0.3 #全色感光层吸收的绿光
        t_b = 0.55 #全色感光层吸收的蓝光
        color_type = ("single") #色彩类型
        sens_factor = 1.0 #高光敏感系数
        d_r = 0 #红色感光层接受的散射光
        l_r = 0 #红色感光层接受的直射光
        x_r = 0 #红色感光层的响应系数
        d_g = 0 #绿色感光层接受的散射光
        l_g = 0 #绿色感光层接受的直射光
        x_g = 0 #绿色感光层的响应系数
        d_b = 0 #蓝色感光层接受的散射光
        l_b = 0 #蓝色感光层接受的直射光
        x_b = 0 #蓝色感光层的响应系数
        d_l = 0.85 #全色感光层接受的散射光
        l_l = 0.32 #全色感光层接受的直射光
        x_l = 1.15 #全色感光层的响应系数
        grainy = 0.1 #颗粒度
        contrast=3.2 #contrast: 对比度系数，控制曲线陡峭程度
        center=0.3 #center: 曲线中心点位置，控制整体曝光偏移
    elif film_type == ("AS100"):
        r_r = 0 #红色感光层吸收的红光
        r_g = 0 #红色感光层吸收的绿光
        r_b = 0 #红色感光层吸收的蓝光
        g_r = 0 #绿色感光层吸收的红光
        g_g = 0 #绿色感光层吸收的绿光
        g_b = 0 #绿色感光层吸收的蓝光
        b_r = 0 #蓝色感光层吸收的红光
        b_g = 0 #蓝色感光层吸收的绿光
        b_b = 0 #蓝色感光层吸收的蓝光
        t_r = 0.38 #全色感光层吸收的红光
        t_g = 0.12 #全色感光层吸收的绿光
        t_b = 0.5 #全色感光层吸收的蓝光
        color_type = ("single") #色彩类型
        sens_factor = 1.28 #高光敏感系数
        d_r = 0 #红色感光层接受的散射光
        l_r = 0 #红色感光层接受的直射光
        x_r = 0 #红色感光层的响应系数
        d_g = 0 #绿色感光层接受的散射光
        l_g = 0 #绿色感光层接受的直射光
        x_g = 0 #绿色感光层的响应系数
        d_b = 0 #蓝色感光层接受的散射光
        l_b = 0 #蓝色感光层接受的直射光
        x_b = 0 #蓝色感光层的响应系数
        d_l = 0.55 #全色感光层接受的散射光
        l_l = 0.38 #全色感光层接受的直射光
        x_l = 1.35 #全色感光层的响应系数
        grainy = 0.05 #颗粒度
        contrast=2.5 #contrast: 对比度系数，控制曲线陡峭程度
        center=0.3 #center: 曲线中心点位置，控制整体曝光偏移
    
    return r_r,r_g,r_b,g_r,g_g,g_b,b_r,b_g,b_b,t_r,t_g,t_b,color_type,sens_factor,d_r,l_r,x_r,d_g,l_g,x_g,d_b,l_b,x_b,d_l,l_l,x_l,grainy,contrast,center
    #选取胶片类型

def standardize(image):
    """标准化图像尺寸"""
    min_size=3112
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
    # 使用高质量插值方法调整尺寸
    # 缩小图像时使用INTER_AREA，放大时使用INTER_LANCZOS4
    interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LANCZOS4
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    return resized_image
    #统一尺寸

def luminance(image,color_type,r_r,r_g,r_b,g_r,g_g,g_b,b_r,b_g,b_b,t_r,t_g,t_b):
    """计算亮度图像 (0-1范围)"""
    # 分离RGB通道
    b, g, r = cv2.split(image)
    
    # 转换为浮点数
    b_float = b.astype(np.float32) / 255.0
    g_float = g.astype(np.float32) / 255.0
    r_float = r.astype(np.float32) / 255.0
    
    # 按比例计算不同频段亮度
    if color_type == ("color"):
        lux_r = r_r * r_float + r_g * g_float + r_b * b_float
        lux_r = np.power(lux_r, 1.0/2.2)

        lux_g = g_r * r_float + g_g * g_float + g_b * b_float
        lux_g = np.power(lux_g, 1.0/2.2)

        lux_b = b_r * r_float + b_g * g_float + b_b * b_float
        lux_b = np.power(lux_b, 1.0/2.2)

        lux_total = t_r * r_float + t_g * g_float + t_b * b_float
    else:
        lux_total = t_r * r_float + t_g * g_float + t_b * b_float
        lux_total = np.power(lux_total, 1.0/2.2)

        lux_r = None
        lux_g = None
        lux_b = None

    return lux_r,lux_g,lux_b,lux_total
    #实现对源图像的分光并整合输出

def average(lux_total):
    """计算图像的平均亮度 (0-1)"""
    # 计算平均亮度 (0-255)
    avg_lux = np.mean(lux_total)
    # 归一化到0-1范围
    return avg_lux
    #计算平均亮度

def grain(lux_r,lux_g,lux_b,lux_total,color_type,sens):
    # 基于亮度加权的颗粒模拟
    if color_type == ("color"):
        # 创建正态分布噪声
        noise_p = np.random.normal(0,1, lux_r.shape).astype(np.float16)
        noise_n = np.random.normal(0,1, lux_r.shape).astype(np.float16)
        noise = noise_p ** 2 - noise_n ** 2
        # 创建权重图 
        weights =(0.5 - np.abs(lux_r - 0.5)) * 2
        weights = np.clip(weights,0.2,1)
        # 应用权重
        sens_grain = np.clip (sens,0.4,0.6)
        weighted_noise = noise * weights* sens_grain
        # 为颗粒赋形
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_r = np.clip(weighted_noise, -1,1)
        # 应用颗粒
    
        # 创建正态分布噪声
        noise_p = np.random.normal(0,1, lux_g.shape).astype(np.float16)
        noise_n = np.random.normal(0,1, lux_g.shape).astype(np.float16)
        noise = noise_p ** 2 - noise_n ** 2
        # 创建权重图 
        weights =(0.5 - np.abs(lux_g - 0.5)) * 2
        weights = np.clip(weights,0.2,1)
        # 应用权重
        sens_grain = np.clip (sens,0.4,0.6)
        weighted_noise = noise * weights* sens_grain
        # 为颗粒赋形
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_g = np.clip(weighted_noise, -1,1)
        # 应用颗粒
        
        # 创建正态分布噪声
        noise_p = np.random.normal(0,1, lux_b.shape).astype(np.float16)
        noise_n = np.random.normal(0,1, lux_b.shape).astype(np.float16)
        noise = noise_p ** 2 - noise_n ** 2
        # 创建权重图 
        weights =(0.5 - np.abs(lux_b - 0.5)) * 2
        weights = np.clip(weights,0.2,1)
        # 应用权重
        sens_grain = np.clip (sens,0.4,0.6)
        weighted_noise = noise * weights* sens_grain
        # 为颗粒赋形
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_b = np.clip(weighted_noise, -1,1)
        # 应用颗粒
        weighted_noise_total = None
    
    else:
        # 创建正态分布噪声
        noise_p = np.random.normal(0,1, lux_total.shape).astype(np.float16)
        noise_n = np.random.normal(0,1, lux_total.shape).astype(np.float16)
        noise = noise_p ** 2 - noise_n ** 2
        # 创建权重图 
        weights =(0.5 - np.abs(lux_total - 0.5)) * 2
        weights = np.clip(weights,0.2,1)
        # 应用权重
        sens_grain = np.clip (sens,0.4,0.6)
        weighted_noise = noise * weights* sens_grain
        # 为颗粒赋形
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_total = np.clip(weighted_noise, -1,1)
        # 应用颗粒
        weighted_noise_r = None
        weighted_noise_g = None
        weighted_noise_b = None

    # 应用颗粒


    return weighted_noise_r,weighted_noise_g,weighted_noise_b,weighted_noise_total
    #创建颗粒函数

def response(lux_r,lux_g,lux_b,lux_total,color_type,contrast,center):
    """
    使用Logistic函数计算胶片密度响应
    
    参数:
    lux: 曝光量 (0-1)
    d_min: 最小密度
    d_max: 最大密度
    返回:
    density: 显影密度 (0-1归一化)
    """
    
    d_min=0.15
    d_max=2.8

    # Logistic函数
    # 将lux从[0,1]映射到[-∞,+∞]的对数空间

    if color_type == "color":
        #感红色乳剂层
        lux_r = np.clip(lux_r,0,100)
        log_exposure = np.log(lux_r + 1e-6)  # 加小值避免log(0)
        # 标准化对数曝光量
        normalized_log_r = (log_exposure - np.log(center)) * contrast
        # Logistic函数计算原始密度
        logistic_response_r = 1.0 / (1.0 + np.exp(-normalized_log_r))
        # 映射到实际密度范围
        density_r = d_min + (d_max - d_min) * logistic_response_r
        # 归一化到0-1范围
        density_r = (density_r - d_min) / (d_max - d_min)
        np.clip(density_r, 0.0, 1.0)
    
        #感绿色乳剂层
        lux_g = np.clip(lux_g,0,100)
        log_exposure = np.log(lux_g + 1e-6)  # 加小值避免log(0)
        # 标准化对数曝光量
        normalized_log_g = (log_exposure - np.log(center)) * contrast
        # Logistic函数计算原始密度
        logistic_response_g = 1.0 / (1.0 + np.exp(-normalized_log_g))
        # 映射到实际密度范围
        density_g = d_min + (d_max - d_min) * logistic_response_g
        # 归一化到0-1范围
        density_g = (density_g - d_min) / (d_max - d_min)
        np.clip(density_g, 0.0, 1.0)

        #感蓝色乳剂层
        lux_b = np.clip(lux_b,0,100)
        log_exposure = np.log(lux_b + 1e-6)  # 加小值避免log(0)
        # 标准化对数曝光量
        normalized_log_b = (log_exposure - np.log(center)) * contrast
        # Logistic函数计算原始密度
        logistic_response_b = 1.0 / (1.0 + np.exp(-normalized_log_b))
        # 映射到实际密度范围
        density_b = d_min + (d_max - d_min) * logistic_response_b
        # 归一化到0-1范围
        density_b = (density_b - d_min) / (d_max - d_min)
        np.clip(density_b, 0.0, 1.0)

        density_total = None

    else:
        #全色乳剂层
        lux_total = np.clip(lux_total,0,100)
        log_exposure = np.log(lux_total + 1e-6)  # 加小值避免log(0)
        # 标准化对数曝光量
        normalized_log_total = (log_exposure - np.log(center)) * contrast
        # Logistic函数计算原始密度
        logistic_response_total = 1.0 / (1.0 + np.exp(-normalized_log_total))
        # 映射到实际密度范围
        density_total = d_min + (d_max - d_min) * logistic_response_total
        # 归一化到0-1范围
        density_total = (density_total - d_min) / (d_max - d_min)
        np.clip(density_total, 0.0, 1.0)

        density_r = None
        density_g = None
        density_b = None

    return density_r,density_g,density_b,density_total

def opt(lux_r,lux_g,lux_b,lux_total,color_type,sens_factor,d_r,l_r,x_r,d_g,l_g,x_g,d_b,l_b,x_b,d_l,l_l,x_l,grainy,contrast,center):
    avrl = average(lux_total)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.5 + 0.25
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens,0.3,0.7)
    strg = 22 * sens**2 * sens_factor
    rads = np.clip(int(15 * sens**2 * sens_factor),1,100)
    base = 0.10 * sens_factor
    #opt 光学扩散函数
    #sens -- 高光敏感度(0.1-2.0)，值越大更多区域受影响
    #strg -- 光晕强度(0.5-3.0)，值越大柔化效果越强
    #rads -- 光晕扩散半径(5-50)，值越大光晕范围越广
    #base -- 基础扩散强度(0.1-0.5)，保证非高光区也有自然过渡
    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    # 确保核大小为奇数
    
    if color_type == ("color"):

        weights = (base + lux_r**2) * sens 
        weights = np.clip(weights,0,0.9)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_r * weights, (ksize * 3 , ksize * 3),sens * 55)
        #开始高斯模糊
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_r = bloom_effect
        #应用光晕
    
        weights = (base + lux_g**2 ) * sens
        weights = np.clip(weights,0,0.9)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_g * weights, (ksize * 2 +1 , ksize * 2 +1 ),sens * 35)
        #开始高斯模糊
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_g = bloom_effect
        #应用光晕
    
        weights = (base + lux_b**2 ) * sens
        weights = np.clip(weights,0,0.9)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_b * weights, (ksize, ksize),sens * 15)
        #开始高斯模糊
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_b = bloom_effect
        #应用光晕

        (weighted_noise_r,weighted_noise_g,weighted_noise_b,weighted_noise_total) = grain(lux_r,lux_g,lux_b,lux_total,color_type,sens)
        #应用颗粒

        lux_r = bloom_effect_r * d_r + (lux_r**x_r) * l_r + weighted_noise_r * grainy
        lux_g = bloom_effect_g * d_g + (lux_g**x_g) * l_g + weighted_noise_g * grainy
        lux_b = bloom_effect_b * d_b + (lux_b**x_b) * l_b + weighted_noise_b * grainy
        #拼合光层
        lux_total = None
        
        (density_r,density_g,density_b,density_total) = response(lux_r,lux_g,lux_b,lux_total,color_type,contrast,center)
        combined_b = (density_b * 255).astype(np.uint8)
        combined_g = (density_g * 255).astype(np.uint8)
        combined_r = (density_r * 255).astype(np.uint8)
        film = cv2.merge([combined_r, combined_g, combined_b])
        
    else:
        weights = (base + lux_total**2) * sens 
        weights = np.clip(weights,0,0.9)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_total * weights, (ksize * 3 , ksize * 3),sens * 55)
        #开始高斯模糊
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        #应用光晕
        (weighted_noise_r,weighted_noise_g,weighted_noise_b,weighted_noise_total) = grain(lux_r,lux_g,lux_b,lux_total,color_type,sens)
        #应用颗粒
        lux_total = bloom_effect * d_l + (lux_total**x_l) * l_l + weighted_noise_total * grainy

        (density_r,density_g,density_b,density_total) = response(lux_r,lux_g,lux_b,lux_total,color_type,contrast,center)
        film = (density_total * 255).astype(np.uint8)

    return film
    #返回渲染后的光度

def process(uploaded_image,film_type,grain_type):
    
    start_time = time.time()

    # 读取上传的文件
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # 获取胶片参数
    (r_r,r_g,r_b,g_r,g_g,g_b,b_r,b_g,b_b,t_r,t_g,t_b,color_type,sens_factor,d_r,l_r,x_r,d_g,l_g,x_g,d_b,l_b,x_b,d_l,l_l,x_l,grainy,contrast,center) = film_choose(film_type)
    if grain_type == ("默认"):
        grainy = grainy * 1.0
    elif grain_type == ("柔和"):
        grainy = grainy * 0.5
    elif grain_type == ("较粗"):
        grainy = grainy * 1.5
    elif grain_type == ("不使用"):
        grainy = grainy * 0

    # 调整尺寸
    image = standardize(image)

    (lux_r,lux_g,lux_b,lux_total) = luminance(image,color_type,r_r,r_g,r_b,g_r,g_g,g_b,b_r,b_g,b_b,t_r,t_g,t_b)
    #重建光线
    film = opt(lux_r,lux_g,lux_b,lux_total,color_type,sens_factor,d_r,l_r,x_r,d_g,l_g,x_g,d_b,l_b,x_b,d_l,l_l,x_l,grainy,contrast,center)
    #冲洗底片
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"phos_{timestamp}.jpg"
    process_time = time.time() - start_time

    return film,process_time,output_path
    #执行胶片模拟处理

# 创建侧边栏
with st.sidebar:
    st.header("Phos. 胶片模拟")
    st.subheader("基于「计算光学」的胶片模拟")
    st.text("")
    st.text("原理验证demo")
    st.text("ver_0.1.1")
    st.text("")
    st.text("🎞️ 胶片设置")

    # 胶片类型选择
    film_type = st.selectbox(
        "请选择胶片:",
        ["NC200","AS100","FS200"],
        index=0,
        help='''选择要模拟的胶片类型:

        NC200:灵感来自富士C200彩色负片和扫描仪
        SP3000，旨在模仿经典的“富士色调”，通过
        还原“记忆色”，唤起对胶片的情感。

        AS100：灵感来自富士ACROS系列黑白胶片，
        为正全色黑白胶片，对蓝色最敏感，红色次
        之，绿色最弱，成片灰阶细腻，颗粒柔和，
        对比适中。

        FS200：黑白正片⌈光⌋，在开发初期作为原理
        验证模型所使用，对蓝色较敏感，对红色较
        不敏感，对比鲜明，颗粒较粗。
        '''
    )

    grain_type = st.selectbox(
        "胶片颗粒度：",
        ["默认","柔和","较粗","不使用"],
        index = 0,
        help="选择胶片的颗粒度",
    )

    st.success(f"已选择胶片: {film_type}") 
    # 文件上传器
    uploaded_image = None
    uploaded_image = st.file_uploader(
    "选择一张照片来开始冲洗",
    type=["jpg", "jpeg", "png"],
    help="上传一张照片冲洗试试看吧"
    )

if uploaded_image is not None:
    (film,process_time,output_path) = process(uploaded_image,film_type,grain_type)
    st.image(film, width="stretch")
    st.success(f"底片显影好了，用时 {process_time:.2f}秒") 
    
    # 添加下载按钮
    film_pil = Image.fromarray(film)
    buf = io.BytesIO()
    film_pil.save(buf, format="JPEG", quality=100)
    byte_im = buf.getvalue()
    
    # 创建字节缓冲区
    buf = io.BytesIO()
    film_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 下载高清图像",
        data=byte_im,
        file_name=output_path,
        mime="image/jpeg"
    )
    uploaded_image = None