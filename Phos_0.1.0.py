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
        exp = 1.0 #曝光补偿系数
        gam = 2.15 #gamma值
        fog = 0 #灰雾
        sens_factor = 1.20 #高光敏感系数
        d_r = 1.48 #红色感光层接受的散射光
        l_r = 0.95 #红色感光层接受的直射光
        x_r = 1.18 #红色感光层的响应系数
        n_r = 0.4 #红色感光层的颗粒度
        d_g = 1.02 #绿色感光层接受的散射光
        l_g = 0.80 #绿色感光层接受的直射光
        x_g = 1.02 #绿色感光层的响应系数
        n_g = 0.4 #绿色感光层的颗粒度
        d_b = 1.02 #蓝色感光层接受的散射光
        l_b = 0.88 #蓝色感光层接受的直射光
        x_b = 0.78 #蓝色感光层的响应系数
        n_b = 0.4 #蓝色感光层的颗粒度
        d_l = None #全色感光层接受的散射光
        l_l = None #全色感光层接受的直射光
        x_l = None #全色感光层的响应系数
        n_l = None #全色感光层的颗粒度
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
        t_g = 0.35 #全色感光层吸收的绿光
        t_b = 0.45 #全色感光层吸收的蓝光
        color_type = ("single") #色彩类型
        exp = 1.0 #曝光补偿系数
        gam = 2.0 #gamma值
        fog = 0 #灰雾
        sens_factor = 1.0 #高光敏感系数
        d_r = 0 #红色感光层接受的散射光
        l_r = 0 #红色感光层接受的直射光
        x_r = 0 #红色感光层的响应系数
        n_r = 0 #红色感光层的颗粒度
        d_g = 0 #绿色感光层接受的散射光
        l_g = 0 #绿色感光层接受的直射光
        x_g = 0 #绿色感光层的响应系数
        n_g = 0 #绿色感光层的颗粒度
        d_b = 0 #蓝色感光层接受的散射光
        l_b = 0 #蓝色感光层接受的直射光
        x_b = 0 #蓝色感光层的响应系数
        n_b = 0 #蓝色感光层的颗粒度
        d_l = 1.25 #全色感光层接受的散射光
        l_l = 1.1 #全色感光层接受的直射光
        x_l = 1.0 #全色感光层的响应系数
        n_l = 0.35 #全色感光层的颗粒度
    elif film_type == ("LC400"):
        r_r = 0.65 #红色感光层吸收的红光
        r_g = 0.19 #红色感光层吸收的绿光
        r_b = 0.19 #红色感光层吸收的蓝光
        g_r = 0.15 #绿色感光层吸收的红光
        g_g = 0.68 #绿色感光层吸收的绿光
        g_b = 0.15 #绿色感光层吸收的蓝光
        b_r = 0.22 #蓝色感光层吸收的红光
        b_g = 0.25 #蓝色感光层吸收的绿光
        b_b = 0.50 #蓝色感光层吸收的蓝光
        t_r = 0.22 #全色感光层吸收的红光
        t_g = 0.30 #全色感光层吸收的绿光
        t_b = 0.45 #全色感光层吸收的蓝光
        color_type = ("color") #色彩类型
        exp = 1.0 #曝光补偿系数
        gam = 2.25 #gamma值
        fog = 0.05 #灰雾
        sens_factor = 1.15 #高光敏感系数
        d_r = 1.25 #红色感光层接受的散射光
        l_r = 1.10 #红色感光层接受的直射光
        x_r = 1.05 #红色感光层的响应系数
        n_r = 0.50 #红色感光层的颗粒度
        d_g = 1.25 #绿色感光层接受的散射光
        l_g = 1.05 #绿色感光层接受的直射光
        x_g = 1.10 #绿色感光层的响应系数
        n_g = 0.50 #绿色感光层的颗粒度
        d_b = 1.05 #蓝色感光层接受的散射光
        l_b = 1.20 #蓝色感光层接受的直射光
        x_b = 0.85 #蓝色感光层的响应系数
        n_b = 0.60 #蓝色感光层的颗粒度
        d_l = 0 #全色感光层接受的散射光
        l_l = 0 #全色感光层接受的直射光
        x_l = 0 #全色感光层的响应系数
        n_l = 0 #全色感光层的颗粒度
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
        t_r = 0.30 #全色感光层吸收的红光
        t_g = 0.12 #全色感光层吸收的绿光
        t_b = 0.45 #全色感光层吸收的蓝光
        color_type = ("single") #色彩类型
        exp = 1.0 #曝光补偿系数
        gam = 2.3 #gamma值
        fog = 0 #灰雾
        sens_factor = 1.28 #高光敏感系数
        d_r = 0 #红色感光层接受的散射光
        l_r = 0 #红色感光层接受的直射光
        x_r = 0 #红色感光层的响应系数
        n_r = 0 #红色感光层的颗粒度
        d_g = 0 #绿色感光层接受的散射光
        l_g = 0 #绿色感光层接受的直射光
        x_g = 0 #绿色感光层的响应系数
        n_g = 0 #绿色感光层的颗粒度
        d_b = 0 #蓝色感光层接受的散射光
        l_b = 0 #蓝色感光层接受的直射光
        x_b = 0 #蓝色感光层的响应系数
        n_b = 0 #蓝色感光层的颗粒度
        d_l = 1.25 #全色感光层接受的散射光
        l_l = 1.10 #全色感光层接受的直射光
        x_l = 1.25 #全色感光层的响应系数
        n_l = 0.20 #全色感光层的颗粒度
    
    return r_r,r_g,r_b,g_r,g_g,g_b,b_r,b_g,b_b,t_r,t_g,t_b,color_type,exp,gam,fog,sens_factor,d_r,l_r,x_r,n_r,d_g,l_g,x_g,n_g,d_b,l_b,x_b,n_b,d_l,l_l,x_l,n_l
    #选取胶片类型
def standardize(image):
    """标准化图像尺寸"""
    min_size=2400
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
        lux_g = g_r * r_float + g_g * g_float + g_b * b_float
        lux_b = b_r * r_float + b_g * g_float + b_b * b_float
        lux_total = t_r * r_float + t_g * g_float + t_b * b_float
    else:
        lux_total = t_r * r_float + t_g * g_float + t_b * b_float
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
def reinhard(hdr,exp,gam):
    #定义reinhard算法，exp为曝光度，gam为伽马值
    
    mapped = hdr * exp
    #定义输入的图像
    mapped = mapped * (mapped/ (1.0 + mapped))
    #应用reinhard算法
    mapped = np.power(mapped, 1.0/gam)
    mapped = np.clip(mapped,0,1)

    #应用gamma矫正
    return mapped
    #创建reinhard函数
def grain(lux_r,lux_g,lux_b,lux_total,color_type,sens):
    """添加胶片颗粒 (输入和输出都是0-1范围)"""
    if color_type == ("color"):
        # 创建正态分布噪声
        noise = np.random.normal(-0.7,0.7, lux_r.shape).astype(np.float32)
        # 创建权重图 (中等亮度区域权重最高)
        weights =(0.65 - np.abs(lux_r - 0.65)) * 2
        weights = np.clip(weights,0.1,1)
        # 应用权重
        sens_grain = np.clip (sens,0.4,0.6)
        weighted_noise = noise * weights* sens_grain
        # 添加轻微模糊
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_r = np.clip(weighted_noise, -1,1)
        # 应用颗粒
    
        # 创建正态分布噪声
        noise = np.random.normal(-0.7,0.7, lux_g.shape).astype(np.float32)
        # 创建权重图 (中等亮度区域权重最高)
        weights =(0.65 - np.abs(lux_g - 0.65)) * 2
        weights = np.clip(weights,0.1,1)
        # 应用权重
        sens_grain = np.clip (sens,0.4,0.65)
        weighted_noise = noise * weights* sens_grain
        # 添加轻微模糊
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_g = np.clip(weighted_noise, -1,1)
        # 应用颗粒
        
        # 创建正态分布噪声
        noise = np.random.normal(-0.7,0.7, lux_b.shape).astype(np.float32)
        # 创建权重图 (中等亮度区域权重最高)
        weights =(0.65 - np.abs(lux_b - 0.65)) * 2
        weights = np.clip(weights,0.1,1)
        # 应用权重
        sens_grain = np.clip (sens,0.4,1)
        weighted_noise = noise * weights* sens_grain
        # 添加轻微模糊
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_b = np.clip(weighted_noise, -1,1)
        weighted_noise_total = None
    # 应用颗粒
    else:
        noise = np.random.normal(-0.7,0.7, lux_total.shape).astype(np.float32)
        # 创建权重图 (中等亮度区域权重最高)
        weights =(0.65 - np.abs(lux_total - 0.65)) * 2
        weights = np.clip(weights,0.1,1)
        # 应用权重
        sens_grain = np.clip (sens,0.4,1)
        weighted_noise = noise * weights* sens_grain
        # 添加轻微模糊
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_total = np.clip(weighted_noise, -1,1)
        weighted_noise_r = None
        weighted_noise_g = None
        weighted_noise_b = None

    # 应用颗粒


    return weighted_noise_r,weighted_noise_g,weighted_noise_b,weighted_noise_total
    #创建颗粒函数
def opt(lux_r,lux_g,lux_b,lux_total,color_type, exp, gam, fog, sens_factor, d_r, l_r, x_r, n_r, d_g, l_g, x_g, n_g, d_b, l_b, x_b, n_b, d_l, l_l, x_l, n_l):
    avrl = average(lux_total)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.75 + 0.10
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens,0.10,0.7)
    strg = 23 * sens**2 * sens_factor
    rads = np.clip(int(20 * sens**2 * sens_factor),1,50)
    base = 0.05 * sens_factor
    #opt 光学扩散函数
    #sens -- 高光敏感度(0.1-2.0)，值越大更多区域受影响
    #strg -- 光晕强度(0.5-3.0)，值越大柔化效果越强
    #rads -- 光晕扩散半径(5-50)，值越大光晕范围越广
    #base -- 基础扩散强度(0.1-0.5)，保证非高光区也有自然过渡
    k = sens * 10.0
    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    # 确保核大小为奇数
    
    if color_type == ("color"):
        weights = (base + lux_r**2) * sens 
        weights = np.clip(weights,0,0.9)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_r * weights, (ksize * 3 , ksize * 3),sens * 55)
        #开始高斯模糊
        blend_factors = weights * strg * 1.5
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_r = bloom_effect
        #应用光晕
    
        weights = (base + lux_g**2 ) * sens
        weights = np.clip(weights,0,0.9)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_g * weights, (ksize * 2 +1 , ksize * 2 +1 ),sens * 35)
        #开始高斯模糊
        blend_factors = weights * strg
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_g = bloom_effect
        #应用光晕
    
        weights = (base + lux_b**2 ) * sens
        weights = np.clip(weights,0,0.9)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_b * weights, (ksize, ksize),sens * 15)
        #开始高斯模糊
        blend_factors = weights * strg
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_b = bloom_effect
        #应用光晕

        (weighted_noise_r,weighted_noise_g,weighted_noise_b,weighted_noise_total) = grain(lux_r,lux_g,lux_b,lux_total,color_type,sens)
        #应用颗粒

        hdr_r = bloom_effect_r * d_r + (lux_r**x_r) * l_r + weighted_noise_r *n_r + fog
        hdr_g = bloom_effect_g * d_g + (lux_g**x_g) * l_g + weighted_noise_g *n_g + fog
        hdr_b = bloom_effect_b * d_b + (lux_b**x_b) * l_b + weighted_noise_b *n_b + fog
        #拼合光层
        
        hdr=hdr_r
        (result) = reinhard(hdr,exp,gam)
        result_r = result    
        hdr=hdr_g
        (result) = reinhard(hdr,exp,gam)
        result_g = result
        hdr=hdr_b
        (result) = reinhard(hdr,exp,gam)
        result_b = result
        #应用reinhard算法防止过曝

        combined_b = (result_b * 255).astype(np.uint8)
        combined_g = (result_g * 255).astype(np.uint8)
        combined_r = (result_r * 255).astype(np.uint8)
        film = cv2.merge([combined_r, combined_g, combined_b])
    else:
        weights = (base + lux_total**2) * sens 
        weights = np.clip(weights,0,0.9)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_total * weights, (ksize * 3 , ksize * 3),sens * 55)
        #开始高斯模糊
        blend_factors = weights * strg * 1.5
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        #应用光晕
        (weighted_noise_r,weighted_noise_g,weighted_noise_b,weighted_noise_total) = grain(lux_r,lux_g,lux_b,lux_total,color_type,sens)
        #应用颗粒
        hdr = bloom_effect * d_l + (lux_total**x_l) * l_l + weighted_noise_total *n_l + fog
        #拼合光层
        (result) = reinhard(hdr,exp,gam)
        film = (result * 255).astype(np.uint8)

    return film
    #返回渲染后的光度
    #进行底片成像
    #准备暗房工具

def process(uploaded_image,film_type):
    
    start_time = time.time()

    # 读取上传的文件
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # 获取胶片参数
    (r_r, r_g, r_b, g_r, g_g, g_b, b_r, b_g, b_b, t_r, t_g, t_b, color_type, exp, gam, fog, sens_factor, d_r, l_r, x_r, n_r, d_g, l_g, x_g, n_g, d_b, l_b, x_b, n_b, d_l, l_l, x_l, n_l) = film_choose(film_type)
    # 调整尺寸
    image = standardize(image)

    (lux_r,lux_g,lux_b,lux_total) = luminance(image,color_type,r_r,r_g,r_b,g_r,g_g,g_b,b_r,b_g,b_b,t_r,t_g,t_b)
    #重建光线
    film = opt(lux_r,lux_g,lux_b,lux_total,color_type, exp, gam, fog, sens_factor, d_r, l_r, x_r, n_r, d_g, l_g, x_g, n_g, d_b, l_b, x_b, n_b, d_l, l_l, x_l, n_l)
    #冲洗底片
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"phos_{timestamp}.jpg"
    process_time = time.time() - start_time

    return film,process_time,output_path
    #执行胶片模拟处理

# 创建侧边栏
with st.sidebar:
    st.header("Phos. 胶片模拟")
    st.subheader("基于计算光学的胶片模拟")
    st.text("")
    st.text("早期演示demo")
    st.text("ver_0.1.0")
    st.text("")
    st.text("🎞️ 胶片设置")
    # 胶片类型选择
    film_type = st.selectbox(
        "请选择胶片:",
        ["NC200","AS100","LC400","FS200"],
        index=0,
        help='''选择要模拟的胶片类型:

        NC200:灵感来自富士C200彩色负片和扫描仪
        SP3000，旨在模仿经典的“富士色调”，通过
        还原“记忆色”，唤起对胶片的情感。

        AS100：灵感来自富士ACROS系列黑白胶片，
        为正全色黑白胶片，对蓝色最敏感，红色次
        之，绿色最弱，成片灰阶细腻，颗粒柔和，
        画面锐利，对光影有很好的还原力。
        
        LC400：灵感来自乐凯在21世纪初的“彩韵”
        系列彩色胶片，饱和度低，对比度低，带有
        轻微灰雾，旨在唤起对家中老照片的回忆。

        FS200：黑白正片⌈光⌋，在开发初期作为原理
        验证模型所使用，对蓝色较敏感，对红色较
        不敏感，对比鲜明，颗粒适中。
        '''
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
    (film,process_time,output_path) = process(uploaded_image,film_type)
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