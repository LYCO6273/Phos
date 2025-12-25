"""
"No LUTs, we calculate LUX."

你说的对，但是 Phos. 是基于「计算光学」概念的胶片模拟。
通过计算光在底片上的行为，复现自然、柔美、立体的胶片质感。

这是一个原理验证demo，图像处理部分基于opencv，交互基于
streamlit平台制作，部分代码使用了AI辅助生成。

如果您发现了项目中的问题，或是有更好的想法想要分享，还请
通过邮箱 lyco_p@163.com 与我联系，我将不胜感激。

Hello! Phos. is a film simulation app based on 
the idea of "Computational optical imaging". 
By calculating the optical effects on the film,
we could recurrent the natural, soft, and elegant
tone of these classical films.

This is a demo for idea testing. The image processing
part is based on OpenCV, and the interaction is built
on the Streamlit. Some pieces of the code was generated 
with the assistance of AI.

If you find any issues in the project or have better
ideas you would like to share, please contact me via
email at lyco_p@163.com. I would be very grateful.

——————————————————————————————————————————————————————

在0.1.4版本中，简化了算法，移除了Filmic和Reinhard映射，
引入基于对数的映射，并添加了自定义胶片参数功能。

In the update of version 0.1.3, we simplified the algorithms,
removed Filmic and Reinhard mapping, focusing on Log mapping,
and added custom film parameter function.
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

# 文件上传器放在最前面，方便后续处理
uploaded_image = st.file_uploader(
    "选择一张照片来开始冲洗",
    type=["jpg", "jpeg", "png"],
    help="上传一张照片冲洗试试看吧"
)

def film_choose(film_type):
    """获取胶片参数 - 修复版本，只处理预设胶片"""
    if film_type == "NC200":
        return (
            0.77, 0.12, 0.18,  # r_r, r_g, r_b
            0.08, 0.85, 0.23,  # g_r, g_g, g_b
            0.08, 0.09, 0.92,  # b_r, b_g, b_b
            0.25, 0.35, 0.35,  # t_r, t_g, t_b
            "color", 1.20,     # color_type, sens_factor
            1.48, 0.95, 1.18, 0.18,  # d_r, l_r, x_r, n_r
            1.02, 0.80, 1.02, 0.18,  # d_g, l_g, x_g, n_g
            1.02, 0.88, 0.78, 0.18,  # d_b, l_b, x_b, n_b
            None, None, None, 0.08,  # d_l, l_l, x_l, n_l
            1.10, 0.95               # gam_for_log, exp_for_log
        )
    elif film_type == "FS200":
        return (
            0, 0, 0,            # r_r, r_g, r_b
            0, 0, 0,            # g_r, g_g, g_b
            0, 0, 0,            # b_r, b_g, b_b
            0.15, 0.35, 0.45,   # t_r, t_g, t_b
            "single", 1.0,      # color_type, sens_factor
            0, 0, 0, 0,         # d_r, l_r, x_r, n_r
            0, 0, 0, 0,         # d_g, l_g, x_g, n_g
            0, 0, 0, 0,         # d_b, l_b, x_b, n_b
            1.85, 0.75, 1.35, 0.18,  # d_l, l_l, x_l, n_l
            1.35, 1.15               # gam_for_log, exp_for_log
        )
    elif film_type == "AS100":
        return (
            0, 0, 0,            # r_r, r_g, r_b
            0, 0, 0,            # g_r, g_g, g_b
            0, 0, 0,            # b_r, b_g, b_b
            0.30, 0.12, 0.45,   # t_r, t_g, t_b
            "single", 1.28,     # color_type, sens_factor
            0, 0, 0, 0,         # d_r, l_r, x_r, n_r
            0, 0, 0, 0,         # d_g, l_g, x_g, n_g
            0, 0, 0, 0,         # d_b, l_b, x_b, n_b
            1.0, 1.05, 1.25, 0.10,  # d_l, l_l, x_l, n_l
            1.05, 1.15               # gam_for_log, exp_for_log
        )
    else:
        # 默认返回AS100参数
        return (
            0, 0, 0,            # r_r, r_g, r_b
            0, 0, 0,            # g_r, g_g, g_b
            0, 0, 0,            # b_r, b_g, b_b
            0.30, 0.12, 0.45,   # t_r, t_g, t_b
            "single", 1.28,     # color_type, sens_factor
            0, 0, 0, 0,         # d_r, l_r, x_r, n_r
            0, 0, 0, 0,         # d_g, l_g, x_g, n_g
            0, 0, 0, 0,         # d_b, l_b, x_b, n_b
            1.0, 1.05, 1.25, 0.10,  # d_l, l_l, x_l, n_l
            1.05, 1.15               # gam_for_log, exp_for_log
        )

def get_custom_params():
    """从UI获取自定义参数 - 修复版本"""
    custom_params = {}
    
    # 第一步：选择颜色类型
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 自定义胶片参数")
    
    color_type = st.sidebar.selectbox(
        "颜色类型:",
        ["single", "color"],
        index=0,
        help="选择胶片类型：single为黑白，color为彩色"
    )
    custom_params["color_type"] = color_type
    
    # 高光敏感系数
    custom_params["sens_factor"] = st.sidebar.slider(
        "高光敏感系数",
        min_value=0.5, max_value=2.0, value=1.28, step=0.01,
        help="控制高光区域的敏感度"
    )
    
    # Log映射参数
    custom_params["gam_for_log"] = st.sidebar.slider(
        "曲线gamma",
        min_value=0.5, max_value=2.5, value=1.05, step=0.01,
        help="控制曲线的形状"
    )
    
    custom_params["exp_for_log"] = st.sidebar.slider(
        "曝光补偿",
        min_value=0.5, max_value=2.0, value=1.00, step=0.01,
        help="调整整体曝光"
    )
    
    # 全色感光层吸收特性（黑白和彩色都需要）
    custom_params["t_r"] = st.sidebar.slider(
        "全色层吸收红光", 0.0, 1.0, 0.30, 0.01,
        help="全色感光层对红光的吸收比例"
    )
    custom_params["t_g"] = st.sidebar.slider(
        "全色层吸收绿光", 0.0, 1.0, 0.12, 0.01,
        help="全色感光层对绿光的吸收比例"
    )
    custom_params["t_b"] = st.sidebar.slider(
        "全色层吸收蓝光", 0.0, 1.0, 0.45, 0.01,
        help="全色感光层对蓝光的吸收比例"
    )
    
    # 如果是彩色胶片，显示彩色层参数
    if color_type == "color":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔴 红色感光层")
        custom_params["r_r"] = st.sidebar.slider("红层吸收红光", 0.0, 1.0, 0.77, 0.01)
        custom_params["r_g"] = st.sidebar.slider("红层吸收绿光", 0.0, 1.0, 0.12, 0.01)
        custom_params["r_b"] = st.sidebar.slider("红层吸收蓝光", 0.0, 1.0, 0.18, 0.01)
        
        st.sidebar.subheader("🟢 绿色感光层")
        custom_params["g_r"] = st.sidebar.slider("绿层吸收红光", 0.0, 1.0, 0.08, 0.01)
        custom_params["g_g"] = st.sidebar.slider("绿层吸收绿光", 0.0, 1.0, 0.85, 0.01)
        custom_params["g_b"] = st.sidebar.slider("绿层吸收蓝光", 0.0, 1.0, 0.23, 0.01)
        
        st.sidebar.subheader("🔵 蓝色感光层")
        custom_params["b_r"] = st.sidebar.slider("蓝层吸收红光", 0.0, 1.0, 0.08, 0.01)
        custom_params["b_g"] = st.sidebar.slider("蓝层吸收绿光", 0.0, 1.0, 0.09, 0.01)
        custom_params["b_b"] = st.sidebar.slider("蓝层吸收蓝光", 0.0, 1.0, 0.92, 0.01)
    else:
        # 黑白胶片，彩色层参数设为0
        custom_params["r_r"] = 0.0
        custom_params["r_g"] = 0.0
        custom_params["r_b"] = 0.0
        custom_params["g_r"] = 0.0
        custom_params["g_g"] = 0.0
        custom_params["g_b"] = 0.0
        custom_params["b_r"] = 0.0
        custom_params["b_g"] = 0.0
        custom_params["b_b"] = 0.0
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 光学响应参数")
    
    if color_type == "color":
        # 彩色胶片的各层光学响应
        st.sidebar.markdown("#### 红色感光层")
        custom_params["d_r"] = st.sidebar.slider("红层散射光", 0.0, 3.0, 1.48, 0.01)
        custom_params["l_r"] = st.sidebar.slider("红层直射光", 0.0, 2.0, 0.95, 0.01)
        custom_params["x_r"] = st.sidebar.slider("红层响应系数", 0.5, 2.0, 1.18, 0.01)
        custom_params["n_r"] = st.sidebar.slider("红层颗粒度", 0.0, 1.0, 0.18, 0.01)
        
        st.sidebar.markdown("#### 绿色感光层")
        custom_params["d_g"] = st.sidebar.slider("绿层散射光", 0.0, 3.0, 1.02, 0.01)
        custom_params["l_g"] = st.sidebar.slider("绿层直射光", 0.0, 2.0, 0.80, 0.01)
        custom_params["x_g"] = st.sidebar.slider("绿层响应系数", 0.5, 2.0, 1.02, 0.01)
        custom_params["n_g"] = st.sidebar.slider("绿层颗粒度", 0.0, 1.0, 0.18, 0.01)
        
        st.sidebar.markdown("#### 蓝色感光层")
        custom_params["d_b"] = st.sidebar.slider("蓝层散射光", 0.0, 3.0, 1.02, 0.01)
        custom_params["l_b"] = st.sidebar.slider("蓝层直射光", 0.0, 2.0, 0.88, 0.01)
        custom_params["x_b"] = st.sidebar.slider("蓝层响应系数", 0.5, 2.0, 0.78, 0.01)
        custom_params["n_b"] = st.sidebar.slider("蓝层颗粒度", 0.0, 1.0, 0.18, 0.01)
        
        # 彩色胶片的全色层参数设为None
        custom_params["d_l"] = None
        custom_params["l_l"] = None
        custom_params["x_l"] = None
        custom_params["n_l"] = 0.08  # 基础颗粒度
    else:
        # 黑白胶片的彩色层参数设为0
        custom_params["d_r"] = 0.0
        custom_params["l_r"] = 0.0
        custom_params["x_r"] = 0.0
        custom_params["n_r"] = 0.0
        custom_params["d_g"] = 0.0
        custom_params["l_g"] = 0.0
        custom_params["x_g"] = 0.0
        custom_params["n_g"] = 0.0
        custom_params["d_b"] = 0.0
        custom_params["l_b"] = 0.0
        custom_params["x_b"] = 0.0
        custom_params["n_b"] = 0.0
        
        # 黑白胶片的光学响应（全色层）
        st.sidebar.markdown("#### 全色感光层")
        custom_params["d_l"] = st.sidebar.slider("全色层散射光", 0.0, 3.0, 1.0, 0.01)
        custom_params["l_l"] = st.sidebar.slider("全色层直射光", 0.0, 2.0, 1.05, 0.01)
        custom_params["x_l"] = st.sidebar.slider("全色层响应系数", 0.5, 2.0, 1.25, 0.01)
        custom_params["n_l"] = st.sidebar.slider("全色层颗粒度", 0.0, 1.0, 0.10, 0.01)
    
    return custom_params

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

def luminance(image, color_type, r_r, r_g, r_b, g_r, g_g, g_b, b_r, b_g, b_b, t_r, t_g, t_b):
    """计算亮度图像 (0-1范围)"""
    # 分离RGB通道
    b, g, r = cv2.split(image)
    
    # 转换为浮点数
    b_float = b.astype(np.float32) / 255.0
    g_float = g.astype(np.float32) / 255.0
    r_float = r.astype(np.float32) / 255.0
    
    # 模拟不同乳剂层的吸收特性
    if color_type == "color":
        lux_r = r_r * r_float + r_g * g_float + r_b * b_float
        lux_g = g_r * r_float + g_g * g_float + g_b * b_float
        lux_b = b_r * r_float + b_g * g_float + b_b * b_float
        lux_total = t_r * r_float + t_g * g_float + t_b * b_float
    else:
        lux_total = t_r * r_float + t_g * g_float + t_b * b_float
        lux_r = None
        lux_g = None
        lux_b = None

    return lux_r, lux_g, lux_b, lux_total

def average(lux_total):
    """计算图像的平均亮度 (0-1)"""
    # 计算平均亮度
    avg_lux = np.mean(lux_total)
    avg_lux = np.clip(avg_lux, 0, 1)
    return avg_lux

def grain(lux_r, lux_g, lux_b, lux_total, color_type, sens):
    """基于加权随机的颗粒模拟"""
    if color_type == "color":
        # 红色通道颗粒
        noise = np.random.normal(0, 1, lux_r.shape).astype(np.float32)
        noise = noise ** 2
        noise = noise * (np.random.choice([-1, 1], lux_r.shape))
        weights = (0.5 - np.abs(lux_r - 0.5)) * 2
        weights = np.clip(weights, 0.05, 0.9)
        sens_grain = np.clip(sens, 0.4, 0.6)
        weighted_noise = noise * weights * sens_grain
        noise = None
        weights = None
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_r = np.clip(weighted_noise, -1, 1)
        weighted_noise = None

        # 绿色通道颗粒
        noise = np.random.normal(0, 1, lux_g.shape).astype(np.float32)
        noise = noise ** 2
        noise = noise * (np.random.choice([-1, 1], lux_g.shape))
        weights = (0.5 - np.abs(lux_g - 0.5)) * 2
        weights = np.clip(weights, 0.05, 0.9)
        sens_grain = np.clip(sens, 0.4, 0.6)
        weighted_noise = noise * weights * sens_grain
        noise = None
        weights = None
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_g = np.clip(weighted_noise, -1, 1)
        weighted_noise = None

        # 蓝色通道颗粒
        noise = np.random.normal(0, 1, lux_b.shape).astype(np.float32)
        noise = noise ** 2
        noise = noise * (np.random.choice([-1, 1], lux_b.shape))
        weights = (0.5 - np.abs(lux_b - 0.5)) * 2
        weights = np.clip(weights, 0.05, 0.9)
        sens_grain = np.clip(sens, 0.4, 0.6)
        weighted_noise = noise * weights * sens_grain
        noise = None
        weights = None
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_b = np.clip(weighted_noise, -1, 1)
        weighted_noise = None
        weighted_noise_total = None
        
    else:
        # 黑白胶片颗粒
        noise = np.random.normal(0, 1, lux_total.shape).astype(np.float32)
        noise = noise ** 2
        noise = noise * (np.random.choice([-1, 1], lux_total.shape))
        weights = (0.5 - np.abs(lux_total - 0.5)) * 2
        weights = np.clip(weights, 0.05, 0.9)
        sens_grain = np.clip(sens, 0.4, 0.6)
        weighted_noise = noise * weights * sens_grain
        noise = None
        weights = None
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        weighted_noise_total = np.clip(weighted_noise, -1, 1)
        weighted_noise = None
        weighted_noise_r = None
        weighted_noise_g = None
        weighted_noise_b = None
    
    return weighted_noise_r, weighted_noise_g, weighted_noise_b, weighted_noise_total

def log_tone(lux_r, lux_g, lux_b, lux_total, color_type, gam_for_log, exp_for_log):
    """定义log tone mapping算法"""
    if color_type == "color":
        lux_r = np.maximum(lux_r, 0)
        lux_g = np.maximum(lux_g, 0)
        lux_b = np.maximum(lux_b, 0)

        result_r = np.log(((lux_r * exp_for_log) ** gam_for_log) + 1.000001)
        result_r = np.clip(result_r, 0, 1)

        result_g = np.log(((lux_g * exp_for_log) ** gam_for_log) + 1.000001)
        result_g = np.clip(result_g, 0, 1)

        result_b = np.log(((lux_b * exp_for_log) ** gam_for_log) + 1.000001)
        result_b = np.clip(result_b, 0, 1)
        result_total = None
    else:
        lux_total = np.maximum(lux_total, 0)
        result_total = np.log(((lux_total * exp_for_log) ** gam_for_log) + 1.000001)
        result_total = np.clip(result_total, 0, 1)
        result_r = None
        result_g = None
        result_b = None

    return result_r, result_g, result_b, result_total

def opt(lux_r, lux_g, lux_b, lux_total, color_type, sens_factor, 
        d_r, l_r, x_r, n_r, d_g, l_g, x_g, n_g, d_b, l_b, x_b, n_b, 
        d_l, l_l, x_l, n_l, grain_style, gam_for_log, exp_for_log):
    """光学扩散函数"""
    
    avrl = average(lux_total)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.75 + 0.10
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens, 0.10, 0.7)
    strg = 23 * sens**2 * sens_factor
    rads = np.clip(int(20 * sens**2 * sens_factor), 1, 50)
    base = 0.05 * sens_factor

    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    if color_type == "color":
        # 红色通道散射
        weights = (base + lux_r**2) * sens 
        weights = np.clip(weights, 0, 1)
        bloom_layer = cv2.GaussianBlur(lux_r * weights, (ksize * 3, ksize * 3), sens * 55)
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect / (1.0 + bloom_effect))
        bloom_effect_r = bloom_effect
        bloom_effect = None
        weights = None
        bloom_layer = None

        # 绿色通道散射
        weights = (base + lux_g**2) * sens
        weights = np.clip(weights, 0, 1)
        bloom_layer = cv2.GaussianBlur(lux_g * weights, (ksize * 2 + 1, ksize * 2 + 1), sens * 35)
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect / (1.0 + bloom_effect))
        bloom_effect_g = bloom_effect
        bloom_effect = None
        weights = None
        bloom_layer = None
    
        # 蓝色通道散射
        weights = (base + lux_b**2) * sens
        weights = np.clip(weights, 0, 1)
        bloom_layer = cv2.GaussianBlur(lux_b * weights, (ksize, ksize), sens * 15)
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect / (1.0 + bloom_effect))
        bloom_effect_b = bloom_effect
        bloom_effect = None
        weights = None
        bloom_layer = None

        # 应用颗粒
        if grain_style == "不使用":
            lux_r = bloom_effect_r * d_r + (lux_r**x_r) * l_r
            lux_g = bloom_effect_g * d_g + (lux_g**x_g) * l_g
            lux_b = bloom_effect_b * d_b + (lux_b**x_b) * l_b
        else:    
            (weighted_noise_r, weighted_noise_g, weighted_noise_b, weighted_noise_total) = grain(
                lux_r, lux_g, lux_b, lux_total, color_type, sens
            )
            # 应用颗粒
            lux_r = bloom_effect_r * d_r + (lux_r**x_r) * l_r + weighted_noise_r * n_r + weighted_noise_g * n_l + weighted_noise_b * n_l
            lux_g = bloom_effect_g * d_g + (lux_g**x_g) * l_g + weighted_noise_r * n_l + weighted_noise_g * n_g + weighted_noise_b * n_l
            lux_b = bloom_effect_b * d_b + (lux_b**x_b) * l_b + weighted_noise_r * n_l + weighted_noise_g * n_l + weighted_noise_b * n_b
        
        bloom_effect_r = None
        bloom_effect_g = None
        bloom_effect_b = None
        weighted_noise_r = None
        weighted_noise_g = None
        weighted_noise_b = None

        # 应用tone mapping
        (result_r, result_g, result_b, result_total) = log_tone(
            lux_r, lux_g, lux_b, lux_total, color_type, gam_for_log, exp_for_log
        )

        lux_r = None
        lux_g = None
        lux_b = None

        # 合并通道
        result_b = (result_b * 255).astype(np.uint8)
        result_g = (result_g * 255).astype(np.uint8)
        result_r = (result_r * 255).astype(np.uint8)
        film = cv2.merge([result_r, result_g, result_b])
        result_r = None
        result_g = None
        result_b = None

    else:
        # 黑白胶片处理
        weights = (base + lux_total**2) * sens 
        weights = np.clip(weights, 0, 1)
        bloom_layer = cv2.GaussianBlur(lux_total * weights, (ksize * 3, ksize * 3), sens * 55)
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect / (1.0 + bloom_effect))
        weights = None
        bloom_layer = None

        if grain_style == "不使用":
            lux_total = bloom_effect * d_l + (lux_total**x_l) * l_l
        else:
            (weighted_noise_r, weighted_noise_g, weighted_noise_b, weighted_noise_total) = grain(
                lux_r, lux_g, lux_b, lux_total, color_type, sens
            )
            lux_total = bloom_effect * d_l + (lux_total**x_l) * l_l + weighted_noise_total * n_l
        
        bloom_effect = None
        weighted_noise_total = None

        # 应用log色调映射
        (result_r, result_g, result_b, result_total) = log_tone(
            lux_r, lux_g, lux_b, lux_total, color_type, gam_for_log, exp_for_log
        )

        lux_total = None
        film = (result_total * 255).astype(np.uint8)
        lux_total = None

    return film

def process(uploaded_image, film_type, grain_style, custom_params=None):
    """主处理函数 - 修复版本"""
    start_time = time.time()

    # 读取上传的文件
    image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    uploaded_image = None

    # 获取胶片参数
    if film_type == "自定义" and custom_params is not None:
        # 使用自定义参数 - 直接从字典中获取
        color_type = custom_params["color_type"]
        sens_factor = float(custom_params["sens_factor"])
        gam_for_log = float(custom_params["gam_for_log"])
        exp_for_log = float(custom_params["exp_for_log"])
        
        # 吸收特性
        r_r = float(custom_params["r_r"])
        r_g = float(custom_params["r_g"])
        r_b = float(custom_params["r_b"])
        g_r = float(custom_params["g_r"])
        g_g = float(custom_params["g_g"])
        g_b = float(custom_params["g_b"])
        b_r = float(custom_params["b_r"])
        b_g = float(custom_params["b_g"])
        b_b = float(custom_params["b_b"])
        t_r = float(custom_params["t_r"])
        t_g = float(custom_params["t_g"])
        t_b = float(custom_params["t_b"])
        
        # 光学响应参数 - 确保所有参数都是浮点数
        d_r = float(custom_params["d_r"]) if custom_params["d_r"] is not None else 0.0
        l_r = float(custom_params["l_r"]) if custom_params["l_r"] is not None else 0.0
        x_r = float(custom_params["x_r"]) if custom_params["x_r"] is not None else 0.0
        n_r = float(custom_params["n_r"]) if custom_params["n_r"] is not None else 0.0
        
        d_g = float(custom_params["d_g"]) if custom_params["d_g"] is not None else 0.0
        l_g = float(custom_params["l_g"]) if custom_params["l_g"] is not None else 0.0
        x_g = float(custom_params["x_g"]) if custom_params["x_g"] is not None else 0.0
        n_g = float(custom_params["n_g"]) if custom_params["n_g"] is not None else 0.0
        
        d_b = float(custom_params["d_b"]) if custom_params["d_b"] is not None else 0.0
        l_b = float(custom_params["l_b"]) if custom_params["l_b"] is not None else 0.0
        x_b = float(custom_params["x_b"]) if custom_params["x_b"] is not None else 0.0
        n_b = float(custom_params["n_b"]) if custom_params["n_b"] is not None else 0.0
        
        d_l = float(custom_params["d_l"]) if custom_params["d_l"] is not None else 0.0
        l_l = float(custom_params["l_l"]) if custom_params["l_l"] is not None else 0.0
        x_l = float(custom_params["x_l"]) if custom_params["x_l"] is not None else 0.0
        n_l = float(custom_params["n_l"]) if custom_params["n_l"] is not None else 0.0
        
    else:
        # 使用预设参数
        (r_r, r_g, r_b, 
         g_r, g_g, g_b, 
         b_r, b_g, b_b, 
         t_r, t_g, t_b, 
         color_type, sens_factor, 
         d_r, l_r, x_r, n_r, 
         d_g, l_g, x_g, n_g, 
         d_b, l_b, x_b, n_b, 
         d_l, l_l, x_l, n_l, 
         gam_for_log, exp_for_log) = film_choose(film_type)
    
    # 调整颗粒度 - 安全处理None值
    def safe_multiply(param, factor):
        if param is None:
            return None
        return param * factor
    
    if grain_style == "默认":
        n_r = safe_multiply(n_r, 1.0)
        n_g = safe_multiply(n_g, 1.0)
        n_b = safe_multiply(n_b, 1.0)
        n_l = safe_multiply(n_l, 1.0)
    elif grain_style == "柔和":
        n_r = safe_multiply(n_r, 0.5)
        n_g = safe_multiply(n_g, 0.5)
        n_b = safe_multiply(n_b, 0.5)
        n_l = safe_multiply(n_l, 0.5)
    elif grain_style == "较粗":
        n_r = safe_multiply(n_r, 1.5)
        n_g = safe_multiply(n_g, 1.5)
        n_b = safe_multiply(n_b, 1.5)
        n_l = safe_multiply(n_l, 1.5)
    elif grain_style == "不使用":
        n_r = safe_multiply(n_r, 0)
        n_g = safe_multiply(n_g, 0)
        n_b = safe_multiply(n_b, 0)
        n_l = safe_multiply(n_l, 0)

    # 调整尺寸
    image = standardize(image)

    # 重建光线
    (lux_r, lux_g, lux_b, lux_total) = luminance(
        image, color_type, r_r, r_g, r_b, g_r, g_g, g_b, 
        b_r, b_g, b_b, t_r, t_g, t_b
    )
    
    # 冲洗底片
    film = opt(lux_r, lux_g, lux_b, lux_total, color_type, sens_factor, 
               d_r, l_r, x_r, n_r, d_g, l_g, x_g, n_g, 
               d_b, l_b, x_b, n_b, d_l, l_l, x_l, n_l,
               grain_style, gam_for_log, exp_for_log)
    
    # 生成输出文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"phos_{timestamp}.jpg"
    process_time = time.time() - start_time

    return film, process_time, output_path

# 创建侧边栏
with st.sidebar:
    st.header("Phos.")
    st.subheader("基于计算光学的胶片模拟")
    st.text("")
    st.text("原理验证demo")
    st.text("ver_0.1.4")
    st.text("")
    st.text("🎞️ 胶片设置")
    
    # 胶片类型选择
    film_type = st.selectbox(
        "胶片模拟配方:",
        ["NC200", "AS100", "FS200", "自定义"],
        index=0,
        help='''选择胶片模拟配方:

        NC200: 灵感来自富士C200彩色负片和扫描仪SP3000，
               旨在模仿经典的"富士色调"

        AS100：灵感来自富士ACROS系列黑白胶片，
               为正全色黑白胶片，对蓝色最敏感

        FS200：高对比度黑白正片⌈光⌋，对蓝色较敏感，
               对红色较不敏感，对比鲜明，颗粒适中

        自定义：自由调整各项参数，创造你的胶片配方
        '''
    )
    
    # 如果选择自定义，获取自定义参数
    custom_params = None
    if film_type == "自定义":
        custom_params = get_custom_params()
    
    # 胶片颗粒度选择
    grain_style = st.selectbox(
        "胶片颗粒度：",
        ["默认", "柔和", "较粗", "不使用"],
        index=0,
        help="选择胶片的颗粒度"
    )
    
    # 显示当前选择的胶片类型
    if film_type != "自定义":
        st.success(f"已选择胶片: {film_type}")
    else:
        st.success("已选择自定义胶片配方")

# 主处理流程
if uploaded_image is not None:
    try:
        # 处理图像
        (film, process_time, output_path) = process(
            uploaded_image, film_type, grain_style, custom_params
        )
        
        # 显示结果
        st.image(film, width="stretch")
        st.success(f"底片显影好了，用时 {process_time:.2f}秒") 
        
        # 添加下载按钮
        film_pil = Image.fromarray(film)
        buf = io.BytesIO()
        film_pil.save(buf, format="JPEG", quality=100)
        byte_im = buf.getvalue()
        
        buf = io.BytesIO()
        film_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="📥 下载高清图像",
            data=byte_im,
            file_name=output_path,
            mime="image/jpeg"
        )
        
    except Exception as e:
        st.error(f"处理图像时出现错误: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("请尝试调整参数或更换图像")
    
    uploaded_image = None