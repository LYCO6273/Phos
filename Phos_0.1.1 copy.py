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
on the Streamlit. Some pieces of the code was generated
with the assistance of AI.

If you find any issues in the project or have better
ideas you would like to share, please contact me via
email at lyco_p@163.com. I would be very grateful.
"""

from __future__ import annotations

import os
import time

import streamlit as st

from phos.presets import FILM_DESCRIPTIONS, FILM_TYPES
from phos.processing import RAW_EXTENSIONS, ProcessingOptions, make_zip_bytes, process_uploaded_file


st.set_page_config(
    page_title="Phos. 胶片模拟",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _uploader_types() -> list[str]:
    base = ["jpg", "jpeg", "png", "tif", "tiff"]
    return base + sorted(RAW_EXTENSIONS)


        lux_r = 10 * (lux_r ** gamma)
        lux_g = 10 * (lux_g ** gamma)
        lux_b = 10 * (lux_b ** gamma)

        result_r = ((lux_r * (A * lux_r + C * B) + D * E) / (lux_r * (A * lux_r + B) + D * F)) - E/F
        result_g = ((lux_g * (A * lux_g + C * B) + D * E) / (lux_g * (A * lux_g + B) + D * F)) - E/F
        result_b = ((lux_b * (A * lux_b + C * B) + D * E) / (lux_b * (A * lux_b + B) + D * F)) - E/F
        result_total = None
    else:
        lux_total = np.maximum(lux_total, 0)
        lux_total = 10 * (lux_total ** gamma)
        result_r = None
        result_g = None
        result_b = None
        result_total = ((lux_total * (A * lux_total + C * B) + D * E) / (lux_total * (A * lux_total + B) + D * F)) - E/F
    
    return result_r,result_g,result_b,result_total

def opt(lux_r,lux_g,lux_b,lux_total,color_type, sens_factor, d_r, l_r, x_r, n_r, d_g, l_g, x_g, n_g, d_b, l_b, x_b, n_b, d_l, l_l, x_l, n_l,grain_style,gamma,A,B,C,D,E,F,Tone_style):
    #opt 光学扩散函数

    avrl = average(lux_total)
    # 根据平均亮度计算敏感度
    sens = (1.0 - avrl) * 0.75 + 0.10
    # 将敏感度限制在0-1范围内
    sens = np.clip(sens,0.10,0.7) #sens -- 高光敏感度
    strg = 23 * sens**2 * sens_factor #strg -- 光晕强度
    rads = np.clip(int(20 * sens**2 * sens_factor),1,50) #rads -- 光晕扩散半径
    base = 0.05 * sens_factor #base -- 基础扩散强度

    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    # 确保核大小为奇数

    if color_type == ("color"):
        weights = (base + lux_r**2) * sens 
        weights = np.clip(weights,0,1)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_r * weights, (ksize * 3 , ksize * 3),sens * 55)
        #开始高斯模糊
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_r = bloom_effect
        #应用光晕
    
        weights = (base + lux_g**2 ) * sens
        weights = np.clip(weights,0,1)
        bloom_layer = cv2.GaussianBlur(lux_g * weights, (ksize * 2 +1 , ksize * 2 +1 ),sens * 35)
        #开始高斯模糊
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_g = bloom_effect
        #应用光晕
    
        weights = (base + lux_b**2 ) * sens
        weights = np.clip(weights,0,1)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_b * weights, (ksize, ksize),sens * 15)
        #开始高斯模糊
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        bloom_effect_b = bloom_effect
        #应用光晕
        
        if grain_style == ("不使用"):
            lux_r = bloom_effect_r * d_r + (lux_r**x_r) * l_r
            lux_g = bloom_effect_g * d_g + (lux_g**x_g) * l_g
            lux_b = bloom_effect_b * d_b + (lux_b**x_b) * l_b
        else:    
            (weighted_noise_r,weighted_noise_g,weighted_noise_b,weighted_noise_total) = grain(lux_r,lux_g,lux_b,lux_total,color_type,sens)
            #应用颗粒
            lux_r = bloom_effect_r * d_r + (lux_r**x_r) * l_r + weighted_noise_r *n_r + weighted_noise_g *n_l+ weighted_noise_b *n_l
            lux_g = bloom_effect_g * d_g + (lux_g**x_g) * l_g + weighted_noise_r *n_l + weighted_noise_g *n_g+ weighted_noise_b *n_l
            lux_b = bloom_effect_b * d_b + (lux_b**x_b) * l_b + weighted_noise_r *n_l + weighted_noise_g *n_l + weighted_noise_b *n_b
        
        #拼合光层
        if Tone_style == "filmic":
            (result_r,result_g,result_b,result_total) = filmic(lux_r,lux_g,lux_b,lux_total,color_type,gamma,A,B,C,D,E,F)
            #应用flimic映射
        else:
            (result_r,result_g,result_b,result_total) = reinhard(lux_r,lux_g,lux_b,lux_total,color_type,gamma)
            #应用映射

        combined_b = (result_b * 255).astype(np.uint8)
        combined_g = (result_g * 255).astype(np.uint8)
        combined_r = (result_r * 255).astype(np.uint8)
        film = cv2.merge([combined_r, combined_g, combined_b])
    else:
        weights = (base + lux_total**2) * sens 
        weights = np.clip(weights,0,1)
        #创建光晕层
        bloom_layer = cv2.GaussianBlur(lux_total * weights, (ksize * 3 , ksize * 3),sens * 55)
        #开始高斯模糊
        bloom_effect = bloom_layer * weights * strg
        bloom_effect = (bloom_effect/ (1.0 + bloom_effect))
        #应用光晕
        if grain_style == ("不使用"):
            lux_total = bloom_effect * d_l + (lux_total**x_l) * l_l
        else:
            (weighted_noise_r,weighted_noise_g,weighted_noise_b,weighted_noise_total) = grain(lux_r,lux_g,lux_b,lux_total,color_type,sens)
            #应用颗粒
            lux_total = bloom_effect * d_l + (lux_total**x_l) * l_l + weighted_noise_total *n_l
        
        #拼合光层
        
        if Tone_style == "filmic":
            (result_r,result_g,result_b,result_total) = filmic(lux_r,lux_g,lux_b,lux_total,color_type,gamma,A,B,C,D,E,F)
            #应用flimic映射
        else:
            (result_r,result_g,result_b,result_total) = reinhard(lux_r,lux_g,lux_b,lux_total,color_type,gamma)
            #应用reinhard映射

        film = (result_total * 255).astype(np.uint8)

    return film
    #返回渲染后的光度
    #进行底片成像
    #准备暗房工具

def process(uploaded_image,film_type,grain_style,Tone_style):
    
    start_time = time.time()

    # 读取上传的文件
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # 获取胶片参数
    (r_r,r_g,r_b,g_r,g_g,g_b,b_r,b_g,b_b,t_r,t_g,t_b,color_type,sens_factor,d_r,l_r,x_r,n_r,d_g,l_g,x_g,n_g,d_b,l_b,x_b,n_b,d_l,l_l,x_l,n_l,gamma,A,B,C,D,E,F) = film_choose(film_type)
    
    if grain_style == ("默认"):
        n_r = n_r * 1.0
        n_g = n_g * 1.0
        n_b = n_b * 1.0
        n_l = n_l * 1.0
    elif grain_style == ("柔和"):
        n_r = n_r * 0.5
        n_g = n_g * 0.5
        n_b = n_b * 0.5
        n_l = n_l * 0.5
    elif grain_style == ("较粗"):
        n_r = n_r * 1.5
        n_g = n_g * 1.5
        n_b = n_b * 1.5
        n_l = n_l * 1.5
    elif grain_style == ("不使用"):
        n_r = n_r * 0
        n_g = n_g * 0
        n_b = n_b * 0
        n_l = n_l * 0


    # 调整尺寸
    image = standardize(image)

    (lux_r,lux_g,lux_b,lux_total) = luminance(image,color_type,r_r,r_g,r_b,g_r,g_g,g_b,b_r,b_g,b_b,t_r,t_g,t_b)
    #重建光线
    film = opt(lux_r,lux_g,lux_b,lux_total,color_type, sens_factor, d_r, l_r, x_r, n_r, d_g, l_g, x_g, n_g, d_b, l_b, x_b, n_b, d_l, l_l, x_l, n_l,grain_style,gamma,A,B,C,D,E,F,Tone_style)
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
    st.text("原理验证demo")
    st.text("ver_0.1.1")
    st.text("")
    st.text("🎞️ 胶片设置")

    default_film_index = FILM_TYPES.index("FUJI200") if "FUJI200" in FILM_TYPES else 0
    film_type = st.selectbox(
        "请选择胶片:",
        FILM_TYPES,
        index=default_film_index,
        help="\n\n".join(FILM_DESCRIPTIONS.get(t, t) for t in FILM_TYPES),
    )

    tone_style = st.selectbox(
        "曲线映射：",
        ["filmic", "reinhard"],
        index=0,
        help="filmic 更像胶片肩部/趾部；reinhard 动态范围更直接。",
    )

    st.divider()
    st.text("🌾 颗粒")
    grain_enabled = st.checkbox("启用胶片颗粒", value=True)
    grain_strength = st.slider("颗粒强度", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    grain_size = st.slider("颗粒粗细", min_value=0.4, max_value=3.0, value=1.0, step=0.05)

    st.divider()
    uploaded_files = st.file_uploader(
        "选择照片来开始冲洗（支持批处理 / RAW）",
        type=_uploader_types(),
        accept_multiple_files=True,
        help="可一次上传多张；RAW 需要可选安装 rawpy/libraw（requirements-raw.txt）。",
    )

run = st.button("开始冲洗", type="primary", disabled=not uploaded_files)


if run and uploaded_files:
    options = ProcessingOptions(
        film_type=film_type,
        tone_style=tone_style,
        grain_enabled=grain_enabled,
        grain_strength=grain_strength,
        grain_size=grain_size,
        jpeg_quality=100,
    )

    named_outputs: list[tuple[str, bytes]] = []
    progress = st.progress(0.0)
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            result = process_uploaded_file(uploaded_file, options=options)
        except Exception as exc:
            st.error(f"处理失败：{getattr(uploaded_file, 'name', 'unknown')} - {exc}")
            progress.progress(idx / total)
            continue

        st.image(result.film_rgb, use_container_width=True)
        st.caption(f"{os.path.basename(result.output_filename)}（{result.process_time_s:.2f}s）")
        named_outputs.append((os.path.basename(result.output_filename), result.jpeg_bytes))

        st.download_button(
            label=f"📥 下载 {os.path.basename(result.output_filename)}",
            data=result.jpeg_bytes,
            file_name=os.path.basename(result.output_filename),
            mime="image/jpeg",
            key=f"dl_{idx}_{os.path.basename(result.output_filename)}_{time.time_ns()}",
        )
        progress.progress(idx / total)

    if len(named_outputs) >= 2:
        zip_name = f"phos_batch_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        zip_bytes = make_zip_bytes(named_outputs)
        st.download_button(
            label="📦 打包下载（ZIP）",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
        )
