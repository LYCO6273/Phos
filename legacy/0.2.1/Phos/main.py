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
"""

#这个python文件只是前端部分，实际上会调用子文件夹中各个胶片的单独模型

#赛博请神
import streamlit as st
import time
from PIL import Image
import io
import os

# 设置页面配置 
st.set_page_config(
    page_title="Phos. 胶片模拟",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 文件上传
uploaded_image = st.file_uploader(
    "选择一张照片来开始冲洗",
    type=["jpg", "jpeg", "png"],
    help="上传一张照片冲洗试试看吧"
)

# 创建侧边栏
with st.sidebar:
    st.header("Phos.")
    st.subheader("基于计算光学的胶片模拟")
    st.text("")
    st.text("原理验证demo")
    st.text("ver_0.2.1")
    st.text("")
    st.text("胶片设置")
    
    # 胶片类型选择
    film_type = st.selectbox(
        "胶片模拟模型:",
        ["FS200","NC200"],
        index=0,
        help='''选择胶片模拟配方:

        FS200：
        秽土转生的FS200（笑），这次的模型
        基于依尔福HP5黑白负片，均衡的全色
        黑白胶片，作为新的负片模型的实验品

        Gold 200：
        灵感来自富士C200彩色负片和扫描仪
        SP3000，旨在模仿经典的"富士色调"
        在真实数据的基础上融入了一些调整

        '''
    )
    
    # 胶片颗粒度选择
    grain_style = st.selectbox(
        "胶片颗粒度：",
        ["默认", "柔和", "较粗", "不使用"],
        index=0,
        help="选择胶片的颗粒度"
    )
    
    # 显示当前选择的胶片类型
    st.success(f"已选择胶片: {film_type}")

# 根据上传的图像和选择的胶片类型调用相应的处理函数
if uploaded_image is not None:
    
    start_time = time.time()
    
    try:
        file_extension = os.path.splitext(uploaded_image.name)[1]
        if file_extension.lower() in [".jpg", ".jpeg", ".png"]:
            import Helpers.Jpeg_reader as Jpeg_reader
            lux_r, lux_g, lux_b = Jpeg_reader.Jpeg_to_lux(uploaded_image)
        elif file_extension.lower() in [".dng"]:
            import Helpers.RAW_reader as RAW_reader
            lux_r, lux_g, lux_b = RAW_reader.RAW_to_lux(uploaded_image)
        else:
            st.error("不支持的文件格式，请上传JPEG或PNG图像。")
    except Exception as e:
        st.error(f"处理上传文件时出现错误: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("请尝试重新上传图像或检查文件格式。")

    try:
        if film_type == "FS200":
            import Films.FS200.FS200 as FS200
            film = FS200.process(lux_r, lux_g, lux_b, grain_style)
        elif film_type == "NC200":
            import Films.NC200.NC200 as NC200
            film = NC200.process(lux_r, lux_g, lux_b, grain_style)
    except Exception as e:
        st.error(f"处理图像时出现错误: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("请尝试调整参数或更换图像")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"phos_{timestamp}.jpg"
    process_time = time.time() - start_time

    if film is not None:
        try:
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
                label="下载高清图像",
                data=byte_im,
                file_name=output_path,
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"显示或下载图像时出现错误: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    uploaded_image = None

