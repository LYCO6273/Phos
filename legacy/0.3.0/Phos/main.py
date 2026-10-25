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
import importlib


class _NamedBytes(io.BytesIO):
    """给内存字节流补一个 .name，兼容原 Helpers 的读取接口。"""
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def run_film(file_bytes, file_name, film_type, grain_style,
             use_spectral_input, psf_mode, grain_ss, exposure_offset,
             gamma, clip_mode, wb_strength, wb_gray, s_strength, slope):
    """读图 + 胶片处理（每次调用都从头计算，不缓存）。"""
    uploaded = _NamedBytes(file_bytes, file_name)
    file_extension = os.path.splitext(file_name)[1]
    if file_extension.lower() in [".jpg", ".jpeg", ".png"]:
        import Helpers.Jpeg_reader as Jpeg_reader
        lux_r, lux_g, lux_b = Jpeg_reader.Jpeg_to_lux(uploaded)
    elif file_extension.lower() in [".dng"]:
        import Helpers.RAW_reader as RAW_reader
        lux_r, lux_g, lux_b = RAW_reader.RAW_to_lux(uploaded)
    else:
        raise RuntimeError("不支持的文件格式，请上传JPEG或PNG图像。")

    if film_type == "HP5（单色负片）":
        import Films.HP5.HP5 as HP5
        return HP5.process(lux_r, lux_g, lux_b, grain_style)
    if film_type == "Gold 200（彩色负片）":
        import Films.Gold200.Gold200 as Gold200
        return Gold200.process(lux_r, lux_g, lux_b, grain_style)
    if film_type == "5213（光谱实验）":
        # 5213 包名以数字开头，普通 import 语法不可用，这里用 importlib
        model_5213 = importlib.import_module("Films.5213.5213")
        return model_5213.process(
            lux_r, lux_g, lux_b,
            use_spectral_input=use_spectral_input,
            psf_mode=psf_mode,
            grain_ss=grain_ss,
            exposure_offset=exposure_offset,
            gamma=gamma,
            clip_mode=clip_mode,
            wb_strength=wb_strength,
            wb_gray=wb_gray,
            s_strength=s_strength,
            slope=slope,
        )
    raise RuntimeError(f"未知胶片类型：{film_type}")


# 设置页面配置 
st.set_page_config(
    page_title="Phos. 胶片模拟",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 文件上传
uploaded_image = st.file_uploader(
    "选择一张底片来开始冲洗（建议使用RAW（.dng)格式）",
    type=["jpg", "jpeg", "png", "dng"],
    help="上传一张底片冲洗试试看吧，选用RAW文件（.dng）效果会好很多哦"
)

# 创建侧边栏
with st.sidebar:
    st.header("Phos.")
    st.subheader("基于计算光学的胶片模拟")
    st.text("")
    st.text("原理验证demo")
    st.text("ver_0.3.0")
    st.text("")
    st.text("胶片设置")
    
    # 胶片类型选择
    film_type = st.selectbox(
        "胶片模拟模型:",
        ["HP5（单色负片）","Gold 200（彩色负片）","5213（光谱实验）"],
        index=0,
        help='''选择胶片模拟配方:

        HP5：
        基于依尔福HP5黑白负片，均衡的全色
        黑白胶片，作为单色负片模型的实验品
        ，反差适中，影调鲜明，似乎拍什么都
        不会出错。

        Gold 200：
        基于柯达Gold 200彩色负片，常用的
        彩色负片，作为彩负的实验品，融入了
        一小部分主观调整。

        5213（光谱实验）：
        基于柯达VISION3 200T（5213/7213），
        0.3.0 所有新特性的宿主：光谱输入、
        PSF光扩散、密度域颗粒，输出端固定
        走染料堆叠→客观扫描→反转
        （对数密度域三点锚定）。光谱输入采用
        CIE-RGB色匹配函数伪发射迁移，默认开启。

        '''
    )
    
    # 胶片颗粒度选择
    grain_style = st.selectbox(
        "胶片颗粒度：",
        ["默认", "柔和", "较粗", "不使用"],
        index=0,
        help="选择胶片的颗粒度"
    )

    # 0.3.0 新增：5213 光谱实验模型的参数（只在选择 5213 时出现）
    # 染料堆叠 / 客观扫描 / 反转固定开启，不再提供开关
    use_spectral_input = True
    psf_mode = "new"
    grain_ss = 2
    exposure_offset = 0.0
    gamma = 1.60
    clip_mode = "dynamic"
    wb_strength = 0.0
    wb_gray = 0.18
    s_strength = 0.10
    slope = 1.0
    if film_type == "5213（光谱实验）":
        st.text("")
        st.text("5213 光谱实验设置")
        use_spectral_input = st.checkbox(
            "光谱输入",
            value=True,
            help="开启：CIE-RGB标准色匹配函数曲线（r̄/ḡ/b̄）作伪发射源"
                 "→按实测灵敏度吸收积分（默认）；"
                 "关闭：直接标准RGB作层曝光"
        )
        psf_mode = st.selectbox(
            "光扩散（PSF）",
            ["new", "old"],
            index=0,
            format_func=lambda x: {
                "new": "新（MTF曲线拟合）",
                "old": "旧（猜测双高斯）",
            }[x],
            help="新=由 datasheet 模量传递函数逐层频域拟合（核心+翼-邻界）；"
                 "旧=0.3.0 早期猜测的双高斯核（保留作对照）"
        )
        s_strength = st.slider(
            "趾肩 S 曲线强度",
            min_value=0.0, max_value=0.30, value=0.10, step=0.01,
            help="输出曲线：端点不动、中亮部压一点、中暗部提一点；0=关闭"
        )
        slope = st.slider(
            "全局斜率",
            min_value=0.5, max_value=1.5, value=1.0, step=0.05,
            help="围绕中灰目标点的全局对比度：1.0=关闭；只改对比、不改色彩平衡"
        )
        grain_ss = int(st.selectbox(
            "颗粒超采样",
            ["1", "2", "3"],
            index=1,
            format_func=lambda x: {
                "1": "关闭（1x）",
                "2": "2x 超采样",
                "3": "3x 超采样",
            }[x],
            help="颗粒采用密度域伽马聚簇+平滑赋形+unsharp（固定算法）"
                 "+48μm孔径闭环）；超采样在高分辨率生成颗粒纹理再块平均"
                 "降采样，模拟细颗粒被像素积分、消除网格采样伪影；"
                 "2x 内存约为关闭的 4 倍，3x 约 9 倍"
        ))
        exposure_offset = st.slider(
            "曝光补偿（档）",
            min_value=-3.0, max_value=3.0, value=0.0, step=0.5
        )
        gamma = st.slider(
            "输出伽马",
            min_value=1.0, max_value=3.0, value=1.60, step=0.1,
            help="反转后叠的显示伽马：数值越大中间调越压暗、对比越高；越小越柔和"
        )
        clip_mode = st.selectbox(
            "两端截止",
            ["dynamic", "static"],
            index=0,
            help="dynamic=1%/99% 分位动态截止（默认）；static=理论白/黑点静态截止，均匀场安全"
        )
        wb_strength = st.slider(
            "白平衡强度",
            min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
            help="中灰白平衡（密度域加法偏移）：0=不起用（默认）；"
                 "正=校平中灰；负=反向增强原有偏色"
        )
        wb_gray = st.slider(
            "白平衡灰点",
            min_value=0.05, max_value=0.60, value=0.18, step=0.05,
            help="白平衡参考灰点（线性输入灰值），试试 0.10/0.18/0.30/0.50 看哪个影响最小"
        )
    
    # 显示当前选择的胶片类型
    st.success(f"已选择胶片: {film_type}")

# 根据上传的图像和选择的胶片类型调用相应的处理函数
if uploaded_image is not None:
    
    start_time = time.time()

    # 0.3.0 修复：先把后续会引用的变量初始化，
    # 避免读取或处理失败时走到下方判断处抛出 NameError
    film = None

    try:
        # 一次性取出文件字节（后续重跑直接走缓存，不再重复处理）
        if hasattr(uploaded_image, "getvalue"):
            file_bytes = uploaded_image.getvalue()
        else:
            file_bytes = uploaded_image.read()
        film = run_film(
            file_bytes, uploaded_image.name, film_type, grain_style,
            use_spectral_input, psf_mode, grain_ss, exposure_offset,
            gamma, clip_mode, wb_strength, wb_gray, s_strength, slope,
        )

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
