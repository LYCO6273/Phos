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

#这个python文件只是前端部分，实际上会按三段式调用：
#   第一段 Input   原片解析 → 线性光照度
#   第二段 Films   胶片模拟 → 未处理扫描（负片/正片）
#   第三段 Scanner 扫描仪渲染 → 成片

#赛博请神
import streamlit as st
import time
from PIL import Image
import io
import importlib
import hashlib
import numpy as np
import cv2


# 第二段扫描缓存：最多保留 3 组（13.7MP 彩色扫描约 165MB/组，需控制内存）
_SCAN_CACHE_MAX = 3


def _file_id(file_bytes):
    """文件字节指纹：作为扫描缓存键的一部分（避免仅凭文件名撞键）。"""
    return hashlib.md5(file_bytes).hexdigest()[:16]


def _cached_scan(key, compute):
    """按 (文件ID, 胶片, 颗粒, 曝光) 缓存第二段扫描，跨 rerun 复用。

    Streamlit 每次滑块变化都会整脚本重跑：对比度变化只重算第三段
    （render，约 50ms），曝光变化需重算第二段，但回到用过的档位
    直接命中缓存。颗粒随机性随缓存固定：同一参数组合的颗粒不再闪动。
    """
    st.session_state.setdefault("_phos_scan_cache", {})
    cache = st.session_state["_phos_scan_cache"]
    hit = cache.get(key)
    if hit is not None:
        return hit
    scan = compute()
    cache[key] = scan
    if len(cache) > _SCAN_CACHE_MAX:
        for k in list(cache)[:len(cache) - _SCAN_CACHE_MAX]:
            del cache[k]
    return scan


def run_film(file_bytes, file_name, film_type,
             grain_ss, exposure_offset, contrast, export_negative=False,
             optical_correction=True):
    """三段式主流程（第二段扫描按参数缓存，第三段实时重算）。

    第一段：原片字节 → 线性光照度；
    第二段：光照度 → 未反转扫描（彩负/正片）或密度（黑白），按
            文件ID + 胶片 + 颗粒 + 曝光缓存；
    第三段：扫描仪渲染 → 成片（对比度/导出即时，含可选 16bit 原片）。
    """
    from Input import decode_to_lux
    from Scanner import render

    films = {
        "HP5（单色负片）": ("hp5", "Films.HP5.HP5"),
        "Gold 200（彩色负片）": ("gold200", "Films.Gold200.Gold200"),
        "5213（彩色负片）": ("5213", "Films.5213.5213"),
        "Provia 100F（彩色正片）": ("provia100f", "Films.Provia100F.Provia100F"),
    }
    if film_type not in films:
        raise RuntimeError(f"未知胶片类型：{film_type}")
    key, module_path = films[film_type]

    # 第一段（每次重跑约 0.2s，不值得占内存缓存）
    lux_r, lux_g, lux_b = decode_to_lux(
        file_bytes, file_name, optical=optical_correction)

    # 第二段（按 文件ID+胶片+颗粒+曝光 缓存；曝光档位复用即时命中）
    def compute_scan():
        model = importlib.import_module(module_path)
        return model.process_scan(
            lux_r, lux_g, lux_b,
            grain_ss=grain_ss,
            exposure_offset=exposure_offset,
        )

    scan = _cached_scan(
        (_file_id(file_bytes), key, int(grain_ss), float(exposure_offset)),
        compute_scan)

    # 第三段（对比度变化只重算这里，约 50ms）
    out = render(scan, key, contrast=contrast)
    return (out, scan) if export_negative else (out, None)


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
    st.text("ver_0.3.3")
    st.text("")
    st.text("胶片设置")
    
    # 胶片类型选择
    film_type = st.selectbox(
        "胶片模拟模型:",
        ["HP5（单色负片）", "Gold 200（彩色负片）", "5213（彩色负片）",
         "Provia 100F（彩色正片）"],
        index=0,
        help='''选择胶片模拟配方:

        HP5：
        基于依尔福HP5黑白负片，均衡的全色
        黑白胶片，作为单色负片模型的实验品
        ，反差适中，影调鲜明，似乎拍什么都
        不会出错。

        Gold 200：
        基于柯达Gold 200彩色负片，C-41 入门
        消费彩负；与 5213 同结构（光谱重建输入、
        物理PSF、密度域颗粒、染料堆叠→客观扫描
        →扫描仪反转渲染）。

        5213：
        基于柯达VISION3 200T（5213/7213），
        新特性的宿主：光谱重建输入、PSF光扩散、
        密度域颗粒，输出端固定走染料堆叠→客观
        扫描→扫描仪反转渲染（对数密度域固定锚点）。

        Provia 100F：
        基于富士 FUJICHROME Provia 100F（RDP III）
        彩色反转片；正片模型：光谱重建输入、物理
        PSF、密度域颗粒、染料堆叠→客观扫描
        （扫描值即正像，无需反转）→显示编码。
        无校色、无去色罩：正片色彩是它自己的。

        '''
    )
    
    # 所有胶片统一控件（光谱重建输入固定开启，无开关）
    grain_ss = 2
    exposure_offset = 0.0
    optical_correction = st.checkbox(
        "镜头光学修正（暗角/畸变）",
        value=True,
        help="按 DNG 规范应用文件内嵌的 OpcodeList 修正：部分相机在 "
             "OpcodeList3 里携带 GainMap 暗角增益表（四角可超过 +2 EV），"
             "rawpy 不执行 opcode，需第一段自行应用；无此数据的 DNG "
             "不受影响。畸变 opcode 暂未实现（当前样张均无）。"
    )

    export_negative = False
    grain_ss = int(st.selectbox(
        "颗粒超采样",
        ["1", "2"],
        index=1,
        format_func=lambda x: {
            "1": "1x",
            "2": "2x 超采样",
        }[x],
        help="颗粒采用密度域伽马聚簇+可分离高斯赋形（固定算法）"
             "+48μm孔径闭环）；2x 在高分辨率生成颗粒纹理再块平均"
             "降采样，模拟细颗粒被像素积分、消除网格采样伪影；"
             "2x 内存约为关闭的 4 倍"
    ))
    exposure_offset = st.slider(
        "曝光补偿（档）",
        min_value=-3.0, max_value=3.0, value=0.0, step=0.5
    )
    contrast = st.slider(
        "对比度",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
        help="风格对比度：以中灰（0.5）为锚叠正反 S 曲线，"
             "正值增加对比（暗更暗、亮更亮），负值降低对比（压向中灰），"
             "0 为线性。曝光正确性已由三锚点保证，此处纯为风格层。"
    )
    if film_type != "HP5（单色负片）":
        export_negative = st.checkbox(
            "导出 16bit 原片扫描（无反转/无显示编码）",
            value=False,
            help="同时导出一份未编码的线性扫描 TIFF（16bit，0-65535 全范围），"
                 "供自行调色/研究使用。彩负为负像，正片为正像。"
        )
    
    # 显示当前选择的胶片类型
    st.success(f"已选择胶片: {film_type}")

# 根据上传的图像和选择的胶片类型调用相应的处理函数
if uploaded_image is not None:
    
    start_time = time.time()

    # 0.3.0 修复：先把后续会引用的变量初始化，
    # 避免读取或处理失败时走到下方判断处抛出 NameError
    film = None
    negative_scan = None

    try:
        # 一次性取出文件字节（后续重跑直接走缓存，不再重复处理）
        if hasattr(uploaded_image, "getvalue"):
            file_bytes = uploaded_image.getvalue()
        else:
            file_bytes = uploaded_image.read()
        film, negative_scan = run_film(
            file_bytes, uploaded_image.name, film_type,
            grain_ss, exposure_offset, contrast, export_negative,
            optical_correction,
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
            if negative_scan is not None:
                # 未反转负片扫描 → 16bit 线性（0-65535 全范围）
                neg16 = (np.clip(negative_scan, 0.0, 1.0) * 65535.0 + 0.5)
                neg16 = neg16.astype(np.uint16)[:, :, ::-1]  # RGB -> BGR
                ok_tiff, tiff_bytes = cv2.imencode(".tiff", neg16)
                if ok_tiff:
                    st.download_button(
                        label="下载 16bit 负片原片（TIFF）",
                        data=tiff_bytes.tobytes(),
                        file_name=f"phos_{timestamp}_negative.tiff",
                        mime="image/tiff",
                    )
        except Exception as e:
            st.error(f"显示或下载图像时出现错误: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
