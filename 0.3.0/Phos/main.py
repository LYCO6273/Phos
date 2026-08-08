"""
Phos 0.3.0 - spectral migration validation demo.

The input is converted to linear XYZ(D65), a shared xyz->reflectance LUT is
sampled per pixel, and each film integrates the reconstructed spectrum
against its emulsion layer sensitivities before the tone/dye pipeline.

No LUTs, we calculate LUX.
"""

from __future__ import annotations

import io
import os
import time

import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Phos. 胶片模拟 0.3.0",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="正在构建光谱重建 LUT（首次约 10~30 秒）...")
def get_spectrum_lut():
    from Core.spectral import SpectrumLUT

    return SpectrumLUT.build(verbose=False)


uploaded_image = st.file_uploader(
    "选择一张底片来开始冲洗（建议使用 RAW（.dng）格式）",
    type=["jpg", "jpeg", "png", "dng"],
    help="0.3.0 会先把输入重建为 61 波段光谱，再进入各乳剂层。",
)

with st.sidebar:
    st.header("Phos.")
    st.subheader("基于计算光学的胶片模拟")
    st.text("原理验证 demo · ver_0.3.0")
    st.text("")
    st.text("胶片设置")

    film_type = st.selectbox(
        "胶片模拟模型:",
        ["HP5（单色负片）", "Gold 200（彩色负片）", "Vision 200T（电影负片）"],
        index=0,
        help="HP5/Gold200 沿用 0.2.3 的影调；Vision200T 使用完整的 Datasheet 数据。",
    )
    grain_style = st.selectbox(
        "胶片颗粒度：",
        ["默认", "柔和", "较粗", "不使用"],
        index=0,
    )
    exposure_ev = st.slider(
        "曝光补偿 (EV)",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
        help="正值让底片更亮（负片变厚、正片变亮）。",
    )
    st.success(f"已选择胶片: {film_type}")


if uploaded_image is not None:
    start = time.time()
    try:
        with st.spinner("解析输入文件..."):
            ext = os.path.splitext(uploaded_image.name)[1].lower()
            if ext in (".jpg", ".jpeg", ".png"):
                import Helpers.Jpeg_reader as jpeg

                xyz = jpeg.Jpeg_to_xyz(uploaded_image)
            elif ext == ".dng":
                import Helpers.RAW_reader as raw

                xyz = raw.RAW_to_xyz(uploaded_image, bright=3.0)
            else:
                st.error("不支持的文件格式，请上传 JPEG/PNG/DNG。")
                xyz = None

        if xyz is not None:
            lut = get_spectrum_lut()
            with st.spinner("正在冲洗..."):
                if film_type == "HP5（单色负片）":
                    import Films.HP5.HP5 as hp5

                    film = hp5.process(xyz, lut, grain_style, exposure_ev)
                elif film_type == "Gold 200（彩色负片）":
                    import Films.Gold200.Gold200 as gold

                    film = gold.process(xyz, lut, grain_style, exposure_ev)
                else:
                    import Films.Vision200T.Vision200T as vision

                    film = vision.process(xyz, lut, grain_style, exposure_ev)

            st.image(film, width="stretch")
            st.success(f"底片显影好了，用时 {time.time() - start:.2f} 秒")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"phos_0.3_{timestamp}.jpg"
            film_pil = Image.fromarray(film)
            buf = io.BytesIO()
            film_pil.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="下载高清图像",
                data=buf.getvalue(),
                file_name=output_path,
                mime="image/jpeg",
            )
    except Exception as exc:  # noqa: BLE001 - demo UI
        st.error(f"处理图像时出现错误: {exc}")
        import traceback

        st.code(traceback.format_exc())

