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


with st.sidebar:
    st.header("Phos. 胶片模拟")
    st.subheader("基于计算光学的胶片模拟")
    st.text("")
    st.text("原理验证demo")
    st.text("ver_0.1.1")
    st.text("")
    st.text("🎞️ 胶片设置")

    film_type = st.selectbox(
        "请选择胶片:",
        FILM_TYPES,
        index=0,
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

