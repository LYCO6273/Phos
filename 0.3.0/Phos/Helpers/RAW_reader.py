"""RAW/DNG reader producing linear XYZ(D65) floats.

Highlights are preserved up to the 16-bit output ceiling (bright x white
level).  iPhone DNGs carry a BaselineExposure compensation in their metadata
that LibRaw does not apply; the bright parameter is the demo's stand-in for
that compensation and defaults to 3.0 to match the 0.2.3 behaviour.
"""

from __future__ import annotations

import cv2
import numpy as np

from Core.color import srgb_to_xyz


def standardize(image: np.ndarray, min_size: int = 3024) -> np.ndarray:
    """Resize so the short side is min_size, keeping even dimensions."""
    height, width = image.shape[:2]
    if height < width:
        scale = min_size / height
        new_h, new_w = min_size, int(width * scale)
    else:
        scale = min_size / width
        new_w, new_h = min_size, int(height * scale)
    new_w += new_w % 2
    new_h += new_h % 2
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def RAW_to_xyz(uploaded_image, bright: float = 3.0) -> np.ndarray:
    """Decode a DNG/RAW upload into linear XYZ(D65) float32, shape (H, W, 3).

    The pipeline uses rawpy/LibRaw's internal black/white-level handling and
    camera white balance, then treats the linear sRGB output as the input
    colour space before converting to XYZ.
    """
    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "缺少 rawpy 依赖，请先安装：pip install rawpy"
        ) from exc

    with rawpy.imread(uploaded_image) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            bright=bright,
            output_bps=16,
        )
    rgb_float = rgb.astype(np.float32) / 65535.0
    bgr = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2BGR)
    bgr = standardize(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    xyz = srgb_to_xyz(rgb).astype(np.float32)
    return np.maximum(xyz, 0.0)

