"""JPEG/PNG reader producing linear XYZ(D65) floats (approximate)."""

from __future__ import annotations

import cv2
import numpy as np

from Core.color import srgb_to_linear, srgb_to_xyz


def standardize(image: np.ndarray, min_size: int = 3024) -> np.ndarray:
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


def Jpeg_to_xyz(uploaded_image) -> np.ndarray:
    """Decode an 8-bit sRGB upload into linear XYZ(D65) float32."""
    data = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("无法解码图片，请确认文件格式")
    bgr = standardize(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    linear = srgb_to_linear(rgb)
    xyz = srgb_to_xyz(linear).astype(np.float32)
    return np.maximum(xyz, 0.0)

