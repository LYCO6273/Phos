from __future__ import annotations

import numpy as np


try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python 未安装：请先安装 opencv-python 才能使用颗粒/处理功能。")
    return cv2


def _gaussian_blur_float(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    cv2_mod = _require_cv2()
    ksize = int(max(3, round(sigma * 6)))
    if ksize % 2 == 0:
        ksize += 1
    return cv2_mod.GaussianBlur(
        image,
        (ksize, ksize),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2_mod.BORDER_REFLECT101,
    )


def _grain_weight(lux: np.ndarray, sens: float) -> np.ndarray:
    shadow = np.power(np.clip(1.0 - lux, 0.0, 1.0), 0.7)
    mid = 1.0 - np.power(np.abs(lux - 0.5) * 2.0, 1.8)
    mid = np.clip(mid, 0.0, 1.0)
    weight = 0.25 + 0.75 * (0.65 * shadow + 0.35 * mid)
    weight *= np.clip(0.55 + 0.9 * sens, 0.6, 1.1)
    return np.clip(weight, 0.05, 1.0).astype(np.float32)


def _grain_field(shape: tuple[int, int], rng: np.random.Generator, sigma: float) -> np.ndarray:
    white = rng.standard_normal(shape).astype(np.float32)
    low = _gaussian_blur_float(white, sigma)
    high = white - _gaussian_blur_float(white, sigma * 2.5)
    field = 0.65 * low + 0.35 * high
    return np.tanh(field * 1.25).astype(np.float32)


def grain(
    lux_r,
    lux_g,
    lux_b,
    lux_total,
    color_type,
    sens: float,
    grain_size: float,
    grain_seed: int,
):
    sigma = float(np.clip(grain_size, 0.35, 6.0))
    rng = np.random.default_rng(int(grain_seed) & 0xFFFFFFFF)

    if color_type == ("color"):
        base = _grain_field(lux_total.shape, rng, sigma)
        chroma = _grain_field(lux_total.shape, rng, sigma * 0.8)

        grain_r = 0.9 * base + 0.1 * chroma
        grain_g = 0.9 * base + 0.1 * np.roll(chroma, shift=(1, 0), axis=(0, 1))
        grain_b = 0.9 * base + 0.1 * np.roll(chroma, shift=(0, 1), axis=(0, 1))

        weighted_noise_r = np.clip(grain_r * _grain_weight(lux_r, sens), -1.0, 1.0)
        weighted_noise_g = np.clip(grain_g * _grain_weight(lux_g, sens), -1.0, 1.0)
        weighted_noise_b = np.clip(grain_b * _grain_weight(lux_b, sens), -1.0, 1.0)
        weighted_noise_total = None
    else:
        base = _grain_field(lux_total.shape, rng, sigma)
        weighted_noise_total = np.clip(base * _grain_weight(lux_total, sens), -1.0, 1.0)
        weighted_noise_r = None
        weighted_noise_g = None
        weighted_noise_b = None

    return weighted_noise_r, weighted_noise_g, weighted_noise_b, weighted_noise_total
