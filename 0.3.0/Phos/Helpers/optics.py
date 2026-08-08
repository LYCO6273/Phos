"""Optical diffusion and grain helpers, carried over from 0.2.3 unchanged.

These are empirical approximations; the formulas are kept identical to the
0.2.3 film modules so the spectral migration can be compared A/B.
"""

from __future__ import annotations

import cv2
import numpy as np


def average(lux: np.ndarray) -> float:
    avg = float(np.mean(lux))
    return float(np.clip(avg, 0, 1))


def grain(lux: np.ndarray) -> np.ndarray:
    """Weighted random grain, same as 0.2.3."""
    avrl = average(lux)
    sens = np.clip((1.0 - avrl) * 0.75 + 0.10, 0.35, 0.65)
    noise = np.random.normal(0, 1, lux.shape).astype(np.float32) ** 2
    noise = noise * (np.random.choice([-1, 1], lux.shape))
    weights = np.clip((0.5 - np.abs(lux - 0.5)) * 2, 0.05, 0.9)
    sens_grain = np.clip(sens, 0.4, 0.6)
    weighted_noise = noise * weights * sens_grain
    weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
    return np.clip(weighted_noise, -1, 1)


def opt_channel(lux: np.ndarray, radius_cap: int = 50, blend: float = 0.9) -> np.ndarray:
    """Per-channel optical diffusion (0.2.3 Gold200 style)."""
    avrl = average(lux)
    sens = np.clip((1.0 - avrl) * 0.75 + 0.10, 0.35, 0.7)
    strg = 23 * sens**2
    rads = np.clip(int(radius_cap * (sens**2)), 1, radius_cap)
    ksize = rads * 2 + 1
    weights = np.clip(np.log(2.7 * lux + 1) * sens, 0, 1)
    bloom_base = cv2.GaussianBlur(lux * weights, (ksize, ksize), sens * 35)
    bloom_effect = bloom_base * weights * strg
    return lux * blend + bloom_effect * 0.05 - weights * 0.05


def opt_hp5(lux: np.ndarray) -> np.ndarray:
    """Optical diffusion exactly as 0.2.3 HP5 used it."""
    avrl = average(lux)
    sens = np.clip((1.0 - avrl) * 0.75 + 0.10, 0.35, 0.7)
    strg = 23 * sens**2
    rads = np.clip(int(35 * (sens**2)), 1, 50)
    ksize = rads * 2 + 1
    weights = np.clip(np.log(2.7 * lux + 1) * sens, 0, 1)
    bloom_base = cv2.GaussianBlur(lux * weights, (ksize, ksize), sens * 35)
    bloom_effect = bloom_base * weights * strg
    return lux * 0.95 + bloom_effect * 0.05 - weights * 0.05


def opt_halo(lux_total: np.ndarray, max_radius: int = 100) -> np.ndarray:
    """Highlight halo, same formula as 0.2.3 Gold200 opt_h."""
    avrl = average(lux_total)
    sens = np.clip((1.0 - avrl) * 0.75 + 0.10, 0.35, 0.7)
    strg = 23 * sens**2
    rads = np.clip(int(max_radius * (sens**2)), 1, max_radius)
    ksize = rads * 2 + 1
    lux_total = np.clip(lux_total - 0.8, 0, 1) * 5
    weights = np.clip((lux_total**5) * sens, 0, 1)
    bloom_base = cv2.GaussianBlur(lux_total * weights, (ksize * 3, ksize * 3), sens * 35)
    bloom_effect = bloom_base * weights * strg
    return bloom_effect * 0.15


def apply_grain(lux_channels: list[np.ndarray], grain_style: str) -> list[np.ndarray]:
    """Apply grain to each exposure channel with the 0.2.3 style rules."""
    out = []
    for lux in lux_channels:
        if grain_style == "较粗":
            noise = grain(lux) * 1.5
        elif grain_style == "柔和":
            noise = grain(lux) * 0.5
        elif grain_style == "不使用":
            noise = np.zeros_like(lux)
        else:
            noise = grain(lux)
        out.append(lux + noise * 0.1)
    return out
