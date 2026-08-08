"""HP5 Plus (B&W negative) - spectral luminance drives the 0.2.3 tone map."""

from __future__ import annotations

import numpy as np

from Core.cie_data import CMF
from Core.spectral import SpectrumLUT
from Films.base import build_exposure_lut, sample_exposures
from Helpers.optics import grain, opt_hp5


# The HP5 datasheet only provides a qualitative wedge spectrogram, so the
# panchromatic response is approximated with the CIE photopic luminosity
# function y_bar.  This makes HP5 exposure equal to scene luminance Y.
PANCHROMATIC = CMF[:, 1][None, :]


def _neg_tone(lux: np.ndarray) -> np.ndarray:
    """Negative tone map, identical to 0.2.3 (curve digitized from datasheet)."""
    xp = np.array([0.000, 0.242, 0.503, 0.758, 0.993, 1.255, 1.497, 1.745, 1.993,
                   2.248, 2.490, 2.745, 3.000, 3.261, 3.490, 3.739, 4.000, 4.235,
                   4.490, 4.758, 5.000], dtype=np.float32)
    fp = np.array([0.175, 0.181, 0.188, 0.208, 0.261, 0.341, 0.467, 0.633, 0.792,
                   0.958, 1.117, 1.277, 1.442, 1.608, 1.754, 1.914, 2.080, 2.199,
                   2.272, 2.312, 2.338], dtype=np.float32)
    relative_log = 5 * (0.247190 * np.log10(5.555556 * lux + 0.122272) + 0.385537)
    density = np.interp(relative_log, xp, fp)
    pt = 10.0 ** (-density)
    result = np.clip((0.669 - pt) * 1.50, 0, 1)
    return result ** 2.0


def process(xyz: np.ndarray,
            spectrum_lut: SpectrumLUT,
            grain_style: str = "默认",
            exposure_ev: float = 0.0) -> np.ndarray:
    """HP5: spectral luminance -> optical diffusion -> grain -> tone map."""
    exp_lut = build_exposure_lut(spectrum_lut, PANCHROMATIC)
    e = sample_exposures(spectrum_lut, exp_lut, xyz)[..., 0]
    e = e * (2.0 ** exposure_ev)

    e = opt_hp5(e)
    if grain_style == "较粗":
        noise = grain(e) * 1.5
    elif grain_style == "柔和":
        noise = grain(e) * 0.5
    elif grain_style == "不使用":
        noise = np.zeros_like(e)
    else:
        noise = grain(e)
    e = np.clip(e + noise * 0.12, 0, 1)

    result = _neg_tone(e)
    return (result * 255.0).astype(np.uint8)

