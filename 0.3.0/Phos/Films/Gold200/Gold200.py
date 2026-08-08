"""Kodak Gold 200 (C-41 colour negative) with spectral layer exposures.

Gold's own spectral-sensitivity chart is stored in the PDF as sampled
vertical strokes, so this build uses the fully extracted Vision 200T
sensitivities as a proxy (documented approximation).  Its characteristic
curves and orange-mask (D-min) curve are digitized from the Gold datasheet;
the C/M/Y dye spectra use the Vision 200T dyes (same Kodak dye family).
"""

from __future__ import annotations

import numpy as np

from Core.color import linear_to_srgb
from Core.spectral import SpectrumLUT
from Films.base import (
    build_exposure_lut,
    calibrate_linear_output,
    color_negative_process,
    interp_on_wl,
    load_curve_csv,
    sample_exposures,
)
from Helpers.optics import grain, opt_channel, opt_halo


def _sensitivities() -> np.ndarray:
    b = interp_on_wl("vision200t/vision200t_sensitivity_b.csv")
    g = interp_on_wl("vision200t/vision200t_sensitivity_g.csv")
    r = interp_on_wl("vision200t/vision200t_sensitivity_r.csv")
    return np.stack([10.0 ** b, 10.0 ** g, 10.0 ** r], axis=0)


def _char_curves():
    curves = []
    for name in ("b", "g", "r"):
        xp, fp = load_curve_csv(f"gold200/gold200_char_curve_{name}.csv")
        curves.append((xp, fp))
    return curves


def _dye_spectra() -> np.ndarray:
    yellow = interp_on_wl("vision200t/vision200t_dye_yellow.csv", taper_nm=0.0)
    magenta = interp_on_wl("vision200t/vision200t_dye_magenta.csv", taper_nm=0.0)
    cyan = interp_on_wl("vision200t/vision200t_dye_cyan.csv", taper_nm=0.0)
    return np.stack([yellow, magenta, cyan], axis=0)


def _dmin_spectrum() -> np.ndarray:
    return interp_on_wl("gold200/gold200_dmin.csv", taper_nm=0.0)


def process(xyz: np.ndarray,
            spectrum_lut: SpectrumLUT,
            grain_style: str = "默认",
            exposure_ev: float = 0.0) -> np.ndarray:
    """Gold 200: spectral layer exposures -> optics -> negative pipeline."""
    exp_lut = build_exposure_lut(spectrum_lut, _sensitivities())
    e = sample_exposures(spectrum_lut, exp_lut, xyz)
    e = e * (2.0 ** exposure_ev)

    # Optical diffusion, same structure as 0.2.3 (per layer + halo).
    e_total = 0.2 * e[..., 0] + 0.35 * e[..., 1] + 0.4 * e[..., 2]
    halo = opt_halo(e_total)
    e_r = opt_channel(e[..., 0], 50) + halo * 0.15
    e_g = opt_channel(e[..., 1], 45)
    e_b = opt_channel(e[..., 2], 35)

    # Grain with the 0.2.3 cross-channel mixing.
    if grain_style == "较粗":
        n_r, n_g, n_b = grain(e_r) * 1.5, grain(e_g) * 1.5, grain(e_b) * 1.5
    elif grain_style == "柔和":
        n_r, n_g, n_b = grain(e_r) * 0.5, grain(e_g) * 0.5, grain(e_b) * 0.5
    elif grain_style == "不使用":
        n_r = n_g = n_b = np.zeros_like(e_r)
    else:
        n_r, n_g, n_b = grain(e_r), grain(e_g), grain(e_b)
    e_r = np.clip(e_r + n_r * 0.1 + n_b * 0.03 + n_g * 0.03, 0, 1)
    e_g = np.clip(e_g + n_g * 0.1 + n_r * 0.03 + n_b * 0.03, 0, 1)
    e_b = np.clip(e_b + n_b * 0.1 + n_r * 0.03 + n_g * 0.03, 0, 1)

    curves = _char_curves()
    dmin_density = np.array([fp[0] for _, fp in curves], dtype=np.float32)
    offsets = np.array([1.818, 1.771, 1.826], dtype=np.float32)
    out = color_negative_process(
        np.stack([e_r, e_g, e_b], axis=-1),
        curves,
        _dye_spectra(),
        _dmin_spectrum(),
        dmin_density,
        speed_offsets=offsets,
    )
    out = calibrate_linear_output(out, curves, _dye_spectra(), _dmin_spectrum(),
                                  dmin_density, offsets)
    out = linear_to_srgb(out)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
