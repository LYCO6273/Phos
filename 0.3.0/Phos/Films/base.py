"""Shared film data loading and the physical colour-negative pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Core.cie_data import DELTA_LAMBDA, D65, WL
from Core.color import scan_weights, xyz_to_srgb
from Core.spectral import SpectrumLUT, normalize_luminance


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"


def load_curve_csv(rel_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column curve CSV as (x, y) float arrays."""
    data = np.loadtxt(DATA_DIR / rel_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def interp_on_wl(rel_path: str, taper_nm: float = 10.0) -> np.ndarray:
    """Load a wavelength/spectral CSV and resample onto the 400-700 nm grid.

    Values outside the measured range are zero; a short cosine taper is
    applied at both ends so the sensitivity does not cut off abruptly.
    """
    data = np.loadtxt(DATA_DIR / rel_path, delimiter=",", skiprows=1)
    wl_src = data[:, 0]
    val_src = data[:, 1]
    out = np.interp(WL, wl_src, val_src, left=0.0, right=0.0)
    n = int(round(taper_nm / 5.0))
    idx = np.flatnonzero(out > 0.0)
    if n > 0 and len(idx) > 2 * n:
        first, last = int(idx[0]), int(idx[-1])
        ramp_head = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n + 1)))
        ramp_tail = ramp_head[::-1]
        if first > 0:
            out[first:first + n + 1] *= ramp_head
        if last < len(out) - 1:
            out[last - n:last + 1] *= ramp_tail
    return out


def normalize_sensitivities(sens: np.ndarray) -> np.ndarray:
    """Scale each layer so that int S_i(lambda) D65(lambda) dlambda == 1.

    A neutral 18% gray then produces E_i = 0.18 for every layer, which keeps
    the exposure scale comparable to the 0.2.3 "lux" convention.
    """
    sens = np.asarray(sens, dtype=np.float64)
    scale = np.sum(sens * D65[None, :] * DELTA_LAMBDA, axis=1, keepdims=True)
    return sens / np.maximum(scale, 1e-12)


def build_exposure_lut(lut: SpectrumLUT, sensitivity: np.ndarray) -> np.ndarray:
    """Build the per-film XYZ -> layer-exposure LUT."""
    sens = normalize_sensitivities(sensitivity)
    return lut.layer_exposure_lut(sens)


def sample_exposures(lut: SpectrumLUT, exp_lut: np.ndarray,
                     xyz: np.ndarray) -> np.ndarray:
    """Sample layer exposures, preserving >1 highlights by luminance scaling."""
    xyz_eff, y_scale = normalize_luminance(xyz)
    e = lut.sample(exp_lut, xyz_eff)
    # y_scale already carries the trailing channel axis.
    return e * y_scale


def color_negative_process(
    layer_exposures: np.ndarray,
    char_curves: list[tuple[np.ndarray, np.ndarray]],
    dye_spectra: np.ndarray,
    dmin_spectrum: np.ndarray,
    dmin_density: np.ndarray,
    speed_offsets: np.ndarray | None = None,
    exposure_ev: float = 0.0,
    print_contrast: float = 1.0,
) -> np.ndarray:
    """Run the physical colour-negative pipeline.

    layer exposures -> characteristic curves -> dye spectral stack (+ orange
    mask) -> scan to XYZ -> per-channel density inversion -> sRGB.
    """
    e = np.asarray(layer_exposures, dtype=np.float32)
    log_h = np.log10(np.maximum(e, 1e-8))
    if speed_offsets is not None:
        log_h = log_h + np.asarray(speed_offsets, dtype=np.float32)
    log_h = log_h + exposure_ev * np.log10(2.0)

    n = len(char_curves)
    density = np.empty_like(e)
    for i, (xp, fp) in enumerate(char_curves):
        density[..., i] = np.interp(log_h[..., i], xp, fp)

    # Remove the mask that is already included in the characteristic curves,
    # then rebuild the spectral stack: dye densities + the orange mask.
    dye = np.asarray(dye_spectra, dtype=np.float64)      # (n, 61), peak=1
    dmin = np.asarray(dmin_spectrum, dtype=np.float64)   # (61,)
    d_extra = np.maximum(density - dmin_density[None, None, :], 0.0)
    d_total = np.tensordot(d_extra, dye, axes=([2], [0])) + dmin[None, None, :]
    trans = np.power(10.0, -d_total)                     # (H, W, 61)

    scan_xyz = np.einsum("...w,wc->...c", trans, scan_weights())
    rgb_scan = np.clip(xyz_to_srgb(scan_xyz), 1e-6, 1.0)

    d_scan = -np.log10(rgb_scan)

    # Fully exposed negative: each layer at its curve maximum.  The orange
    # mask is already inside d_scan, so it cancels in the final ratio and the
    # inversion simply maps bright scene -> clear positive.
    d_max_layer = np.array([fp[-1] for _, fp in char_curves], dtype=np.float64)
    d_extra_max = np.maximum(d_max_layer - dmin_density, 0.0)
    d_total_max = np.tensordot(d_extra_max, dye, axes=([0], [0])) + dmin
    trans_max = np.power(10.0, -d_total_max)
    scan_max = np.einsum("w,wc->c", trans_max, scan_weights())
    rgb_max = np.clip(xyz_to_srgb(scan_max), 1e-6, 1.0)
    d_scan_max = -np.log10(rgb_max)

    d_pos = np.clip(d_scan_max[None, None, :] - d_scan, 0.0, None)
    out_linear = np.power(10.0, -print_contrast * d_pos)
    return out_linear.astype(np.float32)


def calibrate_linear_output(
    out: np.ndarray,
    char_curves: list[tuple[np.ndarray, np.ndarray]],
    dye_spectra: np.ndarray,
    dmin_spectrum: np.ndarray,
    dmin_density: np.ndarray,
    speed_offsets: np.ndarray | None = None,
    target_gray_linear: float = 0.19,
) -> np.ndarray:
    """Scanner-style neutral calibration: black -> 0, 18% gray -> target.

    The orange mask and dye cross-talk leave a residual per-channel offset in
    the scanned density domain; a real scanner removes it by calibrating on
    D-min and a gray card.  This applies the same two-point correction.
    """
    refs = []
    for value in (0.0, 0.18):
        e = np.full((1, 1, 3), value, dtype=np.float32)
        refs.append(
            color_negative_process(
                e,
                char_curves,
                dye_spectra,
                dmin_spectrum,
                dmin_density,
                speed_offsets=speed_offsets,
            )[0, 0]
        )
    black_ref, gray_ref = refs
    scale = target_gray_linear / np.maximum(gray_ref - black_ref, 1e-6)
    offset = -scale * black_ref
    return np.clip(out * scale + offset, 0.0, 1.0)
