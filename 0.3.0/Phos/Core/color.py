"""sRGB / XYZ colour conversion helpers (IEC 61966-2-1, D65)."""

from __future__ import annotations

import numpy as np

from .cie_data import WHITE_D65


SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Decode sRGB display values (0..1) to linear RGB."""
    c = np.asarray(c, dtype=np.float32)
    return np.where(
        c <= 0.04045,
        c / 12.92,
        ((np.maximum(c, 0.0) + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Encode linear RGB to sRGB display values (0..1)."""
    c = np.asarray(c, dtype=np.float32)
    return np.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * np.maximum(c, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB -> XYZ(D65).  rgb may be (..., 3)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.einsum("...j,kj->...k", rgb, SRGB_TO_XYZ)


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """XYZ(D65) -> linear sRGB.  xyz may be (..., 3)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.einsum("...j,kj->...k", xyz, XYZ_TO_SRGB)


def scan_weights() -> np.ndarray:
    """Return (61, 3) integration weights from transmission to XYZ.

    W is exactly M^T, so a flat T = 1 (clear film) integrates to WHITE_D65.
    """
    from .cie_data import M

    return M.T


def xyz_white() -> np.ndarray:
    return WHITE_D65.copy()
