"""CIE standard data and precomputed spectral-reconstruction matrices.

The tables in Data/cie were generated from the public CIE 1931 2 degree
standard observer (1 nm, band-averaged to 5 nm) and the CIE D65 relative SPD
(5 nm), sourced from the colour-science project's BSD-licensed data files
(which in turn follow CIE 15:2004 / ASTM E308).

The 400-700 nm truncation makes the integrated D65 white point
(0.94939, 1.0, 1.08706) slightly differ from the canonical sRGB D65 white
(0.95047, 1.0, 1.08883).  To keep every stage self-consistent, the M matrix is
row-scaled so that a flat reflectance r = 1 maps *exactly* to the sRGB D65
white point used by the colour conversions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "Data" / "cie"

# Wavelength grid shared by the whole pipeline: 400..700 nm, 5 nm step.
WL = np.arange(400, 701, 5, dtype=np.float64)
DELTA_LAMBDA = 5.0
N_BANDS = len(WL)  # 61

# Canonical sRGB D65 white point (Y = 1).
WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _load_csv(name: str) -> np.ndarray:
    path = DATA_DIR / name
    return np.loadtxt(path, delimiter=",", skiprows=1)


def load_cmf() -> np.ndarray:
    """Return the 5 nm CIE 1931 2 deg CMF table as (61, 3)."""
    return _load_csv("cie1931_2deg_5nm.csv")[:, 1:4]


def load_d65() -> np.ndarray:
    """Return the D65 relative SPD on WL (61,)."""
    return _load_csv("d65_5nm.csv")[:, 1]


CMF = load_cmf()
D65 = load_d65()


def _second_difference(n: int) -> np.ndarray:
    """(n-2, n) second-difference operator, rows 1,-2,1."""
    d = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        d[i, i] = 1.0
        d[i, i + 1] = -2.0
        d[i, i + 2] = 1.0
    return d


def build_matrices(smoothness: float = 10.0, ridge: float = 1e-6):
    """Build the forward matrix M and the smoothness Hessian Q.

    M (3 x 61):  c = M @ r,  c = (X, Y, Z),  r = reflectance spectrum.
    The rows are scaled so M @ ones == WHITE_D65 (flat perfect reflector).

    Q (61 x 61): smoothness Hessian, Q = smoothness * D^T D + ridge * I.
    """
    a = CMF * D65[:, None] * DELTA_LAMBDA          # (61, 3) weighted CMF
    m = a.T                                        # (3, 61)
    m = m / (m @ np.ones(N_BANDS))[:, None] * WHITE_D65[:, None]

    d = _second_difference(N_BANDS)                # (59, 61)
    # einsum avoids a spurious "divide by zero in matmul" warning emitted by
    # Apple Accelerate's BLAS for this particular sparse product on macOS.
    q = smoothness * np.einsum("ij,ik->jk", d, d) + ridge * np.eye(N_BANDS)
    return m, q


M, Q = build_matrices()
