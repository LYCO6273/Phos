"""Spectral reconstruction LUT and per-pixel sampling.

The expensive part of the pipeline is the box-constrained QP

    minimize    r^T Q r
    subject to  M r = c
                0 <= r <= 1

with r the 61-band reflectance spectrum and c the pixel XYZ.  This module
solves it once on a 3-D grid of XYZ nodes and stores the result in a LUT.
At run time every pixel only performs one trilinear interpolation.

Feasible nodes (inside the object-colour solid) are solved exactly with an
active-set method.  Infeasible nodes (inside the cube but outside the solid)
fall back to a large-penalty box-QP, which returns the closest smooth
reflectance on the boundary; these nodes are almost never queried by real
pixels.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

from .cie_data import (
    D65,
    DELTA_LAMBDA,
    M,
    N_BANDS,
    Q,
    WHITE_D65,
    WL,
)


CACHE_DIR = Path(__file__).resolve().parents[1] / "Data" / "cache"


# ---------------------------------------------------------------------------
# QP solvers
# ---------------------------------------------------------------------------

_QP_TOL = 1e-8


def _solve_qp_active_set(c: np.ndarray, max_iter: int = 80) -> tuple[np.ndarray, int]:
    """Exact active-set solve of the equality + box QP for a single node."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qinv = np.linalg.inv(Q)
        mqmt_inv = np.linalg.inv(M @ qinv @ M.T)
        r = qinv @ M.T @ (mqmt_inv @ c)
        lo = r < 0.0
        hi = r > 1.0
        r = np.clip(r, 0.0, 1.0)

        for it in range(max_iter):
            free = ~(lo | hi)
            b = r.copy()
            nf = int(free.sum())

            if nf == 0:
                # Everything is pinned to bounds; only feasible if M r == c.
                if np.max(np.abs(M @ b - c)) < _QP_TOL:
                    return b, it
                raise np.linalg.LinAlgError("infeasible node: all variables active")

            qff = Q[np.ix_(free, free)]
            mf = M[:, free]
            qfb = Q[free, :][:, ~free] @ b[~free]
            rhs_c = c - M[:, ~free] @ b[~free]

            kkt = np.zeros((nf + 3, nf + 3))
            kkt[:nf, :nf] = qff
            kkt[:nf, nf:] = mf.T
            kkt[nf:, :nf] = mf
            rhs = np.concatenate([-qfb, rhs_c])
            sol = np.linalg.solve(kkt, rhs)

            r_new = r.copy()
            r_new[free] = sol[:nf]
            nu = sol[nf:]

            changed = False
            new_lo = free & (r_new < -_QP_TOL)
            new_hi = free & (r_new > 1.0 + _QP_TOL)
            if np.any(new_lo):
                r_new[new_lo] = 0.0
                lo |= new_lo
                changed = True
            if np.any(new_hi):
                r_new[new_hi] = 1.0
                hi |= new_hi
                changed = True

            g = Q @ r_new + M.T @ nu
            bad_lo = lo & (g < -_QP_TOL)
            bad_hi = hi & (g > _QP_TOL)
            if not changed and not np.any(bad_lo) and not np.any(bad_hi):
                return r_new, it

            if not changed:
                viol = np.zeros(N_BANDS)
                viol[bad_lo] = -g[bad_lo]
                viol[bad_hi] = g[bad_hi]
                i = int(np.argmax(viol))
                if lo[i]:
                    lo[i] = False
                else:
                    hi[i] = False

            r = r_new
        return r, max_iter


def _soft_admm_fallback(c_batch: np.ndarray,
                        mu: float = 1e8,
                        rho: float = 1e4,
                        iters: int = 150) -> np.ndarray:
    """Large-penalty box-QP for infeasible nodes (vectorized ADMM)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = c_batch.shape[0]
        a_mat = Q + mu * (M.T @ M) + rho * np.eye(N_BANDS)
        a_inv = np.linalg.inv(a_mat)
        rhs_c = mu * (c_batch @ M)
        z = np.zeros((n, N_BANDS))
        v = np.zeros((n, N_BANDS))
        for _ in range(iters):
            r = (rho * (z - v) + rhs_c) @ a_inv.T
            z_new = np.clip(r + v, 0.0, 1.0)
            v = v + r - z_new
            z = z_new
        return z


def solve_spectrum(c: np.ndarray) -> np.ndarray:
    """Return the reconstructed reflectance (61,) for one XYZ triplet."""
    c = np.asarray(c, dtype=np.float64)
    try:
        r, _ = _solve_qp_active_set(c)
    except np.linalg.LinAlgError:
        r = _soft_admm_fallback(c[None, :])[0]
    return r


# ---------------------------------------------------------------------------
# LUT
# ---------------------------------------------------------------------------


class SpectrumLUT:
    """xyz -> reflectance LUT with non-uniform Y spacing."""

    def __init__(self, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray,
                 r_lut: np.ndarray):
        self.x_axis = np.asarray(x_axis, dtype=np.float64)
        self.y_axis = np.asarray(y_axis, dtype=np.float64)
        self.z_axis = np.asarray(z_axis, dtype=np.float64)
        self.r_lut = np.asarray(r_lut, dtype=np.float32)
        self.shape = self.r_lut.shape[:3]

    # -- construction -------------------------------------------------------

    @classmethod
    def build(cls, nx: int = 21, nz: int = 21, ny: int = 33,
              use_cache: bool = True, verbose: bool = False) -> "SpectrumLUT":
        """Build (or load) the shared xyz -> reflectance LUT."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"spectrum_lut_v2_{nx}_{ny}_{nz}.npz"
        if use_cache and cache_path.exists():
            try:
                data = np.load(cache_path)
                return cls(data["x_axis"], data["y_axis"], data["z_axis"], data["r_lut"])
            except (OSError, ValueError):
                pass  # stale/corrupt cache: rebuild below

        x_axis = np.linspace(0.0, WHITE_D65[0], nx)
        z_axis = np.linspace(0.0, WHITE_D65[2], nz)
        # sqrt spacing concentrates nodes in the shadows.
        y_axis = (np.linspace(0.0, 1.0, ny) ** 2) * WHITE_D65[1]

        xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
        nodes = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        t0 = time.time()
        r_lut = np.zeros((nx * ny * nz, N_BANDS), dtype=np.float64)
        infeasible: list[int] = []
        with warnings.catch_warnings():
            # Apple Accelerate emits spurious matmul warnings on macOS; the
            # results are correct, so keep the build output clean.
            warnings.simplefilter("ignore", RuntimeWarning)
            for i, c in enumerate(nodes):
                try:
                    r_lut[i], _ = _solve_qp_active_set(c)
                except np.linalg.LinAlgError:
                    infeasible.append(i)

        if infeasible:
            idx = np.asarray(infeasible, dtype=np.int64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r_lut[idx] = _soft_admm_fallback(nodes[idx])

        if verbose:
            print(f"SpectrumLUT: {nodes.shape[0]} nodes, "
                  f"{len(infeasible)} infeasible, {time.time() - t0:.1f}s")

        obj = cls(x_axis, y_axis, z_axis, r_lut.reshape(nx, ny, nz, N_BANDS).astype(np.float32))
        if use_cache:
            try:
                np.savez_compressed(cache_path, x_axis=x_axis, y_axis=y_axis,
                                    z_axis=z_axis, r_lut=obj.r_lut)
            except OSError:
                pass  # cache is optional
        return obj

    # -- derived LUTs -------------------------------------------------------

    def layer_exposure_lut(self, sensitivity: np.ndarray) -> np.ndarray:
        """Return (nx, ny, nz, n) exposure LUT for n spectral sensitivities.

        exposure_i = sum_lambda S_i(lambda) D65(lambda) r(lambda) dlambda
        """
        sens = np.asarray(sensitivity, dtype=np.float64)
        weights = (sens * D65[None, :] * DELTA_LAMBDA).T  # (61, n)
        return np.tensordot(self.r_lut, weights, axes=([3], [0])).astype(np.float32)

    # -- per-pixel sampling -------------------------------------------------

    def sample(self, lut: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Trilinear sample a (nx, ny, nz, n) LUT at (H, W, 3) XYZ."""
        x = np.asarray(xyz, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        h, w, _ = x.shape
        nx, ny, nz, n_out = lut.shape

        xf = np.clip(x[..., 0] / self.x_axis[-1] * (nx - 1), 0.0, nx - 1)
        zf = np.clip(x[..., 2] / self.z_axis[-1] * (nz - 1), 0.0, nz - 1)
        x0 = np.floor(xf).astype(np.int64)
        x1 = np.minimum(x0 + 1, nx - 1)
        fx = xf - x0
        z0 = np.floor(zf).astype(np.int64)
        z1 = np.minimum(z0 + 1, nz - 1)
        fz = zf - z0

        y = x[..., 1]
        yi = np.searchsorted(self.y_axis, y, side="right") - 1
        yi = np.clip(yi, 0, ny - 2)
        y_lo = self.y_axis[yi]
        y_hi = self.y_axis[yi + 1]
        fy = np.zeros_like(y)
        np.divide(y - y_lo, np.maximum(y_hi - y_lo, 1e-12), out=fy,
                  where=y_hi > y_lo)
        y1i = np.minimum(yi + 1, ny - 1)

        out = np.zeros((h, w, n_out), dtype=np.float32)
        for dx in (0, 1):
            xi = x0 if dx == 0 else x1
            wx = (1.0 - fx) if dx == 0 else fx
            for dy in (0, 1):
                yi_sel = yi if dy == 0 else y1i
                wy = (1.0 - fy) if dy == 0 else fy
                for dz in (0, 1):
                    zi = z0 if dz == 0 else z1
                    wz = (1.0 - fz) if dz == 0 else fz
                    weight = (wx * wy * wz)[..., None]
                    out += weight * lut[xi, yi_sel, zi]
        return out


def normalize_luminance(xyz: np.ndarray):
    """Split XYZ into a unit-luminance colour and a brightness factor.

    Returns (xyz_eff, y_scale) such that:
      - for Y <= 1: xyz_eff == xyz, y_scale == 1
      - for Y > 1 : xyz_eff == xyz / Y (Y=1), y_scale == Y

    Layer exposures are linear in luminance, so after sampling the LUT at
    xyz_eff the caller multiplies by y_scale to preserve >1 highlights.
    """
    y = np.maximum(xyz[..., 1:2], 1e-6)
    over = y > 1.0
    y_scale = np.where(over, y, 1.0).astype(np.float32)
    xyz_eff = np.where(over, xyz / y, xyz).astype(np.float32)
    return xyz_eff, y_scale
