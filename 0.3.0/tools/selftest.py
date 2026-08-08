#!/usr/bin/env python3
"""Phos 0.3.0 self test.

Run from the Phos directory (or with PYTHONPATH set to Phos):

    python tools/selftest.py

Checks:
  1. CIE tables are self-consistent (M @ ones == D65 white).
  2. The active-set QP reconstructs exact spectra for feasible colours.
  3. The spectrum LUT satisfies M r ~= c on a sample of feasible nodes.
  4. Trilinear sampling reproduces a constant field exactly.
  5. All three film models run on a synthetic patch chart and produce
     sane values (neutral gray near mid, black near 0, white bright,
     saturated primaries resolved).
  6. Timing is reported so regressions are easy to spot.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "Phos"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core.cie_data import D65, DELTA_LAMBDA, M, WHITE_D65  # noqa: E402
from Core.color import srgb_to_xyz  # noqa: E402
from Core.spectral import SpectrumLUT, solve_spectrum  # noqa: E402


FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    print("== 1. CIE consistency")
    white = M @ np.ones(61)
    check("M @ 1 == D65 white", np.allclose(white, WHITE_D65, atol=1e-9),
          str(np.round(white, 6)))

    print("== 2. QP solver exactness (feasible colours)")
    targets = [
        0.18 * WHITE_D65,
        np.array([0.412, 0.213, 0.019]),   # sRGB red
        np.array([0.358, 0.715, 0.119]),   # sRGB green
        np.array([0.180, 0.073, 0.950]),   # sRGB blue
        WHITE_D65,
    ]
    for i, c in enumerate(targets):
        r = solve_spectrum(c)
        res = float(np.max(np.abs(M @ r - c)))
        check(f"reconstruct #{i} residual < 1e-6", res < 1e-6, f"res={res:.2e}")
        check(f"reconstruct #{i} inside [0,1]", float(r.min()) >= -1e-8 and float(r.max()) <= 1 + 1e-8,
              f"[{r.min():.3f},{r.max():.3f}]")

    print("== 3. Spectrum LUT")
    t0 = time.time()
    lut = SpectrumLUT.build(verbose=False)
    print(f"  LUT ready in {time.time() - t0:.1f}s, shape {lut.r_lut.shape}")
    # Sample a few feasible nodes directly from the LUT and check residuals.
    probes = np.array(targets)
    xyz_eff = probes.astype(np.float32)[:, None, :]
    # nearest-neighbour sampling through the exposure LUT of a flat
    # sensitivity is equivalent to sampling the reflectance itself.
    flat = np.ones((1, 61), dtype=np.float64)
    flat = flat / float(np.sum(D65 * DELTA_LAMBDA))  # normalize like films
    flat_lut = lut.layer_exposure_lut(flat)[..., 0]
    sampled = lut.sample(flat_lut[..., None], xyz_eff)[..., 0]
    check("LUT sampling shape", sampled.shape == (len(probes), 1),
          str(sampled.shape))
    for i, c in enumerate(probes):
        r_exact = solve_spectrum(c)
        e_exact = float(np.sum(D65 * DELTA_LAMBDA * r_exact) / np.sum(D65 * DELTA_LAMBDA))
        check(f"LUT vs exact solve #{i} < 5e-2",
              abs(float(sampled[i, 0]) - e_exact) < 0.05,
              f"LUT={sampled[i, 0]:.3f} exact={e_exact:.3f}")

    print("== 4. Trilinear interpolation exactness")
    const = np.full(lut.r_lut.shape, 0.37, dtype=np.float32)
    h, w = 33, 41
    xyz = np.random.default_rng(0).random((h, w, 3), dtype=np.float32)
    xyz[..., 0] *= 0.9
    xyz[..., 1] *= 0.9
    xyz[..., 2] *= 1.0
    out = lut.sample(const, xyz)
    check("constant field sampled exactly", np.allclose(out, 0.37, atol=1e-5),
          f"max err={np.max(np.abs(out - 0.37)):.2e}")

    print("== 5. Film models on synthetic patches")
    import Films.HP5.HP5 as hp5
    import Films.Gold200.Gold200 as gold
    import Films.Vision200T.Vision200T as vision

    size = 360
    n = 6
    patch = size // n
    xyz_img = np.zeros((size, size, 3), dtype=np.float32)
    patches = {
        "black": np.zeros(3, np.float32),
        "gray18": np.full(3, 0.18, np.float32),
        "red": np.array([1, 0, 0], np.float32),
        "green": np.array([0, 1, 0], np.float32),
        "blue": np.array([0, 0, 1], np.float32),
        "white": np.ones(3, np.float32),
    }
    for i, (name, rgb) in enumerate(patches.items()):
        xyz_img[i * patch:(i + 1) * patch, :, :] = srgb_to_xyz(rgb).astype(np.float32)

    for label, mod in [("HP5", hp5), ("Gold200", gold), ("Vision200T", vision)]:
        t0 = time.time()
        out = mod.process(xyz_img, lut, grain_style="不使用", exposure_ev=0.0)
        dt = time.time() - t0
        check(f"{label} runs", out.dtype == np.uint8 and out.size > 0, f"{dt:.2f}s")
        check(f"{label} no NaN", not np.isnan(out.astype(np.float32)).any())

        def crop(i):
            return out[i * patch + patch // 4:(i + 1) * patch - patch // 4,
                      patch // 4:3 * patch // 4]

        black_m = float(np.mean(crop(0)))
        gray_m = float(np.mean(crop(1)))
        white_m = float(np.mean(crop(5)))
        check(f"{label} black dark", black_m < 40, f"{black_m:.1f}")
        check(f"{label} gray mid", 80 < gray_m < 180, f"{gray_m:.1f}")
        check(f"{label} white bright", white_m > 120, f"{white_m:.1f}")
        if label != "HP5":
            red_m = crop(2).mean(axis=(0, 1))
            green_m = crop(3).mean(axis=(0, 1))
            blue_m = crop(4).mean(axis=(0, 1))
            check(f"{label} red resolved", red_m[0] > red_m[1] + 20,
                  str(np.round(red_m, 0)))
            check(f"{label} green resolved", green_m[1] > green_m[0] + 20,
                  str(np.round(green_m, 0)))
            check(f"{label} blue resolved", blue_m[2] > blue_m[1] + 30,
                  str(np.round(blue_m, 0)))
        else:
            check(f"{label} luminance order",
                  float(np.mean(crop(3))) > float(np.mean(crop(2))) > float(np.mean(crop(4))),
                  f"green={np.mean(crop(3)):.0f} red={np.mean(crop(2)):.0f} blue={np.mean(crop(4)):.0f}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
