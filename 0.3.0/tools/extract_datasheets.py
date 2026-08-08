#!/usr/bin/env python3
"""
Phos 0.3.0 - Datasheet curve digitizer.

Extracts characteristic / spectral-sensitivity / dye-density curves from the
supplied manufacturer PDFs and writes them as CSV files under Phos/Data.

The chart calibration values below (axis rectangles, tick positions) were
derived by inspecting the PDF vector objects with pdfplumber. They are
documented per block so they can be re-verified if the datasheets change.

Usage:
    python tools/extract_datasheets.py \
        --hp5 "/path/HP5 Technical.pdf" \
        --gold "/path/Kodak_Gold_200.pdf" \
        --vision "/path/VISION-200T-Technical-Data_zh-CN.pdf"

Requires: pdfplumber, numpy
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pdfplumber


ROOT = Path(__file__).resolve().parents[1] / "Phos" / "Data"


def write_csv(path: Path, header: list[str], rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows((round(a, 6), round(b, 6)) for a, b in rows)
    print(f"  wrote {path.name}: {len(rows)} points")


def select_curves(page, x0: float, x1: float, y0: float, y1: float,
                  min_height: float = 8.0, min_pts: int = 8):
    """Return page curves whose bbox lies in the given chart region."""
    out = []
    for c in page.curves:
        if c["x0"] >= x0 - 3 and c["x1"] <= x1 + 3 and c["top"] >= y0 - 3 and c["bottom"] <= y1 + 3:
            if (c["bottom"] - c["top"]) >= min_height and len(c.get("pts", [])) >= min_pts:
                out.append(c)
    return out


def extract_hp5(pdf_path: Path) -> None:
    print("HP5 Technical.pdf")
    dst = ROOT / "hp5"
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[4]  # page 5: characteristic curves (lower chart)
        cands = [c for c in page.curves
                 if c["x0"] > 300 and c["x1"] < 545
                 and (c["bottom"] - c["top"]) > 40
                 and len(c.get("pts", [])) >= 20]
        if not cands:
            raise RuntimeError("HP5 characteristic curve not found")
        curve = max(cands, key=lambda c: len(c["pts"]))
        rows = [((x - 326.5) / 41.1, (435.5 - y) / 40.6) for x, y in curve["pts"]]
        write_csv(dst / "hp5_char_curve.csv", ["rel_log_exposure", "density"], rows)


def extract_gold(pdf_path: Path) -> None:
    print("Kodak Gold 200.pdf")
    dst = ROOT / "gold200"
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[3]  # page 4: CURVES

        # --- characteristic curves (top-left chart) ---
        # Inner axes: x 81.7..266.2, y 101.3..285.8.
        # X: '-1.0' at x=81.7, '0.0' at 127.8, '1.0' at 173.93, '2.0' at 220.06, '3.0' at 266.2
        # Y: '4.0' label center y~101.7 -> unit=46.125pt, 0.0 at y=285.8
        curves = select_curves(page, 45, 300, 90, 315, min_height=40, min_pts=15)
        # B (yellow-forming) sits highest on the page, then G, then R.
        curves.sort(key=lambda c: c["top"])
        for name, match in zip(["b", "g", "r"], curves[:3]):
            rows = [(-1.0 + (x - 81.7) / 46.13, (285.8 - y) / 46.125) for x, y in match["pts"]]
            write_csv(dst / f"gold200_char_curve_{name.lower()}.csv", ["rel_log_exposure", "density"], rows)

        # --- spectral dye density / neutral & D-min (top-right chart) ---
        # Inner axes: x 357.1..541.4, y 102.9..287.2.
        # X: 400 at 357.1, 700 at 541.4; Y: 2.5 at 102.9, 0.0 at 287.2.
        curves = select_curves(page, 328, 560, 80, 315, min_height=30, min_pts=50)
        curves.sort(key=lambda c: len(c["pts"]), reverse=True)
        midscale, dmin = curves[0], curves[1]
        rows = [((x - 357.1) / (541.4 - 357.1) * 300 + 400, (287.2 - y) / 73.72) for x, y in midscale["pts"]]
        write_csv(dst / "gold200_neutral_density.csv", ["wavelength_nm", "density"], rows)
        rows = [((x - 357.1) / (541.4 - 357.1) * 300 + 400, (287.2 - y) / 73.72) for x, y in dmin["pts"]]
        write_csv(dst / "gold200_dmin.csv", ["wavelength_nm", "density"], rows)


def extract_vision(pdf_path: Path) -> None:
    print("VISION 200T (zh-CN).pdf")
    dst = ROOT / "vision200t"
    with pdfplumber.open(str(pdf_path)) as pdf:
        # --- characteristic curves (page 4, right chart) ---
        page = pdf.pages[3]
        # Inner axes: x 357.5..542.1, y 121.0..305.5.
        # X: -3.684 at 357.5, 1.116 at 542.1; Y: 3.0 at 121.0, 0.0 at 305.5.
        curves = select_curves(page, 330, 580, 100, 330, min_height=20, min_pts=12)
        curves = [c for c in curves if 350 < c["x0"] < 560]
        bboxes = {"b": 138.9, "g": 150.0, "r": 193.3}
        for name, ytop in bboxes.items():
            match = min(curves, key=lambda c: abs(c["top"] - ytop))
            rows = [(-3.684 + (x - 357.5) / (542.1 - 357.5) * 4.8, (305.5 - y) / 61.5)
                    for x, y in match["pts"]]
            write_csv(dst / f"vision200t_char_curve_{name}.csv", ["log_exposure_lux_s", "density"], rows)

        # --- spectral sensitivity curves (page 4, bottom chart) ---
        # X: 250nm at x~350.3, 750nm at x~550.9 (20.06pt/50nm)
        # Y: logS 3.0 at y~510.1, 0.0 at y~623.1 (37.7pt/unit)
        curves = select_curves(page, 330, 580, 480, 700, min_height=15, min_pts=10)
        # Drop closed legend swatches (filled paths); keep open data curves.
        curves = [c for c in curves
                  if c["x0"] > 380 and c["x1"] < 560
                  and c.get("non_stroking_color") != 1.0
                  and c["pts"][0] != c["pts"][-1]]
        curves.sort(key=lambda c: c["x0"])
        names = ["b", "g", "r"]
        if len(curves) < 3:
            raise RuntimeError(f"Vision spectral sensitivity: expected 3 curves, got {len(curves)}")
        for name, curve in zip(names, curves[:3]):
            rows = [(250 + (x - 350.3) / 0.4012, (623.1 - y) / 37.7) for x, y in curve["pts"]]
            write_csv(dst / f"vision200t_sensitivity_{name}.csv", ["wavelength_nm", "log_sensitivity"], rows)

        # --- spectral dye density curves (page 5) ---
        page = pdf.pages[4]
        # Inner axes: x 89.9..274.2, y 111.9..296.3.
        # X: 400 at 89.9, 800 at 274.2; Y: 1.8 at 111.9, -0.2 at 296.3 (92.2pt/unit).
        curves = select_curves(page, 54, 281, 60, 340, min_height=20, min_pts=30)
        def peak_nm(c):
            x, _ = min(c["pts"], key=lambda p: p[1])
            return 400 + (x - 89.9) / (274.2 - 89.9) * 400
        by_peak = sorted(curves, key=peak_nm)
        # Expected: yellow ~450, magenta ~540, cyan ~680 (all peak-normalized at ~1.0).
        # There are also two "overall" curves peaking near 440nm (midscale neutral ~1.6,
        # D-min ~0.85); the yellow dye curve is the ~450nm one whose peak density is ~1.0.
        def peak_density(c):
            _, y = min(c["pts"], key=lambda p: p[1])
            return (296.3 - y) / 92.2 - 0.2
        near450 = [c for c in by_peak if 430 <= peak_nm(c) <= 470]
        yellow = min(near450, key=lambda c: abs(peak_density(c) - 1.0))
        magenta = next(c for c in by_peak if 520 <= peak_nm(c) <= 570)
        cyan = next(c for c in by_peak if 650 <= peak_nm(c) <= 710)
        for name, curve in [("yellow", yellow), ("magenta", magenta), ("cyan", cyan)]:
            rows = [(400 + (x - 89.9) / (274.2 - 89.9) * 400, (296.3 - y) / 92.2 - 0.2)
                    for x, y in curve["pts"]]
            write_csv(dst / f"vision200t_dye_{name}.csv", ["wavelength_nm", "density"], rows)
        # D-min / orange-mask curve: peak ~440nm, density < 1.0 (the other ~440nm curve).
        dmin = min((c for c in near450 if c is not yellow), key=peak_density)
        rows = [(400 + (x - 89.9) / (274.2 - 89.9) * 400, (296.3 - y) / 92.2 - 0.2)
                for x, y in dmin["pts"]]
        write_csv(dst / "vision200t_dmin.csv", ["wavelength_nm", "density"], rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp5", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--vision", required=True)
    args = ap.parse_args()

    extract_hp5(Path(args.hp5))
    extract_gold(Path(args.gold))
    extract_vision(Path(args.vision))


if __name__ == "__main__":
    main()
