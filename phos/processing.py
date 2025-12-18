from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
import zlib
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from phos.grain import grain
from phos.presets import FILM_TYPES, film_choose

try:
    import rawpy  # type: ignore
except Exception:
    rawpy = None


RAW_EXTENSIONS = {
    "3fr",
    "arw",
    "cr2",
    "cr3",
    "dcr",
    "dng",
    "erf",
    "kdc",
    "mos",
    "mrw",
    "nef",
    "nrw",
    "orf",
    "pef",
    "raf",
    "raw",
    "rw2",
    "rwl",
    "srw",
}


@dataclass(frozen=True)
class ProcessingOptions:
    film_type: str = "NC200"
    tone_style: str = "filmic"
    grain_enabled: bool = True
    grain_strength: float = 1.0
    grain_size: float = 1.0
    jpeg_quality: int = 95


@dataclass(frozen=True)
class ProcessResult:
    film_rgb: np.ndarray
    jpeg_bytes: bytes
    output_filename: str
    process_time_s: float


def _ext_lower(filename: str) -> str:
    return os.path.splitext(filename)[1].lstrip(".").lower()


def decode_bytes_to_bgr(file_bytes: bytes, filename: str) -> np.ndarray:
    file_ext = _ext_lower(filename)
    if file_ext in RAW_EXTENSIONS:
        if rawpy is None:
            raise RuntimeError("rawpy 未安装：请先安装 rawpy/libraw 才能解析 RAW。")
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}") as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            with rawpy.imread(temp_file.name) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if bgr is None:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return bgr


def encode_jpeg_bytes(image_rgb: np.ndarray, quality: int = 95) -> bytes:
    film_pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    film_pil.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def make_zip_bytes(named_files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, data in named_files:
            zf.writestr(filename, data)
    return buf.getvalue()


def standardize(image_bgr: np.ndarray, min_size: int = 3000) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if height < width:
        scale_factor = min_size / height
        new_height = min_size
        new_width = int(width * scale_factor)
    else:
        scale_factor = min_size / width
        new_width = min_size
        new_height = int(height * scale_factor)

    new_width = new_width + 1 if new_width % 2 != 0 else new_width
    new_height = new_height + 1 if new_height % 2 != 0 else new_height
    interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(image_bgr, (new_width, new_height), interpolation=interpolation)


def luminance(image_bgr, color_type, r_r, r_g, r_b, g_r, g_g, g_b, b_r, b_g, b_b, t_r, t_g, t_b):
    b, g, r = cv2.split(image_bgr)
    b_float = b.astype(np.float32) / 255.0
    g_float = g.astype(np.float32) / 255.0
    r_float = r.astype(np.float32) / 255.0

    if color_type == ("color"):
        lux_r = r_r * r_float + r_g * g_float + r_b * b_float
        lux_g = g_r * r_float + g_g * g_float + g_b * b_float
        lux_b = b_r * r_float + b_g * g_float + b_b * b_float
        lux_total = t_r * r_float + t_g * g_float + t_b * b_float
    else:
        lux_total = t_r * r_float + t_g * g_float + t_b * b_float
        lux_r = None
        lux_g = None
        lux_b = None

    return lux_r, lux_g, lux_b, lux_total


def average(lux_total: np.ndarray) -> float:
    avg_lux = float(np.mean(lux_total))
    return float(np.clip(avg_lux, 0, 1))


def reinhard(lux_r, lux_g, lux_b, lux_total, color_type, gamma: float):
    if color_type == "color":
        mapped = lux_r
        mapped = mapped * (mapped / (1.0 + mapped))
        mapped = np.power(mapped, 1.05 / gamma)
        result_r = np.clip(mapped, 0, 1)

        mapped = lux_g
        mapped = mapped * (mapped / (1.0 + mapped))
        mapped = np.power(mapped, 1.05 / gamma)
        result_g = np.clip(mapped, 0, 1)

        mapped = lux_b
        mapped = mapped * (mapped / (1.0 + mapped))
        mapped = np.power(mapped, 1.05 / gamma)
        result_b = np.clip(mapped, 0, 1)
        result_total = None
    else:
        mapped = lux_total
        mapped = mapped * (mapped / (1.0 + mapped))
        mapped = np.power(mapped, 1.0 / gamma)
        result_total = np.clip(mapped, 0, 1)
        result_r = None
        result_g = None
        result_b = None

    return result_r, result_g, result_b, result_total


def filmic(lux_r, lux_g, lux_b, lux_total, color_type, gamma, A, B, C, D, E, F):
    if color_type == ("color"):
        lux_r = np.maximum(lux_r, 0)
        lux_g = np.maximum(lux_g, 0)
        lux_b = np.maximum(lux_b, 0)

        lux_r = 10 * (lux_r**gamma)
        lux_g = 10 * (lux_g**gamma)
        lux_b = 10 * (lux_b**gamma)

        result_r = ((lux_r * (A * lux_r + C * B) + D * E) / (lux_r * (A * lux_r + B) + D * F)) - E / F
        result_g = ((lux_g * (A * lux_g + C * B) + D * E) / (lux_g * (A * lux_g + B) + D * F)) - E / F
        result_b = ((lux_b * (A * lux_b + C * B) + D * E) / (lux_b * (A * lux_b + B) + D * F)) - E / F
        result_total = None
    else:
        lux_total = np.maximum(lux_total, 0)
        lux_total = 10 * (lux_total**gamma)
        result_r = None
        result_g = None
        result_b = None
        result_total = ((lux_total * (A * lux_total + C * B) + D * E) / (lux_total * (A * lux_total + B) + D * F)) - E / F

    return result_r, result_g, result_b, result_total


def opt(
    lux_r,
    lux_g,
    lux_b,
    lux_total,
    color_type,
    sens_factor,
    d_r,
    l_r,
    x_r,
    n_r,
    d_g,
    l_g,
    x_g,
    n_g,
    d_b,
    l_b,
    x_b,
    n_b,
    d_l,
    l_l,
    x_l,
    n_l,
    grain_enabled: bool,
    grain_size: float,
    grain_seed: int,
    gamma,
    A,
    B,
    C,
    D,
    E,
    F,
    tone_style: str,
):
    avrl = average(lux_total)
    sens = (1.0 - avrl) * 0.75 + 0.10
    sens = np.clip(sens, 0.10, 0.7)
    strg = 23 * sens**2 * sens_factor
    rads = np.clip(int(20 * sens**2 * sens_factor), 1, 50)
    base = 0.05 * sens_factor

    ksize = rads * 2 + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    if color_type == ("color"):
        weights = (base + lux_r**2) * sens
        weights = np.clip(weights, 0, 1)
        bloom_layer = cv2.GaussianBlur(lux_r * weights, (ksize * 3, ksize * 3), sens * 55)
        bloom_effect_r = (bloom_layer * weights * strg) / (1.0 + (bloom_layer * weights * strg))

        weights = (base + lux_g**2) * sens
        weights = np.clip(weights, 0, 1)
        bloom_layer = cv2.GaussianBlur(lux_g * weights, (ksize * 2 + 1, ksize * 2 + 1), sens * 35)
        bloom_effect_g = (bloom_layer * weights * strg) / (1.0 + (bloom_layer * weights * strg))

        weights = (base + lux_b**2) * sens
        weights = np.clip(weights, 0, 1)
        bloom_layer = cv2.GaussianBlur(lux_b * weights, (ksize, ksize), sens * 15)
        bloom_effect_b = (bloom_layer * weights * strg) / (1.0 + (bloom_layer * weights * strg))

        if grain_enabled:
            weighted_noise_r, weighted_noise_g, weighted_noise_b, _ = grain(
                lux_r, lux_g, lux_b, lux_total, color_type, float(sens), float(grain_size), int(grain_seed)
            )
            lux_r = bloom_effect_r * d_r + (lux_r**x_r) * l_r + weighted_noise_r * n_r + weighted_noise_g * n_l + weighted_noise_b * n_l
            lux_g = bloom_effect_g * d_g + (lux_g**x_g) * l_g + weighted_noise_r * n_l + weighted_noise_g * n_g + weighted_noise_b * n_l
            lux_b = bloom_effect_b * d_b + (lux_b**x_b) * l_b + weighted_noise_r * n_l + weighted_noise_g * n_l + weighted_noise_b * n_b
        else:
            lux_r = bloom_effect_r * d_r + (lux_r**x_r) * l_r
            lux_g = bloom_effect_g * d_g + (lux_g**x_g) * l_g
            lux_b = bloom_effect_b * d_b + (lux_b**x_b) * l_b

        if tone_style == "filmic":
            result_r, result_g, result_b, _ = filmic(lux_r, lux_g, lux_b, lux_total, color_type, gamma, A, B, C, D, E, F)
        else:
            result_r, result_g, result_b, _ = reinhard(lux_r, lux_g, lux_b, lux_total, color_type, gamma)

        combined_b = (result_b * 255).astype(np.uint8)
        combined_g = (result_g * 255).astype(np.uint8)
        combined_r = (result_r * 255).astype(np.uint8)
        return cv2.merge([combined_r, combined_g, combined_b])  # RGB order

    weights = (base + lux_total**2) * sens
    weights = np.clip(weights, 0, 1)
    bloom_layer = cv2.GaussianBlur(lux_total * weights, (ksize * 3, ksize * 3), sens * 55)
    bloom_effect = (bloom_layer * weights * strg) / (1.0 + (bloom_layer * weights * strg))

    if grain_enabled:
        _, _, _, weighted_noise_total = grain(
            lux_r, lux_g, lux_b, lux_total, color_type, float(sens), float(grain_size), int(grain_seed)
        )
        lux_total = bloom_effect * d_l + (lux_total**x_l) * l_l + weighted_noise_total * n_l
    else:
        lux_total = bloom_effect * d_l + (lux_total**x_l) * l_l

    if tone_style == "filmic":
        _, _, _, result_total = filmic(lux_r, lux_g, lux_b, lux_total, color_type, gamma, A, B, C, D, E, F)
    else:
        _, _, _, result_total = reinhard(lux_r, lux_g, lux_b, lux_total, color_type, gamma)

    return (result_total * 255).astype(np.uint8)


def process_bytes(file_bytes: bytes, filename: str, options: ProcessingOptions) -> ProcessResult:
    if options.film_type not in FILM_TYPES:
        raise ValueError(f"Unknown film_type: {options.film_type}")

    start_time = time.time()

    image_bgr = decode_bytes_to_bgr(file_bytes, filename)
    grain_seed = zlib.crc32(file_bytes)

    (
        r_r,
        r_g,
        r_b,
        g_r,
        g_g,
        g_b,
        b_r,
        b_g,
        b_b,
        t_r,
        t_g,
        t_b,
        color_type,
        sens_factor,
        d_r,
        l_r,
        x_r,
        n_r,
        d_g,
        l_g,
        x_g,
        n_g,
        d_b,
        l_b,
        x_b,
        n_b,
        d_l,
        l_l,
        x_l,
        n_l,
        gamma,
        A,
        B,
        C,
        D,
        E,
        F,
    ) = film_choose(options.film_type)

    if options.grain_enabled:
        strength = float(options.grain_strength)
        n_r *= strength
        n_g *= strength
        n_b *= strength
        n_l *= strength
    else:
        n_r = n_g = n_b = n_l = 0.0

    image_bgr = standardize(image_bgr)
    lux_r, lux_g, lux_b, lux_total = luminance(image_bgr, color_type, r_r, r_g, r_b, g_r, g_g, g_b, b_r, b_g, b_b, t_r, t_g, t_b)

    film_rgb = opt(
        lux_r,
        lux_g,
        lux_b,
        lux_total,
        color_type,
        sens_factor,
        d_r,
        l_r,
        x_r,
        n_r,
        d_g,
        l_g,
        x_g,
        n_g,
        d_b,
        l_b,
        x_b,
        n_b,
        d_l,
        l_l,
        x_l,
        n_l,
        bool(options.grain_enabled),
        float(options.grain_size),
        int(grain_seed),
        gamma,
        A,
        B,
        C,
        D,
        E,
        F,
        str(options.tone_style),
    )

    jpeg_bytes = encode_jpeg_bytes(film_rgb, quality=options.jpeg_quality)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_stem = os.path.splitext(os.path.basename(filename or "image"))[0]
    output_filename = f"{input_stem}_phos_{options.film_type}_{timestamp}.jpg"
    process_time_s = time.time() - start_time

    return ProcessResult(
        film_rgb=film_rgb,
        jpeg_bytes=jpeg_bytes,
        output_filename=output_filename,
        process_time_s=process_time_s,
    )


def process_uploaded_file(uploaded_file, options: ProcessingOptions) -> ProcessResult:
    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    filename = getattr(uploaded_file, "name", "image")
    return process_bytes(file_bytes=file_bytes, filename=filename, options=options)

