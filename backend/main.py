from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from phos.presets import FILM_DESCRIPTIONS, FILM_META, FILM_TYPES
from phos.processing import ProcessingOptions, make_zip_bytes, process_bytes

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(APP_DIR, "..", "frontend"))

app = FastAPI(title="Phos API", version="0.2.0")


@app.get("/api/presets")
def presets():
    return {
        "film_types": FILM_TYPES,
        "film_descriptions": FILM_DESCRIPTIONS,
        "film_meta": {code: meta.as_dict() for code, meta in FILM_META.items()},
        "default_film_type": "FUJI200",
        "tone_styles": ["filmic", "reinhard"],
    }


@app.post("/api/process")
async def process_one(
    file: Annotated[UploadFile, File(...)],
    film_type: Annotated[str, Form()] = "FUJI200",
    tone_style: Annotated[str, Form()] = "filmic",
    grain_enabled: Annotated[bool, Form()] = True,
    grain_strength: Annotated[float, Form()] = 1.0,
    grain_size: Annotated[float, Form()] = 1.0,
    jpeg_quality: Annotated[int, Form()] = 95,
):
    try:
        file_bytes = await file.read()
        result = process_bytes(
            file_bytes=file_bytes,
            filename=file.filename or "image",
            options=ProcessingOptions(
                film_type=film_type,
                tone_style=tone_style,
                grain_enabled=grain_enabled,
                grain_strength=grain_strength,
                grain_size=grain_size,
                jpeg_quality=jpeg_quality,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=result.jpeg_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{result.output_filename}"'},
    )


@app.post("/api/batch")
async def process_batch(
    files: Annotated[list[UploadFile], File(...)],
    film_type: Annotated[str, Form()] = "FUJI200",
    tone_style: Annotated[str, Form()] = "filmic",
    grain_enabled: Annotated[bool, Form()] = True,
    grain_strength: Annotated[float, Form()] = 1.0,
    grain_size: Annotated[float, Form()] = 1.0,
    jpeg_quality: Annotated[int, Form()] = 95,
):
    outputs: list[tuple[str, bytes]] = []
    options = ProcessingOptions(
        film_type=film_type,
        tone_style=tone_style,
        grain_enabled=grain_enabled,
        grain_strength=grain_strength,
        grain_size=grain_size,
        jpeg_quality=jpeg_quality,
    )

    for f in files:
        try:
            file_bytes = await f.read()
            result = process_bytes(file_bytes=file_bytes, filename=f.filename or "image", options=options)
            outputs.append((result.output_filename, result.jpeg_bytes))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {exc}") from exc

    zip_bytes = make_zip_bytes(outputs)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="phos_batch.zip"'},
    )


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
