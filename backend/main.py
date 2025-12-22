from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from phos.presets import FILM_DESCRIPTIONS, FILM_META, FILM_TYPES
from phos.processing import ProcessingOptions, make_zip_bytes, process_bytes

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(APP_DIR, "..", "frontend"))

app = FastAPI(title="Phos API", version="0.2.0")

_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, object]] = {}


def _job_cleanup(max_age_s: float = 3600.0, max_count: int = 256) -> None:
    now = time.time()
    with _jobs_lock:
        stale = [job_id for job_id, job in _jobs.items() if now - float(job.get("updated_at", now)) > max_age_s]
        for job_id in stale:
            _jobs.pop(job_id, None)

        if len(_jobs) <= max_count:
            return

        oldest = sorted(_jobs.items(), key=lambda kv: float(kv[1].get("updated_at", 0.0)))
        for job_id, _ in oldest[: max(0, len(_jobs) - max_count)]:
            _jobs.pop(job_id, None)


def _job_update(job_id: str, **updates: object) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def _job_add_warning(job_id: str, warning: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        warnings = job.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            job["warnings"] = warnings
        msg = str(warning).strip()
        if msg and msg not in warnings:
            warnings.append(msg)
        job["updated_at"] = time.time()


async def _run_job(job_id: str, *, file_bytes: bytes, filename: str, options: ProcessingOptions) -> None:
    _job_update(job_id, status="processing", progress=0.0, message="开始冲洗…")

    def progress_cb(pct: float, message: str) -> None:
        _job_update(job_id, progress=float(pct), message=str(message))

    def warning_cb(message: str) -> None:
        _job_add_warning(job_id, str(message))

    try:
        result = await asyncio.to_thread(
            process_bytes,
            file_bytes,
            filename,
            options,
            progress_cb,
            warning_cb,
        )
    except Exception as exc:
        _job_update(job_id, status="error", message="冲洗失败", error=str(exc), progress=0.0)
        return

    _job_update(
        job_id,
        status="done",
        message="冲洗完成",
        progress=100.0,
        output_filename=result.output_filename,
        jpeg_bytes=result.jpeg_bytes,
    )


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


@app.post("/api/jobs")
async def create_job(
    file: Annotated[UploadFile, File(...)],
    film_type: Annotated[str, Form()] = "FUJI200",
    tone_style: Annotated[str, Form()] = "filmic",
    grain_enabled: Annotated[bool, Form()] = True,
    grain_strength: Annotated[float, Form()] = 1.0,
    grain_size: Annotated[float, Form()] = 1.0,
    jpeg_quality: Annotated[int, Form()] = 95,
):
    _job_cleanup()

    job_id = uuid4().hex
    filename = file.filename or "image"
    file_bytes = await file.read()
    options = ProcessingOptions(
        film_type=film_type,
        tone_style=tone_style,
        grain_enabled=grain_enabled,
        grain_strength=grain_strength,
        grain_size=grain_size,
        jpeg_quality=jpeg_quality,
    )

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "等待开始",
            "error": None,
            "warnings": [],
            "output_filename": None,
            "jpeg_bytes": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    asyncio.create_task(_run_job(job_id, file_bytes=file_bytes, filename=filename, options=options))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "progress": job.get("progress", 0.0),
            "message": job.get("message"),
            "error": job.get("error"),
            "warnings": job.get("warnings") or [],
            "output_filename": job.get("output_filename"),
        }


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        status = job.get("status")
        if status == "error":
            raise HTTPException(status_code=400, detail=str(job.get("error") or "processing failed"))
        if status != "done":
            raise HTTPException(status_code=425, detail="job not ready")
        jpeg_bytes = job.get("jpeg_bytes")
        output_filename = job.get("output_filename") or f"{job_id}.jpg"
        if not isinstance(jpeg_bytes, (bytes, bytearray)):
            raise HTTPException(status_code=500, detail="result missing")

    return Response(
        content=bytes(jpeg_bytes),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{output_filename}"'},
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
