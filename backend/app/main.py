"""FastAPI service: upload MRI studies, run analysis, serve volumes and masks."""

from __future__ import annotations

import gzip
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import demo, models
from .brain import BrainParams, analyze_brain
from .loaders import LoadError, load_files
from .spine import SpineParams, analyze_spine
from .volume import Volume, from_nifti_image

MAX_STUDIES = 6
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024

DISCLAIMER = (
    "Research prototype. Outputs are candidate findings to direct radiologist review; "
    "they are not diagnoses and are not validated for clinical use."
)


@dataclass
class Study:
    volume: Volume
    mask: np.ndarray | None = None
    result: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class StudyStore:
    """Small in-memory LRU. Nothing is written to disk."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: OrderedDict[str, Study] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, vol: Volume) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._items[sid] = Study(vol)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return sid

    def get(self, sid: str) -> Study:
        with self._lock:
            study = self._items.get(sid)
            if study is None:
                raise HTTPException(404, "Study not found (it may have expired; please re-upload)")
            self._items.move_to_end(sid)
            return study


store = StudyStore(MAX_STUDIES)
app = FastAPI(title="MRI Scanner", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


class BrainOptions(BaseModel):
    sensitivity: float = Field(3.0, ge=1.0, le=10.0)
    polarity: Literal["hyper", "hypo", "both"] = "hyper"
    min_volume_ml: float = Field(0.05, ge=0.0, le=100.0)
    require_asymmetry: bool = True
    engine: Literal["auto", "heuristic", "onnx"] = "auto"


class SpineOptions(BaseModel):
    lowest_level: str = "L5-S1"
    absolute_stenosis_mm: float = Field(10.0, gt=0, le=30)
    relative_stenosis_mm: float = Field(12.0, gt=0, le=30)
    narrowing_ratio: float = Field(0.67, gt=0, le=1)
    displacement_mm: float = Field(3.0, ge=0, le=20)
    sagittal_slice: int | None = None
    canal_seed: list[float] | None = None


class AnalyzeRequest(BaseModel):
    mode: Literal["brain", "spine"]
    brain: BrainOptions = BrainOptions()
    spine: SpineOptions = SpineOptions()


def _study_response(sid: str, vol: Volume) -> dict:
    return {"id": sid, "info": vol.info(), "disclaimer": DISCLAIMER}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "models": models.available(), "disclaimer": DISCLAIMER}


@app.post("/api/studies")
async def upload_study(files: list[UploadFile] = File(...)) -> dict:
    blobs, total = [], 0
    for f in files:
        content = await f.read()
        total += len(content)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload too large")
        blobs.append((f.filename or "upload", content))
    try:
        vol = load_files(blobs)
    except LoadError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _study_response(store.add(vol), vol)


@app.post("/api/demo/{kind}")
def load_demo(kind: Literal["brain", "spine"]) -> dict:
    vol = _demo_volume(kind)
    return _study_response(store.add(vol), vol)


@app.get("/api/demo/{kind}/download")
def download_demo(kind: Literal["brain", "spine"]) -> Response:
    """The demo study as a real file, to try the upload path."""
    if kind == "brain":
        return Response(demo.brain_nifti_bytes(), media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=demo_brain_flair.nii"})
    return Response(demo.spine_dicom_zip(), media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=demo_lumbar_sag_t2_dicom.zip"})


def _demo_volume(kind: str) -> Volume:
    if kind == "brain":
        nii, _ = demo.brain_phantom()
        return from_nifti_image(nii, "nifti", description="Demo brain (synthetic FLAIR phantom)", modality_hint="FLAIR")
    data, spacing, _ = demo.spine_phantom()
    nii = nib.Nifti1Image(data, demo.spine_affine(data.shape, spacing))
    return from_nifti_image(nii, "nifti", description="Demo lumbar spine (synthetic sagittal T2 phantom)", modality_hint="SAG T2")


@app.get("/api/studies/{sid}")
def get_study(sid: str) -> dict:
    study = store.get(sid)
    out = _study_response(sid, study.volume)
    out["result"] = study.result
    return out


@app.get("/api/studies/{sid}/volume")
def get_volume(sid: str) -> Response:
    study = store.get(sid)
    return Response(study.volume.display_bytes(), media_type="application/octet-stream")


@app.get("/api/studies/{sid}/mask")
def get_mask(sid: str) -> Response:
    study = store.get(sid)
    if study.mask is None:
        raise HTTPException(404, "No analysis has been run for this study")
    return Response(study.mask.tobytes(order="C"), media_type="application/octet-stream")


@app.get("/api/studies/{sid}/mask.nii.gz")
def get_mask_nifti(sid: str) -> Response:
    """Candidate mask as NIfTI (RAS+, same grid as the analysed volume)."""
    study = store.get(sid)
    if study.mask is None:
        raise HTTPException(404, "No analysis has been run for this study")
    img = nib.Nifti1Image(study.mask, study.volume.affine)
    return Response(gzip.compress(img.to_bytes()), media_type="application/gzip",
                    headers={"Content-Disposition": f"attachment; filename=candidates_{sid}.nii.gz"})


@app.post("/api/studies/{sid}/analyze")
def analyze(sid: str, req: AnalyzeRequest) -> dict:
    study = store.get(sid)
    with study.lock:
        try:
            if req.mode == "brain":
                mask, result = analyze_brain(study.volume, BrainParams(**req.brain.model_dump()))
            else:
                mask, result = analyze_spine(study.volume, SpineParams(**req.spine.model_dump()))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        result["disclaimer"] = DISCLAIMER
        result["study"] = {"id": sid, "description": study.volume.description, "shape": list(study.volume.shape),
                           "spacing": list(study.volume.spacing)}
        study.mask, study.result = mask, result
    return result


# Serve the built frontend when present (single-process deployment).
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        target = _DIST / path
        if path and target.is_file() and _DIST in target.resolve().parents:
            return FileResponse(target)
        return FileResponse(_DIST / "index.html")
