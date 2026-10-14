"""FastAPI service: load MRI studies, run candidate detection, serve voxels to the viewer."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import phantoms
from .brain import analyze_brain
from .loaders import LoadError, load_dicom_datasets, load_uploads
from .onnx_model import load_configured_model
from .spine import analyze_spine
from .volume import Volume

DISCLAIMER = ("Research prototype. Outputs are candidate findings to support radiologist review, "
              "not diagnoses, and the software is not validated for clinical use.")
MAX_STUDIES = 8

app = FastAPI(title="MRI Scanner", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()
_studies: "OrderedDict[str, dict]" = OrderedDict()
_model, _model_error = load_configured_model()


def _store(vol: Volume, extra: dict | None = None) -> dict:
    sid = uuid.uuid4().hex[:12]
    entry = {"volume": vol, "display": vol.display_uint8(), "analysis": None, "labels": None, "extra": extra or {}}
    with _lock:
        _studies[sid] = entry
        while len(_studies) > MAX_STUDIES:
            _studies.popitem(last=False)
    return _meta(sid, entry)


def _get(sid: str) -> dict:
    with _lock:
        entry = _studies.get(sid)
    if entry is None:
        raise HTTPException(404, "Study not found (it may have been evicted; reload it).")
    return entry


def _meta(sid: str, entry: dict) -> dict:
    vol: Volume = entry["volume"]
    return {
        "id": sid,
        "description": vol.description,
        "source_format": vol.source_format,
        "shape": list(vol.shape),
        "spacing_mm": [round(s, 4) for s in vol.spacing],
        "affine": [[round(float(v), 5) for v in row] for row in vol.affine],
        "orientation": "RAS+ canonical (axis 0: L->R, axis 1: P->A, axis 2: I->S)",
        "warnings": vol.metadata.get("warnings", []),
        "metadata": {k: v for k, v in vol.metadata.items() if k != "warnings"},
        **entry["extra"],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER,
            "pretrained_brain_model": _model.name if _model else None,
            "pretrained_brain_model_error": _model_error}


@app.post("/api/studies")
async def upload_study(files: list[UploadFile] = File(...)):
    payloads = [(f.filename or "upload", await f.read()) for f in files]
    try:
        vol = load_uploads(payloads)
    except LoadError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, f"Could not read the upload: {exc}") from exc
    return _store(vol)


@app.post("/api/demo/{kind}")
def load_demo(kind: Literal["brain", "brain-dicom", "spine"]):
    if kind == "spine":
        vol, truth = phantoms.spine_phantom()
        return _store(vol, {"demo_truth": truth, "suggested_mode": "spine"})
    vol, truth = phantoms.brain_phantom()
    if kind == "brain-dicom":
        # Round-trip through DICOM so the demo exercises the DICOM loader.
        import io

        import pydicom
        files = phantoms.volume_to_dicom_series(vol, "Synthetic brain phantom (DICOM)")
        vol = load_dicom_datasets([pydicom.dcmread(io.BytesIO(b)) for _, b in files])
    return _store(vol, {"demo_truth": truth, "suggested_mode": "brain"})


@app.get("/api/studies/{sid}")
def get_study(sid: str):
    return _meta(sid, _get(sid))


@app.get("/api/studies/{sid}/volume")
def get_volume(sid: str):
    """Display voxels as raw uint8, C order over (X, Y, Z) of the canonical array."""
    arr = _get(sid)["display"]
    return Response(np.ascontiguousarray(arr).tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": ",".join(map(str, arr.shape))})


@app.get("/api/studies/{sid}/labels")
def get_labels(sid: str):
    labels = _get(sid)["labels"]
    if labels is None:
        raise HTTPException(404, "Run an analysis first.")
    return Response(np.ascontiguousarray(labels).tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": ",".join(map(str, labels.shape))})


class AnalyzeRequest(BaseModel):
    mode: Literal["brain", "spine"]
    method: Literal["baseline", "pretrained"] = "baseline"
    sensitivity: float = Field(3.0, ge=1.0, le=10.0)
    min_volume_mm3: float = Field(30.0, ge=0.0)
    polarity: Literal["hyper", "hypo"] = "hyper"
    use_asymmetry: bool = True
    already_skull_stripped: bool = False
    min_protrusion_mm: float = Field(3.0, ge=0.5, le=20.0)


@app.post("/api/studies/{sid}/analyze")
def analyze(sid: str, req: AnalyzeRequest):
    entry = _get(sid)
    vol: Volume = entry["volume"]
    if req.mode == "brain":
        model = None
        if req.method == "pretrained":
            if _model is None:
                raise HTTPException(400, _model_error or "No pretrained model is configured (set MRI_BRAIN_MODEL).")
            model = _model
        result = analyze_brain(vol, sensitivity=req.sensitivity, min_volume_mm3=req.min_volume_mm3,
                               polarity=req.polarity, use_asymmetry=req.use_asymmetry,
                               already_skull_stripped=req.already_skull_stripped, model=model)
    else:
        result = analyze_spine(vol, min_protrusion_mm=req.min_protrusion_mm)

    labels = result.pop("label_map")
    result.pop("context_mask", None)
    entry["labels"] = labels
    entry["analysis"] = result
    return {"study_id": sid, "disclaimer": DISCLAIMER, "request": req.model_dump(), **result}


_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        return FileResponse(_dist / "index.html")
