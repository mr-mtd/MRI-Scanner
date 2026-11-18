"""Load NIfTI files, DICOM series and zip archives of DICOM into a Volume."""

from __future__ import annotations

import gzip
import io
import zipfile
from collections import defaultdict

import nibabel as nib
import numpy as np
import pydicom

from .volume import Volume, from_nifti_image

MAX_ZIP_MEMBERS = 5000
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


class LoadError(ValueError):
    pass


def _is_nifti_name(name: str) -> bool:
    n = name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def load_nifti_bytes(name: str, payload: bytes) -> Volume:
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    try:
        img = nib.Nifti1Image.from_bytes(raw)
    except Exception:
        try:
            img = nib.Nifti2Image.from_bytes(raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise LoadError(f"Could not parse NIfTI file {name}: {exc}") from exc
    return from_nifti_image(img, "nifti", description=name)


def _read_dicom(name: str, payload: bytes):
    try:
        ds = pydicom.dcmread(io.BytesIO(payload), force=True)
    except Exception:
        return None
    if "PixelData" not in ds or "ImagePositionPatient" not in ds or "ImageOrientationPatient" not in ds:
        return None
    return ds


def load_dicom_datasets(datasets: list) -> Volume:
    """Stack one DICOM series (single-frame slices) into a canonical volume."""
    series: dict[str, list] = defaultdict(list)
    for ds in datasets:
        series[str(getattr(ds, "SeriesInstanceUID", "unknown"))].append(ds)
    if not series:
        raise LoadError("No DICOM images with pixel data and patient geometry were found.")

    warnings = []
    uid, slices = max(series.items(), key=lambda kv: len(kv[1]))
    if len(series) > 1:
        warnings.append(f"{len(series)} DICOM series found; loaded the largest ({len(slices)} slices). "
                        "Upload one series at a time to choose another.")

    iop = np.array(slices[0].ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    slices.sort(key=lambda ds: float(np.dot(normal, np.array(ds.ImagePositionPatient, dtype=float))))

    rows, cols = int(slices[0].Rows), int(slices[0].Columns)
    slices = [ds for ds in slices if int(ds.Rows) == rows and int(ds.Columns) == cols]
    if len(slices) < 2:
        raise LoadError("A 3D DICOM series needs at least two slices of the same size.")

    pixel_spacing = [float(v) for v in slices[0].PixelSpacing]  # [between rows, between columns]
    vol = np.empty((cols, rows, len(slices)), dtype=np.float32)
    for k, ds in enumerate(slices):
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        vol[:, :, k] = (arr * slope + intercept).T

    first = np.array(slices[0].ImagePositionPatient, dtype=float)
    last = np.array(slices[-1].ImagePositionPatient, dtype=float)
    slice_step = (last - first) / (len(slices) - 1)
    if np.linalg.norm(slice_step) < 1e-6:
        raise LoadError("DICOM slices share the same position; cannot build a 3D volume.")

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_dir * pixel_spacing[1]
    affine_lps[:3, 1] = col_dir * pixel_spacing[0]
    affine_lps[:3, 2] = slice_step
    affine_lps[:3, 3] = first
    img = nib.Nifti1Image(vol, LPS_TO_RAS @ affine_lps)

    ref = slices[0]
    meta = {
        "series_instance_uid": uid,
        "series_description": str(getattr(ref, "SeriesDescription", "")),
        "modality": str(getattr(ref, "Modality", "")),
        "num_slices": len(slices),
        "warnings": warnings,
    }
    return from_nifti_image(img, "dicom", description=meta["series_description"] or "DICOM series", metadata=meta)


def load_uploads(files: list[tuple[str, bytes]]) -> Volume:
    """Load an upload consisting of one NIfTI, a zip of DICOM, or many DICOM files."""
    if not files:
        raise LoadError("No files were uploaded.")

    for name, payload in files:
        if _is_nifti_name(name):
            return load_nifti_bytes(name, payload)

    dicoms = []
    total = 0
    for name, payload in files:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                members = [m for m in zf.infolist() if not m.is_dir()]
                if len(members) > MAX_ZIP_MEMBERS:
                    raise LoadError(f"Zip has too many files ({len(members)}).")
                for m in members:
                    total += m.file_size
                    if total > MAX_UNCOMPRESSED_BYTES:
                        raise LoadError("Zip contents are too large.")
                    data = zf.read(m)  # read in memory only; nothing is written to disk
                    if _is_nifti_name(m.filename):
                        return load_nifti_bytes(m.filename, data)
                    ds = _read_dicom(m.filename, data)
                    if ds is not None:
                        dicoms.append(ds)
        else:
            ds = _read_dicom(name, payload)
            if ds is not None:
                dicoms.append(ds)

    if not dicoms:
        raise LoadError("No readable NIfTI or DICOM images found. Upload a .nii/.nii.gz file, "
                        "a set of .dcm slices, or a .zip of one DICOM series.")
    return load_dicom_datasets(dicoms)
