"""Load uploaded files (NIfTI, DICOM files, or a zip of DICOM) into a Volume."""

from __future__ import annotations

import gzip
import io
import zipfile
from collections import defaultdict

import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from .volume import Volume, from_nifti_image

# LPS (DICOM patient coordinates) -> RAS (NIfTI world coordinates).
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


class LoadError(ValueError):
    pass


def load_files(files: list[tuple[str, bytes]]) -> Volume:
    """Load a study from uploaded ``(filename, content)`` pairs."""
    if not files:
        raise LoadError("No files uploaded")

    niftis = [f for f in files if _is_nifti(*f)]
    if niftis:
        if len(niftis) > 1:
            raise LoadError("Upload one NIfTI file at a time")
        return load_nifti(*niftis[0])

    dicom_blobs: list[tuple[str, bytes]] = []
    for name, content in files:
        if name.lower().endswith(".zip") or content[:4] == b"PK\x03\x04":
            dicom_blobs.extend(_unzip(content))
        else:
            dicom_blobs.append((name, content))
    return load_dicom(dicom_blobs)


def _is_nifti(name: str, content: bytes) -> bool:
    lower = name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def load_nifti(name: str, content: bytes) -> Volume:
    raw = gzip.decompress(content) if content[:2] == b"\x1f\x8b" else content
    try:
        img = nib.Nifti1Image.from_bytes(raw)
    except Exception:
        try:
            img = nib.Nifti2Image.from_bytes(raw)
        except Exception as exc:  # noqa: BLE001
            raise LoadError(f"Could not read NIfTI file {name}: {exc}") from exc
    descrip = img.header.get("descrip")
    text = descrip.tobytes().decode("ascii", "ignore").strip("\x00 ") if descrip is not None else ""
    return from_nifti_image(img, "nifti", description=name, modality_hint=text)


def _unzip(content: bytes) -> list[tuple[str, bytes]]:
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX"):
                    continue
                out.append((info.filename, zf.read(info)))
    except zipfile.BadZipFile as exc:
        raise LoadError(f"Invalid zip archive: {exc}") from exc
    return out


def load_dicom(blobs: list[tuple[str, bytes]]) -> Volume:
    series: dict[str, list[pydicom.Dataset]] = defaultdict(list)
    for name, content in blobs:
        try:
            ds = pydicom.dcmread(io.BytesIO(content), force=False)
        except (InvalidDicomError, Exception):  # noqa: BLE001 - skip non-DICOM files in archives
            continue
        if "PixelData" not in ds:
            continue
        series[str(getattr(ds, "SeriesInstanceUID", "unknown"))].append(ds)

    if not series:
        raise LoadError("No readable DICOM images found (expected .dcm files or a .zip of a series)")

    # Several series in one upload: analyse the one with the most images.
    uid, slices = max(series.items(), key=lambda kv: len(kv[1]))
    vol = _stack_series(slices)
    vol.meta["series_found"] = len(series)
    vol.meta["series_uid"] = uid
    if len(series) > 1:
        vol.meta["note"] = f"{len(series)} series uploaded; showing the largest ({len(slices)} images)"
    return vol


def _stack_series(slices: list[pydicom.Dataset]) -> Volume:
    first = slices[0]
    if int(getattr(first, "NumberOfFrames", 1)) > 1:
        raise LoadError("Multi-frame (enhanced) DICOM is not supported yet; export as classic single-frame images")

    iop = np.array(getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)

    def position(ds) -> np.ndarray:
        return np.array(getattr(ds, "ImagePositionPatient", [0, 0, float(getattr(ds, "InstanceNumber", 0))]), dtype=float)

    slices = sorted(slices, key=lambda ds: float(position(ds) @ normal))

    shapes = {(int(ds.Rows), int(ds.Columns)) for ds in slices}
    if len(shapes) > 1:
        raise LoadError(f"Images in the series have different sizes: {sorted(shapes)}")

    try:
        frames = []
        for ds in slices:
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            frames.append(arr * slope + intercept)
    except Exception as exc:  # noqa: BLE001
        raise LoadError(f"Could not decode pixel data ({exc}). Compressed DICOM may need pylibjpeg/gdcm installed.") from exc

    # (rows, cols, slices) -> voxel (col, row, slice) so axis 0 follows row_dir.
    data = np.stack(frames, axis=-1).transpose(1, 0, 2)

    ps = [float(v) for v in getattr(first, "PixelSpacing", [1.0, 1.0])]
    if len(slices) > 1:
        gaps = np.diff([float(position(ds) @ normal) for ds in slices])
        dz = float(np.median(gaps)) or float(getattr(first, "SliceThickness", 1.0))
    else:
        dz = float(getattr(first, "SliceThickness", 1.0) or 1.0)

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_dir * ps[1]
    affine_lps[:3, 1] = col_dir * ps[0]
    affine_lps[:3, 2] = normal * dz
    affine_lps[:3, 3] = position(slices[0])

    img = nib.Nifti1Image(data, LPS_TO_RAS @ affine_lps)
    desc = str(getattr(first, "SeriesDescription", "") or getattr(first, "ProtocolName", ""))
    meta = {
        "images": len(slices),
        "modality": str(getattr(first, "Modality", "")),
        "body_part": str(getattr(first, "BodyPartExamined", "")),
        "manufacturer": str(getattr(first, "Manufacturer", "")),
    }
    # Patient identifiers are deliberately not copied into metadata.
    return from_nifti_image(img, "dicom", description=desc or "DICOM series", modality_hint=desc, meta=meta)
