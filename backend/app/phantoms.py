"""Synthetic demo volumes with known ground truth.

These are simple digital phantoms, not patient data. They let the app run end to
end without any real scans and give the tests a known answer to check against.
"""

from __future__ import annotations

import io

import nibabel as nib
import numpy as np
from scipy import ndimage

from .volume import Volume, from_nifti_image


def _grid(shape, spacing):
    axes = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, spacing)]
    return np.meshgrid(*axes, indexing="ij")


def brain_phantom(seed: int = 0):
    """FLAIR-like brain with two hyperintense lesions. Returns (Volume, truth dict)."""
    rng = np.random.default_rng(seed)
    spacing = (1.25, 1.25, 1.25)
    shape = (120, 144, 112)
    x, y, z = _grid(shape, spacing)

    def ell(cx, cy, cz, ax, ay, az):
        return ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 + ((z - cz) / az) ** 2 <= 1

    img = np.full(shape, 0.02, dtype=np.float32)
    img[ell(0, 0, 0, 72, 90, 68)] = 0.75   # scalp (fat is bright)
    img[ell(0, 0, 0, 67, 85, 63)] = 0.05   # skull (dark)
    img[ell(0, 0, 0, 61, 79, 57)] = 0.08   # CSF (dark on FLAIR)
    img[ell(0, 0, 0, 59, 77, 55)] = 0.62   # grey matter
    img[ell(0, 0, 0, 53, 70, 49)] = 0.50   # white matter
    img[ell(-9, 5, 6, 5, 22, 9) | ell(9, 5, 6, 5, 22, 9)] = 0.08  # lateral ventricles

    lesions = [
        {"name": "right frontal", "mask": ell(26, 42, 10, 10, 10, 10)},
        {"name": "left parieto-occipital", "mask": ell(-30, -38, 18, 7, 6, 6)},
    ]
    for les in lesions:
        img[les["mask"]] = 0.95

    bias = 1 + 0.08 * np.sin(x / 60.0) * np.cos(y / 70.0)
    img = ndimage.gaussian_filter(img * bias, 0.6) + rng.normal(0, 0.02, shape).astype(np.float32)
    img = np.clip(img, 0, None).astype(np.float32) * 1000

    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = -(np.array(shape) - 1) / 2 * np.array(spacing)
    vox = float(np.prod(spacing))
    truth = [{"name": l["name"], "volume_mm3": float(l["mask"].sum() * vox),
              "centroid_voxel": [float(c) for c in np.argwhere(l["mask"]).mean(0)]} for l in lesions]
    nii = nib.Nifti1Image(img, affine)
    return from_nifti_image(nii, "demo", "Synthetic brain phantom (FLAIR-like), two lesions"), truth


def spine_phantom(seed: int = 1):
    """Sagittal T2-like lumbar spine with an L4-L5 protrusion narrowing the canal and a
    desiccated L5-S1 disc. Returns (Volume, truth dict)."""
    rng = np.random.default_rng(seed)
    spacing = (4.0, 0.9, 0.9)  # thick sagittal slices, fine in-plane
    shape = (15, 200, 260)
    x, y, z = _grid(shape, spacing)
    img = np.full(shape, 0.03, dtype=np.float32)

    body_y0, body_y1 = 0.0, 36.0       # vertebral bodies / discs (anterior column)
    canal_y0 = body_y0 - 15.0           # thecal sac, AP 13 mm, walled by 3 mm of dark ligament / dura
    lr_body = np.abs(x) <= 22
    lr_canal = np.abs(x) <= 8
    zmin = z.min()

    img[(y > body_y1) & (y < body_y1 + 50) & lr_body] = 0.35                     # prevertebral soft tissue
    img[(y < canal_y0) & (y > canal_y0 - 22) & lr_body] = 0.30                   # posterior elements
    img[(y <= canal_y0 - 22) & (y > canal_y0 - 38)] = 0.72                       # subcutaneous fat
    img[(y >= canal_y0 - 3) & (y < body_y0) & lr_body] = 0.04                   # ligamentum flavum, PLL
    in_canal = (y >= canal_y0) & (y < body_y0 - 3) & lr_canal
    img[in_canal] = 1.0

    levels = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1"]
    disc_h, body_h = 9.0, 24.0
    cursor = zmin + 34.0               # top of S1
    in_column = (y >= body_y0) & (y <= body_y1) & lr_body
    img[in_column & (z < cursor)] = 0.32
    disc_centers = {}
    for name in levels:
        d0, d1 = cursor, cursor + disc_h
        disc_centers[name] = (d0 + d1) / 2
        signal = 0.15 if name == "L5-S1" else 0.70
        img[in_column & (z >= d0) & (z < d1)] = signal
        if name == "L4-L5":
            # Posterior protrusion 7 mm into the canal across the disc height.
            # Posterior protrusion: disc material 7 mm into the canal, still rimmed by dark annulus.
            rim = lr_canal & (z >= d0 - 3) & (z < d1 + 3) & (y < body_y0) & (y >= body_y0 - 12.0)
            img[rim] = 0.04
            img[lr_canal & (z >= d0 - 1) & (z < d1 + 1) & (y < body_y0) & (y >= body_y0 - 9.0)] = signal
        b1 = d1 + body_h
        img[in_column & (z >= d1) & (z < b1)] = 0.32
        cursor = b1

    img = ndimage.gaussian_filter(img, (0.3, 0.7, 0.7)) + rng.normal(0, 0.025, shape).astype(np.float32)
    img = np.clip(img, 0, None).astype(np.float32) * 1000
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = -(np.array(shape) - 1) / 2 * np.array(spacing)
    truth = {"stenosis_level": "L4-L5", "stenosis_ap_mm": 3.0, "normal_ap_mm": 12.0,
             "protrusion_mm": 9.0, "desiccated_level": "L5-S1",
             "disc_center_z_mm": {k: round(v - zmin, 1) for k, v in disc_centers.items()}}
    nii = nib.Nifti1Image(img, affine)
    return from_nifti_image(nii, "demo", "Synthetic lumbar spine phantom (sagittal T2-like)"), truth


def volume_to_nifti_bytes(vol: Volume) -> bytes:
    img = nib.Nifti1Image(vol.data, vol.affine)
    bio = io.BytesIO()
    file_map = nib.Nifti1Image.make_file_map({"image": bio, "header": bio})
    img.to_file_map(file_map)
    return bio.getvalue()


def volume_to_dicom_series(vol: Volume, series_description: str = "Synthetic phantom") -> list[tuple[str, bytes]]:
    """Write the volume as an axial single-frame MR DICOM series (used for demos and tests)."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    data = vol.data
    scale = 4095.0 / max(float(data.max()), 1e-6)
    pix = np.clip(data * scale, 0, 4095).astype(np.uint16)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    aff = ras_to_lps @ vol.affine
    series_uid, study_uid = generate_uid(), generate_uid()
    sx, sy, _ = vol.spacing
    row_dir = aff[:3, 0] / sx
    col_dir = aff[:3, 1] / sy
    files = []
    for k in range(data.shape[2]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "MR"
        ds.PatientName = "PHANTOM^DEMO"
        ds.PatientID = "PHANTOM"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = series_description
        ds.InstanceNumber = k + 1
        ds.ImageOrientationPatient = [float(v) for v in (*row_dir, *col_dir)]
        ds.ImagePositionPatient = [float(v) for v in (aff[:3, :3] @ np.array([0, 0, k]) + aff[:3, 3])]
        ds.PixelSpacing = [float(sy), float(sx)]
        ds.SliceThickness = float(vol.spacing[2])
        ds.Rows, ds.Columns = int(data.shape[1]), int(data.shape[0])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
        ds.PixelRepresentation = 0
        ds.PixelData = np.ascontiguousarray(pix[:, :, k].T).tobytes()
        bio = io.BytesIO()
        ds.save_as(bio, enforce_file_format=True)
        files.append((f"slice_{k:04d}.dcm", bio.getvalue()))
    return files
