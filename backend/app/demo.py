"""Synthetic demo studies so the app runs end to end without patient data.

These are simple digital phantoms, not realistic anatomy. They exist to
exercise loading, analysis, overlays and measurements, and the tests use their
known ground truth to check the pipelines.
"""

from __future__ import annotations

import datetime as _dt
import io
import zipfile

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from scipy import ndimage


def _grid(shape, spacing):
    axes = [(np.arange(n) - (n - 1) / 2.0) * s for n, s in zip(shape, spacing)]
    return np.meshgrid(*axes, indexing="ij")


def brain_phantom(seed: int = 0) -> tuple[nib.Nifti1Image, dict]:
    """FLAIR-like brain with two hyperintense lesions.

    Returns the NIfTI image (RAS+, 1.5 mm isotropic) and ground truth with
    lesion masks and volumes.
    """
    rng = np.random.default_rng(seed)
    shape, spacing = (112, 136, 104), (1.5, 1.5, 1.5)
    x, y, z = _grid(shape, spacing)

    def ellipsoid(cx, cy, cz, rx, ry, rz):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1.0

    head = ellipsoid(0, 0, 0, 72, 90, 68)
    skull = ellipsoid(0, 0, 0, 67, 85, 63)
    csf = ellipsoid(0, 0, 0, 62, 80, 58)
    brain = ellipsoid(0, 0, 0, 59, 77, 55)
    white = ellipsoid(0, 0, 2, 47, 63, 42)

    img = np.zeros(shape, np.float32)
    img[head] = 0.55  # scalp
    img[skull] = 0.05  # bone
    img[csf] = 0.12  # suppressed CSF on FLAIR
    img[brain] = 0.42  # gray matter
    img[white] = 0.34  # white matter

    # Symmetric lateral ventricles (dark on FLAIR).
    for side in (-1, 1):
        img[ellipsoid(side * 11, 4, 8, 7, 26, 11)] = 0.1

    # Lesion 1: irregular right frontal mass (+x is patient right in RAS).
    l1 = ellipsoid(28, 42, 14, 14, 12, 11) | ellipsoid(35, 50, 8, 8, 8, 8)
    # Lesion 2: small left periventricular focus.
    l2 = ellipsoid(-22, -14, 18, 5, 6, 5)
    img[l1] = 0.85
    img[l2] = 0.78

    # Smooth bias field + Rician-like noise + mild texture.
    bias = 1.0 + 0.08 * (x / 72.0) - 0.06 * (z / 68.0)
    img *= bias.astype(np.float32)
    texture = ndimage.gaussian_filter(rng.normal(0, 1, shape), 2.0).astype(np.float32)
    img += 0.02 * texture * (img > 0.2)
    noise = rng.normal(0, 0.02, (2,) + shape).astype(np.float32)
    img = np.sqrt((img + noise[0]) ** 2 + noise[1] ** 2)

    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    affine[:3, 3] = [-(n - 1) / 2.0 * s for n, s in zip(shape, spacing)]
    nii = nib.Nifti1Image((img * 1000).astype(np.int16), affine)
    nii.header["descrip"] = b"DEMO brain FLAIR phantom"
    truth = {
        "lesions": [l1, l2],
        "volumes_ml": [float(m.sum() * np.prod(spacing) / 1000.0) for m in (l1, l2)],
        "spacing": spacing,
    }
    return nii, truth


# Lumbar spine phantom ---------------------------------------------------------

# Craniocaudal disc order from top of FOV to bottom.
SPINE_LEVELS = ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


def spine_phantom(seed: int = 1) -> tuple[np.ndarray, tuple[float, float, float], dict]:
    """Sagittal T2-like lumbar spine.

    Returns ``(data, spacing, truth)`` with data in RAS+ (X, Y, Z). Built-in
    abnormalities: a posterior disc herniation at L4-L5 narrowing the canal,
    and a desiccated, thinned L5-S1 disc.
    """
    rng = np.random.default_rng(seed)
    spacing = (4.0, 0.75, 0.75)
    shape = (13, 240, 400)  # 52 x 180 x 300 mm
    x, y, z = _grid(shape, spacing)

    body_h, disc_h, body_d = 27.0, 10.0, 34.0
    pitch = body_h + disc_h
    # Disc centres along z (top to bottom), sacrum below the last one.
    disc_z = [110.0 - i * pitch for i in range(len(SPINE_LEVELS))]
    disc_heights = [disc_h] * 5 + [3.0]

    # Mild lordosis: anterior offset of the column varies with z.
    curve = 6.0 * np.cos(z / 150.0 * np.pi)
    ya = 18.0 + curve  # anterior surface of vertebral bodies (RAS +y anterior)
    yp = ya - body_d  # posterior surface of bodies = anterior canal wall
    canal_ap = 14.0
    in_mid = np.abs(x) <= 11.0
    in_column = np.abs(x) <= 22.0

    img = np.full(shape, 0.02, np.float32)  # air
    body_outline = (np.abs(x) <= 70) & (y > -95) & (y < 75)
    img[body_outline] = 0.28  # abdomen / muscle
    img[body_outline & (y < -70)] = 0.82  # posterior subcutaneous fat
    img[body_outline & (y >= -70) & (y < yp - canal_ap - 6)] = 0.22  # paraspinal muscle

    # Posterior elements (laminae, ligamentum flavum) - dark/intermediate.
    post = in_column & (y < yp - canal_ap) & (y >= yp - canal_ap - 6)
    img[post] = 0.18

    # Vertebral bodies with dark cortical endplates, then discs.
    column = in_column & (y <= ya) & (y >= yp)
    img[column] = 0.42
    disc_masks = []
    for zc, h in zip(disc_z, disc_heights):
        endplate = column & (np.abs(z - zc) <= h / 2 + 1.0)
        img[endplate] = 0.12
        disc = column & (np.abs(z - zc) <= h / 2) & (y <= ya - 1) & (y >= yp + 1)
        disc_masks.append(disc)
        img[disc] = 0.72
    # L5-S1 desiccation: low T2 signal.
    img[disc_masks[-1]] = 0.2
    # Sacrum: everything below L5-S1 in the column.
    sacrum = column & (z < disc_z[-1] - disc_heights[-1] / 2 - 1.0)
    img[sacrum] = 0.38

    # Spinal canal (CSF) along the whole FOV.
    canal = in_mid & (y < yp) & (y >= yp - canal_ap)
    img[canal] = 1.0
    # Nerve roots: a few faint dark strands.
    img[canal & (np.abs(x) < 2.5) & (np.abs(y - (yp - canal_ap * 0.6)) < 1.0)] = 0.7

    # L4-L5 posterior herniation: disc material extending 8 mm into the canal.
    zc = disc_z[4]
    herniation = in_mid & (y < yp) & (y >= yp - 8.0) & ((z - zc) ** 2 / 7.0**2 + (y - yp) ** 2 / 8.0**2 <= 1.0)
    img[herniation] = 0.34

    img = ndimage.gaussian_filter(img, (0.3, 0.6, 0.6))
    noise = rng.normal(0, 0.025, (2,) + shape).astype(np.float32)
    img = np.sqrt((img + noise[0]) ** 2 + noise[1] ** 2).astype(np.float32)

    truth = {
        "levels": SPINE_LEVELS,
        "disc_z_mm": disc_z,
        "herniation_level": "L4-L5",
        "desiccated_level": "L5-S1",
        "canal_ap_mm": canal_ap,
        "canal_ap_at_herniation_mm": canal_ap - 8.0,
    }
    return img, spacing, truth


def spine_affine(shape, spacing) -> np.ndarray:
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    affine[:3, 3] = [-(n - 1) / 2.0 * s for n, s in zip(shape, spacing)]
    return affine


def spine_dicom_zip(seed: int = 1) -> bytes:
    """The spine phantom written as a sagittal DICOM series inside a zip."""
    data, spacing, _ = spine_phantom(seed)
    affine_ras = spine_affine(data.shape, spacing)
    pixels = np.clip(data * 1000, 0, 4095).astype(np.uint16)

    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    now = _dt.datetime(2026, 1, 1, 12, 0, 0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(data.shape[0]):
            # Sagittal image: rows run superior->inferior, columns anterior->posterior.
            # In LPS, posterior is +y and inferior is -z.
            frame = pixels[i, ::-1, ::-1].T  # (rows=z top->bottom, cols=y ant->post)
            top_left_ras = affine_ras @ np.array([i, data.shape[1] - 1, data.shape[2] - 1, 1.0])
            ipp = [-top_left_ras[0], -top_left_ras[1], top_left_ras[2]]

            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
            ds.Modality = "MR"
            ds.PatientName = "DEMO^PHANTOM"
            ds.PatientID = "DEMO"
            ds.StudyDate = ds.SeriesDate = now.strftime("%Y%m%d")
            ds.SeriesDescription = "DEMO SAG T2 LUMBAR PHANTOM"
            ds.BodyPartExamined = "LSPINE"
            ds.Manufacturer = "Synthetic"
            ds.SeriesNumber, ds.InstanceNumber = 1, i + 1
            ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
            ds.ImagePositionPatient = [float(v) for v in ipp]
            ds.PixelSpacing = [spacing[2], spacing[1]]
            ds.SliceThickness = spacing[0]
            ds.Rows, ds.Columns = frame.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
            ds.PixelRepresentation = 0
            ds.PixelData = np.ascontiguousarray(frame).tobytes()
            out = io.BytesIO()
            ds.save_as(out, enforce_file_format=True)
            zf.writestr(f"sag_t2/IM{i + 1:04d}.dcm", out.getvalue())
    return buf.getvalue()


def brain_nifti_bytes(seed: int = 0) -> bytes:
    nii, _ = brain_phantom(seed)
    return nii.to_bytes()
