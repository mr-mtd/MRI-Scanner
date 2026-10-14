"""In-memory representation of an MRI volume.

Every loader converts its input to a ``Volume`` whose array is in RAS+
canonical orientation: axis 0 runs patient Left -> Right, axis 1 runs
Posterior -> Anterior, axis 2 runs Inferior -> Superior. The analysis code and
the frontend viewer both rely on this, so nothing downstream needs to know
whether a study came from DICOM or NIfTI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
from scipy import ndimage

# Largest volume sent to the browser. Bigger studies are downsampled for
# display and analysis; measurements stay in millimetres so they are unaffected
# apart from the coarser voxel grid.
MAX_VOXELS = 24_000_000


@dataclass
class Volume:
    data: np.ndarray  # float32, shape (X, Y, Z), RAS+ canonical
    affine: np.ndarray  # 4x4 voxel -> world (RAS mm)
    source_format: str  # "nifti" | "dicom"
    description: str = ""
    modality_hint: str = ""  # e.g. series description / sequence name
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)  # type: ignore[return-value]

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(float(s) for s in np.linalg.norm(self.affine[:3, :3], axis=0))  # type: ignore[return-value]

    @property
    def voxel_volume_ml(self) -> float:
        sx, sy, sz = self.spacing
        return sx * sy * sz / 1000.0

    def voxel_to_world(self, ijk) -> list[float]:
        ijk = np.asarray(ijk, dtype=float)
        return [float(v) for v in (self.affine[:3, :3] @ ijk + self.affine[:3, 3])]

    def display_bytes(self) -> bytes:
        """Volume scaled to uint8 (0.5-99.5 percentile window), C order."""
        return to_uint8(self.data).tobytes(order="C")

    def info(self) -> dict:
        return {
            "shape": list(self.shape),
            "spacing": list(self.spacing),
            "affine": self.affine.tolist(),
            "sourceFormat": self.source_format,
            "description": self.description,
            "modalityHint": self.modality_hint,
            "meta": self.meta,
        }


def to_uint8(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    scaled = (np.nan_to_num(data, nan=lo) - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def from_nifti_image(img: nib.Nifti1Image, source_format: str, **kwargs) -> Volume:
    """Reorient any nibabel image to RAS+ and wrap it as a Volume."""
    canonical = nib.as_closest_canonical(img)
    data = np.asanyarray(canonical.dataobj).astype(np.float32)
    affine = canonical.affine.copy()
    if data.ndim == 4:
        # Multi-volume series (e.g. DWI, perfusion): analyse the first volume.
        kwargs.setdefault("meta", {})["note"] = f"4D input; using volume 0 of {data.shape[3]}"
        data = data[..., 0]
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {data.shape}")
    vol = Volume(data=data, affine=affine, source_format=source_format, **kwargs)
    return limit_size(vol)


def limit_size(vol: Volume, max_voxels: int = MAX_VOXELS) -> Volume:
    n = int(np.prod(vol.shape))
    if n <= max_voxels:
        return vol
    factor = (max_voxels / n) ** (1.0 / 3.0)
    data = ndimage.zoom(vol.data, factor, order=1)
    real = np.array(data.shape) / np.array(vol.shape)
    affine = vol.affine.copy()
    affine[:3, :3] = affine[:3, :3] / real
    vol.meta["downsampled_from"] = list(vol.shape)
    return Volume(
        data=data.astype(np.float32),
        affine=affine,
        source_format=vol.source_format,
        description=vol.description,
        modality_hint=vol.modality_hint,
        meta=vol.meta,
    )
