"""In-memory representation of a loaded MRI volume.

Every volume is reoriented to the closest canonical RAS+ orientation on load, so
array axis 0 runs patient left -> right, axis 1 posterior -> anterior and axis 2
inferior -> superior. The analysis code and the viewer both rely on that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import nibabel as nib
import numpy as np


@dataclass
class Volume:
    data: np.ndarray  # float32, shape (X, Y, Z), canonical RAS+
    affine: np.ndarray  # 4x4 voxel -> RAS mm
    source_format: str  # "nifti" | "dicom" | "demo"
    description: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)  # type: ignore[return-value]

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(float(s) for s in np.sqrt((self.affine[:3, :3] ** 2).sum(axis=0)))  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing
        return sx * sy * sz

    def voxel_to_ras(self, ijk) -> list[float]:
        ijk = np.asarray(ijk, dtype=float)
        return [float(v) for v in (self.affine[:3, :3] @ ijk + self.affine[:3, 3])]

    def display_uint8(self) -> np.ndarray:
        """Robustly rescale intensities to 0-255 for the viewer."""
        d = self.data
        nonzero = d[d > 0]
        sample = nonzero if nonzero.size > 1000 else d.ravel()
        lo, hi = np.percentile(sample, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((d - lo) / (hi - lo), 0.0, 1.0) * 255.0
        return out.astype(np.uint8)


def from_nifti_image(img: nib.spatialimages.SpatialImage, source_format: str, description: str = "",
                     metadata: dict | None = None) -> Volume:
    img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:
        # Multi-volume series (e.g. diffusion or multi-echo): analyse the first volume.
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return Volume(data=data, affine=img.affine.copy(), source_format=source_format,
                  description=description, metadata=metadata or {})
