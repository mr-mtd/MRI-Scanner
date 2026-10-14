"""Measurements shared by every detector: volume, extent, diameter, location."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .volume import Volume


def max_diameter_mm(coords: np.ndarray, spacing, max_points: int = 1500) -> float:
    """Largest distance between two voxels of a region (Feret diameter), in mm."""
    if len(coords) < 2:
        return float(min(spacing))
    pts = coords * np.asarray(spacing, dtype=float)
    if len(pts) > max_points:
        # The extreme points lie on the boundary; sampling evenly keeps the estimate close.
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    best = 0.0
    for start in range(0, len(pts), 500):
        chunk = pts[start:start + 500]
        d = np.sqrt(((chunk[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).max()
        best = max(best, float(d))
    return best + float(min(spacing))


def brain_location(centroid_vox, brain_mask: np.ndarray | None, vol: Volume) -> dict:
    """Coarse anatomical description from position inside the brain bounding box.

    This is a geometric approximation (thirds of the brain extent), not atlas
    registration, and is labelled as such in the output.
    """
    x, y, z = centroid_vox
    if brain_mask is not None and brain_mask.any():
        idx = np.argwhere(brain_mask)
        lo, hi = idx.min(0), idx.max(0)
        mid_x = float(ndimage.center_of_mass(brain_mask)[0])
    else:
        lo, hi = np.zeros(3), np.array(vol.shape) - 1
        mid_x = (vol.shape[0] - 1) / 2
    rel = (np.array([x, y, z]) - lo) / np.maximum(hi - lo, 1)
    dist_mid_mm = abs(x - mid_x) * vol.spacing[0]
    if dist_mid_mm < 4:
        side = "midline"
    else:
        side = "right" if x > mid_x else "left"  # RAS+: x increases toward patient right

    ry, rz = rel[1], rel[2]
    if rz < 0.3:
        region = "posterior fossa / skull base" if ry < 0.45 else "temporal (inferior)"
    elif ry > 0.62:
        region = "frontal"
    elif ry < 0.28:
        region = "occipital"
    elif rz > 0.7:
        region = "parietal (superior)"
    elif rz < 0.5:
        region = "temporal"
    else:
        region = "deep / periventricular" if dist_mid_mm < 25 else "parietal"
    return {
        "hemisphere": side,
        "region_estimate": region,
        "region_method": "approximate: position within brain bounding box, not atlas-based",
        "relative_position": {"left_right": round(float(rel[0]), 3), "post_ant": round(float(ry), 3),
                              "inf_sup": round(float(rz), 3)},
    }


def describe_component(vol: Volume, mask: np.ndarray) -> dict:
    coords = np.argwhere(mask)
    n = len(coords)
    centroid = coords.mean(0)
    lo, hi = coords.min(0), coords.max(0)
    spacing = np.array(vol.spacing)
    volume_mm3 = n * vol.voxel_volume_mm3
    return {
        "voxel_count": int(n),
        "volume_mm3": round(float(volume_mm3), 1),
        "volume_ml": round(float(volume_mm3) / 1000.0, 3),
        "centroid_voxel": [round(float(c), 1) for c in centroid],
        "centroid_ras_mm": [round(v, 1) for v in vol.voxel_to_ras(centroid)],
        "bbox_voxel": {"min": [int(v) for v in lo], "max": [int(v) for v in hi]},
        "extent_mm": [round(float(v), 1) for v in (hi - lo + 1) * spacing],
        "max_diameter_mm": round(max_diameter_mm(coords, spacing), 1),
        "equivalent_sphere_diameter_mm": round(float((6 * volume_mm3 / np.pi) ** (1 / 3)), 1),
    }
