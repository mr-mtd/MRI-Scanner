"""Brain MRI candidate lesion detection.

Baseline method (always available, no model weights): intensity-outlier detection
inside an automatically extracted brain mask, with a left/right asymmetry check
so symmetric bright structures (CSF on T2, normal cortex) are not flagged.
It is tuned for single-sequence FLAIR or T2 where tumours, oedema and white
matter lesions appear hyperintense, and for post-contrast T1 enhancement.

An optional pretrained segmentation model (ONNX) can replace the candidate
mask; the measurement code is shared, so outputs look the same either way.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from .measure import brain_location, describe_component
from .volume import Volume

MAX_FINDINGS = 50
GROW_FRACTION = 0.5  # grow each seed to this fraction of its peak contrast


def _ball(radius_mm: float, spacing) -> np.ndarray:
    r = [max(1, int(round(radius_mm / s))) for s in spacing]
    zz, yy, xx = np.ogrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    return (zz / r[0]) ** 2 + (yy / r[1]) ** 2 + (xx / r[2]) ** 2 <= 1.0


def extract_brain_mask(vol: Volume, already_skull_stripped: bool = False) -> np.ndarray:
    """Rough classical skull strip: threshold, erode, largest blob, dilate back."""
    sm = ndimage.gaussian_filter(vol.data, sigma=1.0)
    positive = sm[sm > 0]
    if positive.size < 100:
        return np.zeros(vol.shape, dtype=bool)
    t = threshold_otsu(positive)
    fg = sm > t * 0.6
    if already_skull_stripped:
        return ndimage.binary_fill_holes(vol.data > 0)
    ball = _ball(5.0, vol.spacing)
    eroded = ndimage.binary_erosion(fg, structure=ball)
    labels, n = ndimage.label(eroded)
    if n == 0:
        return ndimage.binary_fill_holes(fg)
    sizes = ndimage.sum(eroded, labels, index=np.arange(1, n + 1))
    core = labels == (int(np.argmax(sizes)) + 1)
    brain = ndimage.binary_dilation(core, structure=ball) & fg
    brain = ndimage.binary_fill_holes(brain)
    # Close small gaps (e.g. dark ventricles on FLAIR) slice by slice.
    for k in range(brain.shape[2]):
        brain[:, :, k] = ndimage.binary_fill_holes(brain[:, :, k])
    return brain


def _bias_corrected(vol: Volume, brain: np.ndarray) -> np.ndarray:
    """Divide out slow intensity drift (coil bias) estimated by heavy smoothing inside the brain.

    Two passes: voxels far from the first estimate (lesions, CSF) are left out of the second, so a
    large tumour does not raise the estimated field around itself and get divided away.
    """
    sigma = [25.0 / s for s in vol.spacing]
    data = vol.data.astype(np.float32)
    use = brain.copy()
    field = None
    for _ in range(2):
        num = ndimage.gaussian_filter(np.where(use, data, 0), sigma)
        den = ndimage.gaussian_filter(use.astype(np.float32), sigma)
        field = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
        ratio = np.where(field > 0, data / np.maximum(field, 1e-6), 0)
        use = brain & (np.abs(ratio - 1) < 0.25)
        if use.sum() < 0.3 * brain.sum():
            use = brain
            break
    ref = float(np.median(data[brain])) if brain.any() else 1.0
    out = np.where(field > 0, data / np.maximum(field, 1e-6) * ref, data)
    return ndimage.gaussian_filter(out.astype(np.float32), 0.7)


def candidate_mask_baseline(vol: Volume, brain: np.ndarray, sensitivity: float = 3.0,
                            polarity: str = "hyper", use_asymmetry: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (candidate mask, robust z-score map).

    Seeds are voxels that are strong outliers (and brighter than the mirrored
    hemisphere when asymmetry is on). Each seed is then grown to a fraction of its
    peak contrast, which follows oedema around a core without leaking into tissue.
    """
    sm = _bias_corrected(vol, brain)
    inner = ndimage.binary_erosion(brain, structure=_ball(2.0, vol.spacing))
    if inner.sum() < 100:
        return np.zeros(vol.shape, dtype=bool), np.zeros(vol.shape, dtype=np.float32)
    sign = 1.0 if polarity == "hyper" else -1.0
    mid_x = float(ndimage.center_of_mass(brain)[0])
    shift = 2 * mid_x - (vol.shape[0] - 1)
    mirrored = ndimage.shift(np.flip(sm, axis=0), (shift, 0, 0), order=1, mode="nearest")

    # Normal-tissue statistics from voxels that match the other hemisphere, so a large lesion does
    # not inflate the reference it is compared with.
    rough_med = float(np.median(sm[inner]))
    normal = inner & (np.abs(sm - mirrored) < 0.15 * abs(rough_med))
    vals = sm[normal] if normal.sum() > 0.2 * inner.sum() else sm[inner]
    med = float(np.median(vals))
    # Scale from the tissue distribution's tail (p90 or p10) rather than the MAD alone, so the
    # normal grey/white matter contrast is not counted as abnormal.
    tail = float(np.percentile(vals, 90 if sign > 0 else 10))
    mad = max(float(np.median(np.abs(vals - med))) * 1.4826, abs(tail - med) / 1.2816) or 1.0
    signed = sign * (sm - med)
    z = (signed / mad).astype(np.float32)

    high = inner & (z > sensitivity)
    if use_asymmetry:
        high &= sign * (sm - mirrored) / mad > sensitivity * 0.5
    high = ndimage.binary_opening(high)

    seeds, n = ndimage.label(high, structure=np.ones((3, 3, 3)))
    out = np.zeros(vol.shape, dtype=bool)
    margin = [int(np.ceil(30.0 / s)) for s in vol.spacing]
    for i, sl in enumerate(ndimage.find_objects(seeds), start=1):
        if sl is None:
            continue
        box = tuple(slice(max(0, a.start - m), min(dim, a.stop + m)) for a, m, dim in zip(sl, margin, vol.shape))
        seed = seeds[box] == i
        peak = float(np.percentile(signed[box][seed], 90))
        grown = inner[box] & (signed[box] > max(GROW_FRACTION * peak, 0.5 * sensitivity * mad))
        lab, _ = ndimage.label(grown, structure=np.ones((3, 3, 3)))
        keep = np.unique(lab[seed & grown])
        out[box] |= np.isin(lab, keep[keep > 0])
    return out, z


def analyze_brain(vol: Volume, sensitivity: float = 3.0, min_volume_mm3: float = 30.0,
                  polarity: str = "hyper", use_asymmetry: bool = True,
                  already_skull_stripped: bool = False, model=None) -> dict:
    brain = extract_brain_mask(vol, already_skull_stripped)
    warnings = []
    if brain.sum() * vol.voxel_volume_mm3 < 300_000:
        warnings.append("Extracted brain mask is small (<300 mL); check that this is a brain MRI and "
                        "consider the 'already skull-stripped' option.")

    if model is not None:
        cand, label_names = model.segment(vol)
        cand = cand.astype(np.int32) * brain
        z = None
        method = {"name": model.name, "kind": "pretrained-onnx", "details": model.describe()}
    else:
        mask, z = candidate_mask_baseline(vol, brain, sensitivity, polarity, use_asymmetry)
        cand = mask.astype(np.int32)
        label_names = {1: "hyperintense candidate" if polarity == "hyper" else "hypointense candidate"}
        method = {
            "name": "baseline-intensity-asymmetry",
            "kind": "classical",
            "details": {"sensitivity_z": sensitivity, "polarity": polarity, "use_asymmetry": use_asymmetry,
                        "min_volume_mm3": min_volume_mm3},
        }

    findings = []
    label_map = np.zeros(vol.shape, dtype=np.uint8)
    for class_id in [c for c in np.unique(cand) if c > 0]:
        labels, n = ndimage.label(cand == class_id, structure=np.ones((3, 3, 3)))
        for comp in range(1, n + 1):
            comp_mask = labels == comp
            count = int(comp_mask.sum())
            if count * vol.voxel_volume_mm3 < min_volume_mm3:
                continue
            findings.append((count, int(class_id), comp_mask))
    findings.sort(key=lambda f: -f[0])
    if len(findings) > MAX_FINDINGS:
        warnings.append(f"{len(findings)} candidates found; showing the {MAX_FINDINGS} largest. "
                        "Lower the sensitivity to reduce noise.")
        findings = findings[:MAX_FINDINGS]

    out = []
    for i, (_, class_id, comp_mask) in enumerate(findings, start=1):
        label_map[comp_mask] = i
        m = describe_component(vol, comp_mask)
        if z is not None:
            mean_z = float(z[comp_mask].mean())
            score = float(np.clip((mean_z - sensitivity * 0.5) / (sensitivity * 1.5), 0, 1))
            score *= float(np.clip(m["volume_mm3"] / (min_volume_mm3 * 4), 0.4, 1.0))
        else:
            mean_z = None
            score = 0.9
        c = m["centroid_voxel"]
        out.append({
            "id": i,
            "label_value": i,
            "type": label_names.get(class_id, f"class {class_id}"),
            "location": brain_location(c, brain, vol),
            "measurements": m,
            "mean_robust_z": None if mean_z is None else round(mean_z, 2),
            "score": round(score, 3),
            "slice_hint": {"axial": int(round(c[2])), "coronal": int(round(c[1])), "sagittal": int(round(c[0]))},
            "review": "pending",
        })

    return {
        "mode": "brain",
        "method": method,
        "summary": {
            "brain_volume_ml": round(float(brain.sum() * vol.voxel_volume_mm3 / 1000), 1),
            "candidate_count": len(out),
            "total_candidate_volume_ml": round(sum(f["measurements"]["volume_ml"] for f in out), 3),
        },
        "findings": out,
        "warnings": warnings,
        "label_map": label_map,
        "context_mask": brain,
    }
