"""Brain MRI: detect and segment candidate lesions.

Two engines produce a binary candidate mask, then shared code turns it into
measured findings (location, volume, diameters):

* ``onnx``: a user-supplied segmentation model (see ``models.py``).
* ``heuristic`` (default fallback): intensity outlier detection inside an
  automatically estimated brain mask, with a left/right asymmetry check that
  suppresses symmetric anatomy such as the ventricles. It works best on
  FLAIR or post-contrast T1 and is meant to point a reader at regions worth a
  look, not to replace a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import models
from .volume import Volume

LABEL_LESION = 1


@dataclass
class BrainParams:
    sensitivity: float = 3.0  # robust z-score threshold; lower = more sensitive
    polarity: str = "hyper"  # "hyper" (FLAIR/T2/T1c), "hypo" (T1 holes), "both"
    min_volume_ml: float = 0.05
    require_asymmetry: bool = True
    engine: str = "auto"  # "auto" | "heuristic" | "onnx"


def analyze_brain(vol: Volume, params: BrainParams) -> tuple[np.ndarray, dict]:
    data = vol.data.astype(np.float32)
    brain = estimate_brain_mask(data, vol.spacing)

    engine = params.engine
    model = models.get_model("brain") if engine in ("auto", "onnx") else None
    if engine == "onnx" and model is None:
        raise ValueError("No brain ONNX model configured (set MRI_BRAIN_ONNX)")

    if model is not None:
        candidates = model.segment(data) & brain
        z = _robust_z(data, brain)
        asym = _asymmetry_z(data, brain, vol.spacing)
        engine_used = f"onnx:{model.name}"
    else:
        candidates, z, asym = _heuristic_candidates(data, brain, vol.spacing, params)
        engine_used = "heuristic"

    labels, n = ndimage.label(candidates, structure=np.ones((3, 3, 3)))
    findings = []
    mask = np.zeros(data.shape, np.uint8)
    min_vox = max(1, int(round(params.min_volume_ml / vol.voxel_volume_ml)))
    objects = ndimage.find_objects(labels)
    brain_box = _bbox(brain)
    midline = _midline_x(brain)

    for idx, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        comp = labels[sl] == idx
        nvox = int(comp.sum())
        if nvox < min_vox:
            continue
        offset = np.array([s.start for s in sl])
        coords = np.argwhere(comp) + offset
        centroid = coords.mean(axis=0)
        mean_z = float(np.abs(z[tuple(coords.T)]).mean())
        mean_asym = float(asym[tuple(coords.T)].mean())
        # Heuristic engine: drop components that look like symmetric anatomy.
        if model is None and params.require_asymmetry and mean_asym < 1.0:
            continue

        mask[tuple(coords.T)] = LABEL_LESION
        extent_mm = (coords.max(axis=0) - coords.min(axis=0) + 1) * np.array(vol.spacing)
        volume_ml = nvox * vol.voxel_volume_ml
        findings.append(
            {
                "id": f"B{len(findings) + 1}",
                "kind": "lesion",
                "label": "Candidate lesion",
                "volumeMl": round(volume_ml, 3),
                "voxels": nvox,
                "centroidVoxel": [round(float(c), 1) for c in centroid],
                "centroidWorldMm": [round(v, 1) for v in vol.voxel_to_world(centroid)],
                "extentMm": [round(float(e), 1) for e in extent_mm],
                "maxDiameterMm": round(float(_max_diameter(coords, vol.spacing)), 1),
                "location": _describe_location(centroid, brain_box, midline),
                "meanIntensityZ": round(mean_z, 2),
                "asymmetryZ": round(mean_asym, 2),
                "confidence": round(_confidence(mean_z, mean_asym, volume_ml, model is not None), 2),
                "bbox": [[int(s.start), int(s.stop - 1)] for s in sl],
            }
        )

    findings.sort(key=lambda f: -f["volumeMl"])
    for i, f in enumerate(findings, start=1):
        f["id"] = f"B{i}"

    summary = {
        "mode": "brain",
        "engine": engine_used,
        "brainVolumeMl": round(float(brain.sum() * vol.voxel_volume_ml), 1),
        "totalLesionVolumeMl": round(sum(f["volumeMl"] for f in findings), 3),
        "candidateCount": len(findings),
        "params": params.__dict__,
        "labels": {str(LABEL_LESION): "Candidate lesion"},
        "notes": [
            "Candidate findings for radiologist review. Not a diagnosis.",
            "Heuristic engine assumes a single FLAIR, T2 or post-contrast T1 series; symmetric or bilateral disease may be missed when the asymmetry check is on."
            if model is None
            else "Model output is limited to the estimated brain mask.",
            "Lobe labels are approximate and derived from position within the brain bounding box, not an atlas.",
        ],
    }
    return mask, {"summary": summary, "findings": findings}


# Brain mask -------------------------------------------------------------------


def estimate_brain_mask(data: np.ndarray, spacing) -> np.ndarray:
    """Rough skull strip: threshold, erode to cut the scalp, keep the largest blob."""
    smooth = ndimage.gaussian_filter(data, 1.0)
    fg = smooth > _otsu(smooth[smooth > np.percentile(smooth, 2)])
    if fg.sum() == 0:
        return np.ones(data.shape, bool)
    head = ndimage.binary_fill_holes(_largest(fg))

    # Already skull-stripped (e.g. BraTS)? Then the head is the brain.
    inside = smooth[head]
    if np.percentile(inside, 5) > 0.25 * np.percentile(inside, 95):
        return head

    r = max(1, int(round(4.0 / min(spacing))))
    tissue = head & (smooth > _otsu(inside))
    core = ndimage.binary_erosion(tissue, _ball(r))
    if core.sum() == 0:
        return head
    core = _largest(core)
    brain = ndimage.binary_dilation(core, _ball(r)) & head
    brain = ndimage.binary_closing(brain, _ball(max(1, r // 2)))
    brain = ndimage.binary_fill_holes(brain)
    # The fill above restores dark interior structures (ventricles) as brain.
    return brain


def _heuristic_candidates(data, brain, spacing, params: BrainParams):
    z = _robust_z(data, brain)
    asym = _asymmetry_z(data, brain, spacing)
    if params.polarity == "hypo":
        z, asym = -z, -asym
    elif params.polarity == "both":
        z, asym = np.abs(z), np.abs(asym)
    z_s = ndimage.gaussian_filter(z, 0.7)
    cand = (z_s > params.sensitivity) & brain
    # Trim the rim: partial-volume voxels at the brain edge are noisy.
    rim = brain & ~ndimage.binary_erosion(brain, _ball(max(1, int(round(3.0 / min(spacing))))))
    cand &= ~rim
    cand = ndimage.binary_opening(cand, np.ones((2, 2, 2)))
    return cand, z, asym


def _robust_z(data, brain):
    vals = data[brain]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 or 1e-6
    return (data - med) / mad


def _asymmetry_z(data, brain, spacing):
    """Signal minus its mirror image across the estimated midline, as a z-score."""
    mid = _midline_x(brain)
    xs = np.arange(data.shape[0])
    mirror_idx = np.clip(np.round(2 * mid - xs).astype(int), 0, data.shape[0] - 1)
    smooth = ndimage.gaussian_filter(data, 1.5 / np.array(spacing))
    diff = smooth - smooth[mirror_idx]
    vals = diff[brain]
    mad = float(np.median(np.abs(vals - np.median(vals)))) * 1.4826 or 1e-6
    return diff / mad


def _midline_x(brain) -> float:
    xs = np.nonzero(brain.any(axis=(1, 2)))[0]
    return float((xs[0] + xs[-1]) / 2.0) if xs.size else brain.shape[0] / 2.0


def _describe_location(c, box, midline) -> dict:
    (x0, x1), (y0, y1), (z0, z1) = box
    side = "right" if c[0] > midline + 1 else "left" if c[0] < midline - 1 else "midline"
    fy = (c[1] - y0) / max(1, y1 - y0)  # 0 = posterior, 1 = anterior
    fz = (c[2] - z0) / max(1, z1 - z0)  # 0 = inferior, 1 = superior
    fx = abs(c[0] - midline) / max(1, (x1 - x0) / 2)
    if fz < 0.3 and fy < 0.35:
        region = "posterior fossa / cerebellum"
    elif fy > 0.62:
        region = "frontal"
    elif fy < 0.25:
        region = "occipital"
    elif fz < 0.5 and fx > 0.45:
        region = "temporal"
    else:
        region = "parietal"
    depth = "deep / periventricular" if fx < 0.4 and 0.3 < fz < 0.75 and 0.3 < fy < 0.7 else "lobar"
    return {"side": side, "region": region, "depth": depth, "text": f"{side} {region}, {depth} (approx.)"}


def _confidence(z, asym, volume_ml, model_based) -> float:
    if model_based:
        return 0.8
    s = 1 / (1 + np.exp(-(z - 4.0))) * 0.6 + 1 / (1 + np.exp(-(asym - 2.5))) * 0.3
    s += 0.1 * min(1.0, volume_ml / 1.0)
    return float(np.clip(s, 0, 1))


# Helpers ----------------------------------------------------------------------


def _max_diameter(coords: np.ndarray, spacing) -> float:
    pts = coords * np.array(spacing)
    if len(pts) > 1500:
        pts = pts[np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
    # Farthest pair approximation via extreme points along many directions.
    dirs = np.random.default_rng(1).normal(size=(64, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    proj = pts @ dirs.T
    return float((proj.max(axis=0) - proj.min(axis=0)).max()) + float(min(spacing))


def _otsu(vals: np.ndarray) -> float:
    hist, edges = np.histogram(vals, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _largest(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _ball(r: int) -> np.ndarray:
    g = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (g**2).sum(axis=0) <= r * r


def _bbox(mask):
    out = []
    for ax in range(3):
        idx = np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        out.append((int(idx[0]), int(idx[-1])) if idx.size else (0, mask.shape[ax] - 1))
    return out
