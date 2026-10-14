"""Lumbar spine MRI: disc levels and central canal stenosis on sagittal T2.

Focused use case, chosen because it is the most common lumbar MRI question
and can be measured on a single mid-sagittal T2 image:

1. Find the mid-sagittal slice and trace the CSF-filled central canal (bright
   band on T2) with a ridge filter plus dynamic programming.
2. Measure the canal's anteroposterior (AP) diameter along its length.
3. Find intervertebral discs along the vertebral column just anterior to the
   canal, and label levels upward from the lowest disc (L5-S1 by default).
4. Per level, report canal AP diameter, posterior disc displacement into the
   canal, disc height and T2 signal, and flag candidate stenosis, disc
   bulge/protrusion, height loss and desiccation.

Thresholds follow common radiology rules of thumb (central lumbar canal AP
< 10 mm absolute stenosis, 10-12 mm relative) and are exposed as parameters.
Everything is a candidate for review: level labels in particular must be
checked against the full spine (transitional vertebrae are common).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from .volume import Volume

LABEL_CANAL, LABEL_DISC, LABEL_DISC_FLAGGED, LABEL_STENOSIS = 2, 3, 4, 5

LEVEL_NAMES = ["T10-T11", "T11-T12", "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


@dataclass
class SpineParams:
    lowest_level: str = "L5-S1"  # label given to the most caudal disc found
    absolute_stenosis_mm: float = 10.0
    relative_stenosis_mm: float = 12.0
    narrowing_ratio: float = 0.67  # canal AP at disc / reference AP
    displacement_mm: float = 3.0  # posterior disc displacement worth flagging
    sagittal_slice: int | None = None  # override automatic mid-sagittal choice
    canal_seed: list[float] | None = None  # optional [y, z] voxel on the canal


def analyze_spine(vol: Volume, params: SpineParams) -> tuple[np.ndarray, dict]:
    sx, sy, sz = vol.spacing
    res = float(np.clip(min(sy, sz), 0.5, 1.0))
    notes = [
        "Candidate findings for radiologist review. Not a diagnosis.",
        f"Levels are counted upward assuming the lowest disc found is {params.lowest_level}; confirm on a full-spine localiser.",
        "Canal AP diameter is measured on one mid-sagittal image; lateral recess and foraminal stenosis are not assessed.",
    ]
    if sz > 3.0 or sy > 3.0:
        notes.append("In-plane sagittal resolution is coarse (likely an axial acquisition); measurements are unreliable.")

    if params.sagittal_slice is not None:
        candidates = [int(np.clip(params.sagittal_slice, 0, vol.shape[0] - 1))]
    else:
        n = vol.shape[0]
        lo, hi = int(n * 0.3), int(np.ceil(n * 0.7))
        candidates = sorted(set(range(lo, max(hi, lo + 1)))) if n > 2 else list(range(n))

    best = None
    for i in candidates:
        img = _resample_slice(vol.data[i], (sy, sz), res)
        seed = None
        if params.canal_seed is not None:
            seed = (params.canal_seed[0] * sy / res, params.canal_seed[1] * sz / res)
        trace = _trace_canal(img, res, seed)
        if trace is None:
            continue
        # Prefer the central slice when neighbours score about the same.
        trace["score"] *= 1.0 - 0.02 * abs(i - (vol.shape[0] - 1) / 2)
        if best is None or trace["score"] > best[2]["score"]:
            best = (i, img, trace)
    if best is None:
        raise ValueError("Could not find the spinal canal. Is this a sagittal T2 spine series?")

    slice_idx, img, trace = best
    center, ant, post, diam = trace["center"], trace["anterior"], trace["posterior"], trace["diameter"]
    valid = trace["valid"]

    # Vertebral column profile just anterior to the canal.
    ant_s = _smooth_edge(ant, valid, int(round(15 / res)))
    profile = _column_profile(img, ant_s, valid, res)
    body_level = float(np.median(profile[valid]))
    discs = _find_discs(profile, valid, body_level, res)
    if not discs:
        raise ValueError("Canal found, but no intervertebral discs could be identified")

    # Reference canal size and anterior wall from mid-body rows (away from discs).
    disc_zone = np.zeros_like(valid)
    for zc in discs:
        disc_zone[max(0, zc - int(12 / res)) : zc + int(12 / res) + 1] = True
    body_rows = valid & ~disc_zone & (diam > 0)
    ref_ap = float(np.median(diam[body_rows])) if body_rows.any() else float(np.median(diam[valid]))
    ref_ant = _interp_rows(ant, body_rows)

    level_names = _level_names(len(discs), params.lowest_level)
    disc_signals = [profile[zc] for zc in discs]
    bright = [s for s in disc_signals if s > body_level]
    ref_signal = float(np.median(bright)) if bright else float(np.median(disc_signals))

    heights = [_disc_height(profile, zc, body_level, res) for zc in discs]
    ref_height = float(np.median(heights))

    mask2d = np.zeros(img.shape, np.uint8)
    rows = np.arange(img.shape[1])
    for z in rows[valid]:
        if diam[z] > 0:
            mask2d[int(post[z]) : int(ant[z]) + 1, z] = LABEL_CANAL

    findings = []
    # Cranial -> caudal for display (largest z first).
    for name, zc, h in sorted(zip(level_names, discs, heights), key=lambda t: -t[1]):
        half = max(int(round(4 / res)), int(round((h / 2 + 2) / res)))
        win = slice(max(0, zc - half), min(len(diam), zc + half + 1))
        wvalid = valid[win]
        ap = float(diam[win][wvalid].min()) if wvalid.any() else float("nan")
        disp = float(np.nanmax((ref_ant[win] - ant[win])[wvalid]) * res) if wvalid.any() else 0.0
        signal_ratio = float(profile[zc] / ref_signal) if ref_signal else 1.0
        height_ratio = h / ref_height if ref_height else 1.0

        flags = []
        if ap < params.absolute_stenosis_mm:
            flags.append({"code": "stenosis_severe", "text": f"Canal AP {ap:.1f} mm (< {params.absolute_stenosis_mm:g} mm): candidate absolute central stenosis"})
        elif ap < params.relative_stenosis_mm:
            flags.append({"code": "stenosis_relative", "text": f"Canal AP {ap:.1f} mm: candidate relative central stenosis"})
        elif ref_ap and ap / ref_ap < params.narrowing_ratio:
            flags.append({"code": "canal_narrowing", "text": f"Canal narrows to {ap / ref_ap:.0%} of reference"})
        if disp >= params.displacement_mm:
            flags.append({"code": "disc_displacement", "text": f"Posterior disc displacement {disp:.1f} mm: candidate bulge/protrusion"})
        if height_ratio < 0.8:
            flags.append({"code": "height_loss", "text": f"Disc height {h:.1f} mm ({height_ratio:.0%} of median): candidate height loss"})
        if signal_ratio < 0.6:
            flags.append({"code": "low_signal", "text": f"Disc T2 signal {signal_ratio:.0%} of reference: candidate desiccation"})

        severity = 0
        if any(f["code"] == "stenosis_severe" for f in flags):
            severity = 3
        elif any(f["code"] in ("stenosis_relative", "disc_displacement") for f in flags):
            severity = 2
        elif flags:
            severity = 1

        _paint_disc(mask2d, img, ant_s, zc, h, body_level, res, LABEL_DISC_FLAGGED if flags else LABEL_DISC)
        if severity >= 2:
            canal_rows = (mask2d[:, win] == LABEL_CANAL)
            mask2d[:, win][canal_rows] = LABEL_STENOSIS

        voxel = [float(slice_idx), float(center[zc] * res / sy), float(zc * res / sz)]
        findings.append(
            {
                "id": name,
                "kind": "disc_level",
                "label": name,
                "level": name,
                "canalApMm": round(ap, 1),
                "canalApRatio": round(ap / ref_ap, 2) if ref_ap else None,
                "posteriorDisplacementMm": round(max(0.0, disp), 1),
                "discHeightMm": round(h, 1),
                "discHeightRatio": round(height_ratio, 2),
                "discSignalRatio": round(signal_ratio, 2),
                "flags": flags,
                "severity": severity,
                "centroidVoxel": [round(v, 1) for v in voxel],
                "centroidWorldMm": [round(v, 1) for v in vol.voxel_to_world(voxel)],
            }
        )

    mask = np.zeros(vol.shape, np.uint8)
    mask[slice_idx] = _fit(ndimage.zoom(mask2d, (vol.shape[1] / img.shape[0], vol.shape[2] / img.shape[1]), order=0), vol.shape[1:])

    zs = rows[valid]
    profile_out = {
        "zMm": [round(float(vol.voxel_to_world([slice_idx, 0, z * res / sz])[2]), 1) for z in zs[:: max(1, len(zs) // 300)]],
        "apMm": [round(float(diam[z]), 1) for z in zs[:: max(1, len(zs) // 300)]],
        "levels": [{"level": f["level"], "zMm": f["centroidWorldMm"][2]} for f in findings],
    }

    summary = {
        "mode": "spine",
        "engine": "heuristic",
        "useCase": "Lumbar disc levels and central canal stenosis (sagittal T2)",
        "sagittalSlice": int(slice_idx),
        "referenceCanalApMm": round(ref_ap, 1),
        "discsFound": len(discs),
        "flaggedLevels": [f["level"] for f in findings if f["flags"]],
        "params": params.__dict__,
        "labels": {
            str(LABEL_CANAL): "Spinal canal (CSF)",
            str(LABEL_DISC): "Disc",
            str(LABEL_DISC_FLAGGED): "Disc with candidate abnormality",
            str(LABEL_STENOSIS): "Candidate canal narrowing",
        },
        "canalProfile": profile_out,
        "notes": notes,
    }
    return mask, {"summary": summary, "findings": findings}


# Canal tracing ----------------------------------------------------------------


def _resample_slice(sl: np.ndarray, spacing, res: float) -> np.ndarray:
    img = ndimage.zoom(sl.astype(np.float32), (spacing[0] / res, spacing[1] / res), order=1)
    p = np.percentile(img, 99.5) or 1.0
    return np.clip(img / p, 0, 1.5)


def _trace_canal(img: np.ndarray, res: float, seed=None) -> dict | None:
    """img is (Y, Z) at ``res`` mm: axis 0 posterior->anterior, axis 1 inferior->superior."""
    ny, nz = img.shape
    if ny < 20 or nz < 20:
        return None
    sm = ndimage.gaussian_filter(img, 0.8)
    w = max(2, int(round(5 / res)))
    c = ndimage.uniform_filter1d(sm, 2 * w + 1, axis=0, mode="nearest")
    flank = 0.5 * (np.roll(c, 2 * w, axis=0) + np.roll(c, -2 * w, axis=0))
    ridge = np.clip(c - flank, 0, None) * c
    ridge[: int(ny * 0.12)] = 0
    ridge[int(ny * 0.88) :] = 0
    if seed is not None:
        yy = np.arange(ny)[:, None]
        ridge *= np.exp(-((yy - seed[0]) ** 2) / (2 * (15 / res) ** 2))

    # Viterbi: maximise ridge along z with bounded lateral jumps.
    jump = max(1, int(round(1.0 / res)))
    penalty = 0.02
    acc = ridge[:, 0].copy()
    back = np.zeros((ny, nz), np.int16)
    offsets = range(-jump, jump + 1)
    for z in range(1, nz):
        stack = np.stack([np.roll(acc, o) - penalty * abs(o) for o in offsets])
        for k, o in enumerate(offsets):  # invalidate wrap-around
            if o > 0:
                stack[k, :o] = -np.inf
            elif o < 0:
                stack[k, o:] = -np.inf
        k = np.argmax(stack, axis=0)
        acc = stack[k, np.arange(ny)] + ridge[:, z]
        back[:, z] = np.array(list(offsets))[k]
    path = np.zeros(nz, int)
    path[-1] = int(np.argmax(acc))
    for z in range(nz - 1, 0, -1):
        path[z - 1] = path[z] - back[path[z], z]

    path_vals = sm[path, np.arange(nz)]
    csf = float(np.percentile(path_vals, 75))
    if csf <= 0:
        return None
    thr = 0.6 * csf
    reach = int(round(12 / res))
    ant = np.zeros(nz)
    post = np.zeros(nz)
    diam = np.zeros(nz)
    for z in range(nz):
        lo, hi = max(0, path[z] - reach), min(ny, path[z] + reach + 1)
        bright = sm[lo:hi, z] > thr
        labels, n = ndimage.label(bright)
        if n == 0:
            ant[z] = post[z] = path[z]
            continue
        # Bright run closest to the path centre.
        centres = ndimage.center_of_mass(bright, labels, range(1, n + 1))
        k = int(np.argmin([abs(cm[0] + lo - path[z]) for cm in centres])) + 1
        idx = np.nonzero(labels == k)[0] + lo
        post[z], ant[z] = idx.min(), idx.max()
        diam[z] = (idx.max() - idx.min() + 1) * res

    valid = ridge[path, np.arange(nz)] > 0.05 * max(1e-6, float(np.percentile(ridge[path, np.arange(nz)], 90)))
    valid &= (diam > 0) | ndimage.binary_dilation(diam > 0, iterations=int(10 / res))
    score = float(ridge[path, np.arange(nz)][valid].sum())
    return {"center": path.astype(float), "anterior": ant, "posterior": post, "diameter": diam, "valid": valid, "score": score}


def _smooth_edge(edge, valid, size) -> np.ndarray:
    filled = _interp_rows(edge, valid)
    # Max filter keeps the edge at the vertebral body wall even where a disc bulges posteriorly.
    return ndimage.median_filter(ndimage.maximum_filter1d(filled, size), size)


def _interp_rows(values, ok) -> np.ndarray:
    z = np.arange(len(values))
    if ok.sum() < 2:
        return np.asarray(values, float)
    return np.interp(z, z[ok], np.asarray(values, float)[ok])


def _column_profile(img, ant_s, valid, res) -> np.ndarray:
    ny, nz = img.shape
    prof = np.zeros(nz)
    a, b = int(round(6 / res)), int(round(24 / res))
    for z in range(nz):
        y0, y1 = int(ant_s[z]) + a, min(ny, int(ant_s[z]) + b)
        prof[z] = img[y0:y1, z].mean() if y1 > y0 else 0.0
    return ndimage.gaussian_filter1d(prof, 0.8 / res)


def _find_discs(profile, valid, body_level, res) -> list[int]:
    dev = ndimage.gaussian_filter1d((profile - body_level) ** 2, 3 / res)
    dev[~valid] = 0
    if dev.max() <= 0:
        return []
    peaks, _ = find_peaks(dev, distance=int(20 / res), prominence=0.15 * dev.max())
    peaks = list(peaks)
    if len(peaks) >= 3:
        # A disc with near-body signal can be missed: fill gaps of ~2x the usual spacing.
        gaps = np.diff(peaks)
        med = float(np.median(gaps))
        filled = [peaks[0]]
        for p, g in zip(peaks[1:], gaps):
            if g > 1.6 * med:
                k = int(round(g / med))
                filled += [int(filled[-1] + g * j / k) for j in range(1, k)]
            filled.append(p)
        peaks = filled
    return [int(p) for p in peaks]


def _disc_height(profile, zc, body_level, res) -> float:
    """Approximate disc height: full width at half maximum of the disc's
    departure from vertebral body signal along the column."""
    sign = np.sign(profile[zc] - body_level) or 1.0
    amp = abs(profile[zc] - body_level)
    inside = sign * (profile - body_level) > 0.5 * amp
    lo = hi = zc
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < len(profile) - 1 and inside[hi + 1]:
        hi += 1
    return float((hi - lo + 1) * res)


def _paint_disc(mask2d, img, ant_s, zc, h, body_level, res, label):
    half = int(round(h / 2 / res))
    sign = np.sign(img[int(ant_s[zc]) + int(10 / res), zc] - body_level) or 1.0
    for z in range(max(0, zc - half), min(img.shape[1], zc + half + 1)):
        y0 = int(ant_s[z]) + int(round(1 / res))
        y1 = min(img.shape[0], int(ant_s[z]) + int(round(32 / res)))
        col = img[y0:y1, z]
        sel = sign * (col - body_level) > 0.08
        mask2d[y0:y1, z][sel & (mask2d[y0:y1, z] == 0)] = label


def _level_names(n: int, lowest: str) -> list[str]:
    """Names for ``n`` discs ordered bottom (index 0 = most caudal) up."""
    try:
        start = LEVEL_NAMES.index(lowest)
    except ValueError:
        start = len(LEVEL_NAMES) - 1
    names = []
    for k in range(n):
        j = start - k
        names.append(LEVEL_NAMES[j] if j >= 0 else f"disc +{k}")
    return names


def _fit(arr, shape):
    out = np.zeros(shape, arr.dtype)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
    out[sl] = arr[sl]
    return out
