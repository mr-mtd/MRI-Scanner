"""Focused lumbar spine analysis on sagittal T2-weighted MRI.

Scope (deliberately narrow): central canal narrowing and disc abnormalities of
the lumbar spine, measured on the midsagittal slice of a sagittal T2 series.

Why this use case: on sagittal T2 the CSF in the thecal sac is the brightest
structure, so the canal can be outlined without a trained model, and the
anteroposterior (AP) canal diameter is a standard, measurable stenosis criterion
(roughly <12 mm relative, <10 mm absolute stenosis). Disc protrusion into the
canal and loss of normal disc T2 signal (desiccation) are measured on the same
slice. Foraminal stenosis, facet disease, fractures, cord signal and tumours
are out of scope and are not assessed.

Pipeline:
1. Pick the midsagittal slice with the most continuous bright canal.
2. Segment the thecal sac (brightest intensity class, tall thin component).
3. Measure AP diameter along the craniocaudal axis.
4. Locate discs from the intensity profile just anterior to the canal and name
   levels assuming the most caudal disc is L5-S1 (flagged for verification).
5. Report canal narrowing, posterior disc protrusion and disc signal loss.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.filters import threshold_multiotsu

from .volume import Volume

LEVEL_NAMES = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1", "T11-T12"]
CANAL_CONTEXT_LABEL = 250
DISC_CONTEXT_LABEL = 251

RELATIVE_STENOSIS_MM = 12.0
ABSOLUTE_STENOSIS_MM = 10.0
SEVERE_STENOSIS_MM = 7.0


def _segment_canal(sl: np.ndarray, dy: float, dz: float):
    """Segment the thecal sac in one sagittal slice (axes: y posterior->anterior, z inferior->superior)."""
    sm = ndimage.gaussian_filter(sl, 0.8)
    if np.ptp(sm) <= 0:
        return None
    try:
        thresholds = threshold_multiotsu(sm, classes=4)
    except ValueError:
        return None
    best = None
    for t in (thresholds[-1], thresholds[-2]):
        labels, n = ndimage.label(sm > t)
        for i in range(1, n + 1):
            comp = labels == i
            zs = np.where(comp.any(axis=0))[0]
            z_extent = (zs.max() - zs.min() + 1) * dz
            if z_extent < 0.35 * sl.shape[1] * dz:
                continue
            widths = comp.sum(axis=0)[zs] * dy
            med_w = float(np.median(widths))
            if not 3.0 <= med_w <= 30.0:
                continue
            score = z_extent * float(sm[comp].mean())
            if best is None or score > best[0]:
                best = (score, comp)
        if best is not None:
            break
    if best is None:
        return None
    comp = ndimage.binary_closing(best[1], structure=np.ones((3, 3)))
    comp = ndimage.binary_fill_holes(comp)
    return comp, best[0]


def _choose_midsagittal(vol: Volume):
    nx = vol.shape[0]
    dy, dz = vol.spacing[1], vol.spacing[2]
    lo, hi = int(nx * 0.2), int(np.ceil(nx * 0.8))
    best = None
    for x in range(lo, max(hi, lo + 1)):
        found = _segment_canal(vol.data[x], dy, dz)
        if found is None:
            continue
        canal, seg_score = found
        score = seg_score * (1.0 - 0.2 * abs(x - (nx - 1) / 2) / nx)
        if best is None or score > best[0]:
            best = (score, x, canal)
    return (best[1], best[2]) if best else (None, None)


def _runs(flags: np.ndarray):
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def analyze_spine(vol: Volume, min_protrusion_mm: float = 3.0) -> dict:
    dy, dz = vol.spacing[1], vol.spacing[2]
    warnings = [
        "Spine analysis assumes a sagittal T2-weighted lumbar series and measures the midsagittal slice only.",
        "Level names assume the most caudal detected disc is L5-S1; verify against a localiser "
        "(transitional anatomy changes numbering).",
    ]
    x, canal = _choose_midsagittal(vol)
    label_map = np.zeros(vol.shape, dtype=np.uint8)
    base = {"mode": "spine", "label_map": label_map, "context_mask": None,
            "method": {"name": "baseline-lumbar-canal-disc", "kind": "classical",
                       "details": {"relative_stenosis_mm": RELATIVE_STENOSIS_MM,
                                   "absolute_stenosis_mm": ABSOLUTE_STENOSIS_MM,
                                   "min_protrusion_mm": min_protrusion_mm}}}
    if canal is None:
        warnings.insert(0, "Could not find a continuous bright spinal canal. Is this a sagittal T2 lumbar series?")
        return {**base, "summary": {"candidate_count": 0}, "findings": [], "warnings": warnings,
                "profile": [], "discs": []}

    sl = ndimage.gaussian_filter(vol.data[x], 0.8)
    nz = sl.shape[1]
    rows = np.where(canal.any(axis=0))[0]
    z0, z1 = int(rows.min()), int(rows.max())

    ap = np.full(nz, np.nan)
    ant_edge = np.full(nz, np.nan)
    for z in range(z0, z1 + 1):
        ys = np.where(canal[:, z])[0]
        if ys.size:
            ap[z] = ys.size * dy
            ant_edge[z] = ys.max()
    valid = ~np.isnan(ap)
    ap_s = ap.copy()
    k = max(1, int(round(2.0 / dz)))
    ap_s[valid] = ndimage.median_filter(ap[valid], size=2 * k + 1, mode="nearest")
    reference_ap = float(np.nanmedian(ap_s))

    # --- discs: intensity just anterior to the canal, along z -----------------------
    strip_lo = int(round(4.0 / dy))
    strip_hi = int(round(28.0 / dy))
    edge_s = ant_edge.copy()
    edge_s[valid] = ndimage.median_filter(ant_edge[valid], size=int(round(30.0 / dz)) | 1, mode="nearest")
    strip = np.full(nz, np.nan)
    for z in range(z0, z1 + 1):
        if np.isnan(edge_s[z]):
            continue
        y_a = int(edge_s[z]) + strip_lo
        y_b = min(sl.shape[0], int(edge_s[z]) + strip_hi)
        if y_b > y_a:
            strip[z] = float(np.median(sl[y_a:y_b, z]))
    sv = ~np.isnan(strip)
    discs = []
    if sv.sum() > int(40 / dz):
        prof = strip.copy()
        prof[~sv] = np.nanmedian(strip)
        baseline = ndimage.median_filter(prof, size=int(round(40.0 / dz)) | 1, mode="nearest")
        dev = np.abs(prof - baseline)
        noise = float(np.median(dev[sv])) * 1.4826 + 1e-6
        peaks, _ = find_peaks(dev, height=max(4 * noise, 0.08 * float(np.nanmax(prof))),
                              distance=int(round(20.0 / dz)))
        edge_guard = int(round(10.0 / dz))
        peaks = [int(p) for p in peaks if sv[p] and z0 + edge_guard <= p <= z1 - edge_guard]
        peaks.sort()
        # Signal of each disc: mean in the strip over the disc's thickness.
        half = max(1, int(round(3.0 / dz)))
        for idx, p in enumerate(peaks):
            name = LEVEL_NAMES[idx] if idx < len(LEVEL_NAMES) else f"disc {idx + 1} above L5-S1"
            discs.append({"level": name, "z_index": p, "z_mm": round(p * dz, 1),
                          "signal": float(np.nanmean(strip[max(0, p - half):p + half + 1])),
                          "body_signal": float(baseline[p])})
        if not discs:
            warnings.append("No intervertebral discs were located; level names are unavailable.")

    def nearest_level(z):
        if not discs:
            return None
        d = min(discs, key=lambda d: abs(d["z_index"] - z))
        return d["level"] if abs(d["z_index"] - z) * dz < 15 else None

    findings = []

    def add_finding(ftype, description, level, measurements, severity, score, region_mask, z_center):
        fid = len(findings) + 1
        for dx in (-1, 0, 1):
            xi = x + dx
            if 0 <= xi < vol.shape[0]:
                label_map[xi][region_mask & (label_map[xi] == 0)] = fid
        ys, zs = np.where(region_mask)
        cvox = [float(x), float(ys.mean()) if ys.size else 0.0, float(z_center)]
        findings.append({
            "id": fid, "label_value": fid, "type": ftype, "description": description,
            "location": {"level": level or "unassigned", "slice": "midsagittal", "sagittal_index": int(x)},
            "measurements": {**measurements,
                             "centroid_voxel": [round(v, 1) for v in cvox],
                             "centroid_ras_mm": [round(v, 1) for v in vol.voxel_to_ras(cvox)],
                             "area_mm2": round(float(region_mask.sum() * dy * dz), 1)},
            "severity": severity, "score": round(float(score), 3),
            "slice_hint": {"sagittal": int(x), "axial": int(round(z_center)), "coronal": int(round(cvox[1]))},
            "review": "pending",
        })

    # --- canal narrowing -----------------------------------------------------------
    flags = valid & ((ap_s < RELATIVE_STENOSIS_MM) | (ap_s < 0.67 * reference_ap))
    flags[:z0 + int(5 / dz)] = False  # ignore the tapering ends of the segmented sac
    flags[z1 - int(5 / dz):] = False
    for a, b in _runs(flags):
        if (b - a + 1) * dz < 2.0:
            continue
        seg = ap_s[a:b + 1]
        zmin = a + int(np.nanargmin(seg))
        min_ap = float(np.nanmin(seg))
        if min_ap < SEVERE_STENOSIS_MM:
            sev = "severe range (<7 mm AP)"
        elif min_ap < ABSOLUTE_STENOSIS_MM:
            sev = "absolute stenosis range (<10 mm AP)"
        elif min_ap < RELATIVE_STENOSIS_MM:
            sev = "relative stenosis range (10-12 mm AP)"
        else:
            sev = "focal narrowing relative to adjacent canal"
        region = np.zeros_like(canal)
        region[:, a:b + 1] = canal[:, a:b + 1]
        level = nearest_level(zmin)
        add_finding("central canal narrowing",
                    f"Thecal sac AP diameter narrows to {min_ap:.1f} mm "
                    f"(reference {reference_ap:.1f} mm along this series).",
                    level,
                    {"min_ap_diameter_mm": round(min_ap, 1), "reference_ap_diameter_mm": round(reference_ap, 1),
                     "ap_ratio": round(min_ap / reference_ap, 2),
                     "craniocaudal_length_mm": round((b - a + 1) * dz, 1)},
                    sev, min(1.0, 0.4 + (reference_ap - min_ap) / reference_ap), region, zmin)

    # --- disc protrusion and signal loss --------------------------------------------
    if discs:
        signals = np.array([d["signal"] for d in discs])
        ref_signal = float(np.percentile(signals, 75))
        csf_signal = float(np.median(sl[canal]))
        half = max(1, int(round(4.0 / dz)))
        for d in discs:
            z = d["z_index"]
            lo_z, hi_z = max(z0, z - half), min(z1, z + half)
            # Posterior displacement of the canal's anterior wall at the disc compared with
            # the smoothed wall position (driven by the adjacent vertebral bodies).
            disp = edge_s[lo_z:hi_z + 1] - ant_edge[lo_z:hi_z + 1]
            protrusion = float(np.nanmax(disp)) * dy if np.isfinite(disp).any() else 0.0
            if protrusion >= min_protrusion_mm:
                region = np.zeros_like(canal)
                for zz in range(lo_z, hi_z + 1):
                    if np.isfinite(ant_edge[zz]) and np.isfinite(edge_s[zz]) and edge_s[zz] > ant_edge[zz]:
                        region[int(ant_edge[zz]) + 1:int(edge_s[zz]) + 1, zz] = True
                if region.any():
                    add_finding("posterior disc protrusion",
                                f"Disc material extends {protrusion:.1f} mm posteriorly into the canal at {d['level']}.",
                                d["level"],
                                {"protrusion_depth_mm": round(protrusion, 1)},
                                "large (>=6 mm)" if protrusion >= 6 else "moderate (3-6 mm)",
                                min(1.0, protrusion / 8.0 + 0.3), region, z)
            rel = d["signal"] / ref_signal if ref_signal > 0 else 1.0
            d["signal_ratio_to_brightest_discs"] = round(rel, 2)
            d["signal_ratio_to_csf"] = round(d["signal"] / csf_signal, 2) if csf_signal > 0 else None
            if rel < 0.6 and d["signal"] < d["body_signal"] * 1.05:
                region = np.zeros_like(canal)
                for zz in range(max(0, z - half // 2), min(nz, z + half // 2 + 1)):
                    if np.isfinite(edge_s[zz]):
                        region[int(edge_s[zz]) + strip_lo:min(sl.shape[0], int(edge_s[zz]) + strip_hi), zz] = True
                add_finding("disc T2 signal loss",
                            f"Disc at {d['level']} is dark on T2 ({rel:.0%} of the brighter discs), "
                            "consistent with possible desiccation.",
                            d["level"],
                            {"signal_ratio_to_brightest_discs": round(rel, 2),
                             "signal_ratio_to_csf": d["signal_ratio_to_csf"]},
                            "marked" if rel < 0.4 else "moderate", min(1.0, 1.2 - rel), region, z)

    context = np.zeros(vol.shape, dtype=bool)
    context[x] = canal
    for d in discs:
        z = d["z_index"]
        if np.isfinite(edge_s[z]):
            y_a = int(edge_s[z]) + strip_lo
            label_map[x, y_a:y_a + int(round(20 / dy)), max(0, z - 1):z + 2][
                label_map[x, y_a:y_a + int(round(20 / dy)), max(0, z - 1):z + 2] == 0] = DISC_CONTEXT_LABEL
    label_map[x][(label_map[x] == 0) & canal] = CANAL_CONTEXT_LABEL

    profile = [{"z_index": int(z), "z_mm": round(z * dz, 1), "ap_mm": round(float(ap_s[z]), 2),
                "level": nearest_level(z)} for z in range(nz) if not np.isnan(ap_s[z])]
    for d in discs:
        d["signal"] = round(d["signal"], 3)
        d["body_signal"] = round(d["body_signal"], 3)

    return {
        **base,
        "summary": {
            "midsagittal_index": int(x),
            "reference_ap_diameter_mm": round(reference_ap, 1),
            "min_ap_diameter_mm": round(float(np.nanmin(ap_s)), 1),
            "discs_found": len(discs),
            "candidate_count": len(findings),
        },
        "findings": findings,
        "warnings": warnings,
        "profile": profile,
        "discs": discs,
        "context_mask": context,
    }
