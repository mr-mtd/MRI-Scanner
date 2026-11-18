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
1. Pick the midsagittal slice: a strong continuous canal, and the widest sac.
2. Segment the thecal sac (brightest class, grown to half-maximum) and measure
   it wall to wall, perpendicular to the canal axis.
3. Locate discs from the intensity just anterior to the canal, repeating at the
   vertebral spacing, and name levels assuming the most caudal disc is L5-S1.
4. Find the conus (the cord fills the sac above it) and assess narrowing below it.
5. Report focal canal narrowing, posterior disc protrusion and disc signal loss.

Checked on healthy-volunteer lumbar T2 scans (see docs/real-data/VALIDATION.md):
the absolute AP cut-offs above would over-call stenosis there, so a candidate
needs focal narrowing against the patient's own canal.
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
FOCAL_RATIO = 0.75
MIN_FOCAL_LENGTH_MM = 5.0
MAX_FOCAL_LENGTH_MM = 40.0


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
    return best[1], best[0]


def _grow_canal(sl: np.ndarray, seed: np.ndarray, dy: float, dz: float) -> np.ndarray:
    """Grow the bright seed to the half-maximum between CSF and the tissue around it, so
    partial-volume CSF at the walls and between nerve roots is included."""
    sm = ndimage.gaussian_filter(sl, 0.8)
    ring_px = [max(1, int(round(6.0 / s))) for s in (dy, dz)]
    near = ndimage.binary_dilation(seed, structure=np.ones((3, 3)), iterations=max(ring_px))
    ring = near & ~seed
    csf = float(np.percentile(sm[seed], 75))
    tissue = float(np.median(sm[ring])) if ring.any() else 0.0
    grown_labels, _ = ndimage.label(near & (sm > 0.5 * (csf + tissue)))
    keep = np.unique(grown_labels[seed])
    comp = np.isin(grown_labels, keep[keep > 0]) | seed
    comp = ndimage.binary_closing(comp, structure=np.ones((3, 3)))
    return ndimage.binary_fill_holes(comp)


def _choose_midsagittal(vol: Volume):
    """Pick the midsagittal slice: among slices with a strong, continuous canal, the one where the
    thecal sac is widest (the sac is widest at the midline and narrows off-axis)."""
    nx = vol.shape[0]
    dy, dz = vol.spacing[1], vol.spacing[2]
    lo, hi = int(nx * 0.2), int(np.ceil(nx * 0.8))
    seeds = {}
    for x in range(lo, max(hi, lo + 1)):
        found = _segment_canal(vol.data[x], dy, dz)
        if found is not None:
            seeds[x] = found
    if not seeds:
        return None, None
    top = max(score for _, score in seeds.values())
    best = None
    for x, (seed, score) in seeds.items():
        if score < 0.75 * top:
            continue
        canal = _grow_canal(vol.data[x], seed, dy, dz)
        rows = np.where(canal.any(axis=0))[0]
        ap, *_ = _measure_canal(ndimage.gaussian_filter(vol.data[x], 0.8), canal, int(rows.min()),
                                int(rows.max()), dy, dz, max_gap_mm=1.5)
        width = float(np.nanpercentile(ap, 60)) if np.isfinite(ap).any() else 0.0
        if best is None or width > best[0]:
            best = (width, x, canal)
    return best[1], best[2]


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


def _measure_canal(sl: np.ndarray, canal: np.ndarray, z0: int, z1: int, dy: float, dz: float,
                   max_gap_mm: float = 10.0):
    """Thecal sac AP diameter per craniocaudal row.

    Measures wall to wall (so dark nerve roots or the conus inside the sac do not
    count as narrowing), refines each wall to the half-maximum between CSF and
    surrounding tissue, and corrects for the canal's tilt (lordosis) so the
    diameter is taken perpendicular to the canal axis.
    """
    nz = sl.shape[1]
    ap = np.full(nz, np.nan)
    ant = np.full(nz, np.nan)
    post = np.full(nz, np.nan)
    cord = np.zeros(nz, dtype=bool)
    center = np.full(nz, np.nan)
    for z in range(z0, z1 + 1):
        ys = np.where(canal[:, z])[0]
        if ys.size:
            center[z] = 0.5 * (ys.min() + ys.max())
    ok = ~np.isnan(center)
    if ok.sum() < 3:
        return ap, ant, post, cord
    zs = np.arange(nz)
    c_s = np.interp(zs, zs[ok], center[ok])
    c_s = ndimage.gaussian_filter1d(ndimage.median_filter(c_s, size=int(round(15 / dz)) | 1), 10 / dz)
    slope = np.gradient(c_s) * dy / dz  # mm of AP shift per mm craniocaudal
    cos_t = 1.0 / np.sqrt(1.0 + slope ** 2)

    csf = float(np.percentile(sl[canal], 75))
    band = int(round(20.0 / dy))
    max_gap = int(round(max_gap_mm / dy))
    win = int(round(20.0 / dz))
    for z in range(z0, z1 + 1):
        ys = np.where(canal[:, z])[0]
        if not ys.size:
            continue
        # Keep only canal pixels near the smoothed centreline (drops side branches).
        ys = ys[np.abs(ys - c_s[z]) <= band]
        if not ys.size:
            continue
        lo, hi = int(ys.min()), int(ys.max())
        # CSF brightness drifts along the coil, so use the canal's brightness in nearby rows.
        za, zb = max(z0, z - win), min(z1, z + win) + 1
        local = sl[:, za:zb][canal[:, za:zb]]
        local_csf = float(np.percentile(local, 75)) if local.size else csf
        thr = 0.45 * local_csf
        dark = 0.12 * local_csf

        # Extend each wall across intermediate-signal gaps (cord, conus, nerve roots) to CSF
        # beyond them, but never across dark tissue (ligament, annulus, dura against fat).
        def extend(start, step, limit):
            edge, gap, y = start, 0, start + step
            while 0 <= y < sl.shape[0] and abs(y - start) <= limit:
                v = sl[y, z]
                if v > thr:
                    edge, gap = y, 0
                elif v < dark:
                    break
                else:
                    gap += 1
                    if gap > max_gap:
                        break
                y += step
            return edge

        hi = extend(hi, 1, band)
        lo = extend(lo, -1, band)
        ap[z] = (hi - lo + 1) * dy * cos_t[z]
        # A long run of intermediate signal inside the sac is the spinal cord / conus.
        inner = sl[lo:hi + 1, z] < thr
        run = best_run = 0
        for v in inner:
            run = run + 1 if v else 0
            best_run = max(best_run, run)
        cord[z] = best_run * dy >= 4.0
        ant[z] = hi  # higher y = anterior in RAS+
        post[z] = lo
    return ap, ant, post, cord


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

    # Bridging intermediate signal finds the sac around the cord (used to locate the conus);
    # below the conus only thin nerve roots are bridged, so bright epidural fat behind the
    # dura is not mistaken for CSF.
    _, _, _, cord_rows = _measure_canal(sl, canal, z0, z1, dy, dz, max_gap_mm=10.0)
    ap, ant_edge, post_edge, _ = _measure_canal(sl, canal, z0, z1, dy, dz, max_gap_mm=1.5)
    valid = ~np.isnan(ap)
    ap_s = ap.copy()
    # Wall detection fails by under-measuring (a missed CSF strip), never by over-measuring, so a
    # running upper percentile is robust to those misses while keeping narrowings longer than ~3 mm.
    k = max(1, int(round(3.0 / dz)))
    ap_s[valid] = ndimage.percentile_filter(ap[valid], 70, size=2 * k + 1, mode="nearest")
    reference_ap = float(np.nanmedian(ap_s))

    # --- discs: intensity in the anterior column just in front of the canal, along z -------
    # Healthy discs are bright on T2 and degenerated ones dark, while vertebral bodies are
    # intermediate, so discs show up as deviations either way from a slowly varying baseline,
    # repeating at the vertebral spacing.
    strip_lo = int(round(8.0 / dy))
    strip_hi = int(round(22.0 / dy))
    zz = np.arange(nz)
    edge_ok = np.isfinite(ant_edge)
    if not edge_ok.any():  # fall back to the segmented canal's anterior-most pixel per row
        edge_ok = canal.any(axis=0)
        ant_edge = np.where(edge_ok, canal.shape[0] - 1 - np.argmax(canal[::-1], axis=0), np.nan)
    edge_s = np.interp(zz, zz[edge_ok], ant_edge[edge_ok])
    edge_s = ndimage.median_filter(edge_s, size=int(round(30.0 / dz)) | 1, mode="nearest")
    edge_s[(zz < z0) | (zz > z1)] = np.nan
    strip = np.full(nz, np.nan)
    for z in range(z0, z1 + 1):
        y_a = int(edge_s[z]) + strip_lo
        y_b = min(sl.shape[0], int(edge_s[z]) + strip_hi)
        if y_b > y_a:
            strip[z] = float(np.median(sl[y_a:y_b, z]))
    sv = ~np.isnan(strip)
    discs = []
    if sv.sum() > int(60 / dz):
        prof = strip.copy()
        prof[~sv] = np.nanmedian(strip)
        prof = ndimage.gaussian_filter1d(prof, 1.0 / dz)
        baseline = ndimage.median_filter(prof, size=int(round(40.0 / dz)) | 1, mode="nearest")
        dev = np.abs(prof - baseline)
        dev[~sv] = 0
        # Vertebral spacing from the autocorrelation of the deviation signal (25-50 mm in adults).
        d0 = dev[sv] - dev[sv].mean()
        ac = np.correlate(d0, d0, mode="full")[len(d0) - 1:]
        lags = np.arange(int(25 / dz), min(len(ac), int(50 / dz)))
        period = float(lags[np.argmax(ac[lags])]) if lags.size else 33.0 / dz
        noise = float(np.median(dev[sv])) * 1.4826 + 1e-6
        peaks, _ = find_peaks(dev, height=1.5 * noise, distance=max(1, int(0.65 * period)))
        edge_guard = int(round(8.0 / dz))
        peaks = sorted(int(p) for p in peaks if sv[p] and z0 + edge_guard <= p <= z1 - edge_guard)
        half = max(1, int(round(3.0 / dz)))
        for idx, p in enumerate(peaks):
            name = LEVEL_NAMES[idx] if idx < len(LEVEL_NAMES) else f"disc {idx + 1} above L5-S1"
            discs.append({"level": name, "z_index": p, "z_mm": round(p * dz, 1),
                          "signal": float(np.nanmean(strip[max(0, p - half):p + half + 1])),
                          "body_signal": float(baseline[p])})
    if not discs:
        warnings.append("No intervertebral discs were located; level names are unavailable.")

    # --- where central stenosis is assessed -------------------------------------------
    # Above the conus the cord fills much of the sac and CSF strips are thin, so the CSF-based
    # diameter is unreliable there; assess the cauda equina segment below the conus tip.
    # The cord, once present, continues up to the top of the canal, so the conus tip is the start
    # of the cord run that reaches the canal's upper end (short runs lower down are clumped roots).
    cord_frac = ndimage.uniform_filter1d(cord_rows.astype(float), size=int(round(20.0 / dz)))
    conus_z = None
    for a_, b_ in _runs(cord_frac > 0.2):
        if (b_ - a_) * dz >= 20.0 and (z1 - b_) * dz <= 15.0:
            conus_z = a_
    assess = valid.copy()
    if conus_z is not None:
        assess[max(0, conus_z - int(round(10.0 / dz))):] = False
        warnings.append(f"Spinal cord / conus detected above {conus_z * dz:.0f} mm; canal narrowing "
                        "is assessed below it only.")
    if discs:
        assess[:max(0, discs[0]["z_index"] - int(round(5.0 / dz)))] = False
    if assess.sum() > int(20 / dz):
        reference_ap = float(np.nanmedian(ap_s[assess]))

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
    # A candidate is a focal narrowing against this patient's own canal: below FOCAL_RATIO of the
    # reference diameter over a short segment. A uniformly small canal is reported in the
    # per-level table rather than flagged, since normal dural sac size varies widely.
    flags = assess & (ap_s < FOCAL_RATIO * reference_ap) & (ap_s < RELATIVE_STENOSIS_MM)
    flags[:z0 + int(5 / dz)] = False  # ignore the tapering ends of the segmented sac
    flags[z1 - int(5 / dz):] = False
    assess_top = int(np.where(assess)[0].max()) if assess.any() else z1
    for a, b in _runs(flags):
        if not MIN_FOCAL_LENGTH_MM <= (b - a + 1) * dz <= MAX_FOCAL_LENGTH_MM:
            continue
        if b >= assess_top - int(round(15.0 / dz)):
            continue  # runs on to the top of the assessed sac (conus entering): not a focal narrowing
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
        # Canal AP at each disc level, reported whether or not it is flagged.
        z = d["z_index"]
        near = ap_s[max(0, z - int(round(2 / dz))):z + int(round(2 / dz)) + 1]
        d["canal_ap_mm"] = round(float(np.nanmedian(near)), 1) if assess[z] and np.isfinite(near).any() else None

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
