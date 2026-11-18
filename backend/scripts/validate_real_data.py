"""Run the detectors on public real MRI data and compare with ground truth where it exists.

Data is downloaded on demand into a cache folder and never committed:

* Brain: Medical Segmentation Decathlon Task01 (BraTS 2016/2017 gliomas), CC-BY-SA 4.0.
  Single cases are pulled out of the public tar on S3 with HTTP range requests.
  Ground truth: expert tumour masks (oedema + non-enhancing + enhancing = whole tumour).
* Spine: OpenNeuro ds008803 (lumbosacral cord fMRI study, sagittal 3D T2 SPACE of
  healthy volunteers), CC0. No pathology labels exist, so this checks false positives,
  canal diameter plausibility and disc level placement.

Usage:
    python scripts/validate_real_data.py brain BRATS_206 BRATS_218 ...
    python scripts/validate_real_data.py spine 01 02 04 ...
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.brain import analyze_brain  # noqa: E402
from app.spine import analyze_spine  # noqa: E402
from app.volume import from_nifti_image  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / ".data-cache"
MSD_TAR = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
OPENNEURO = "https://s3.amazonaws.com/openneuro.org/ds008803"


def _range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def msd_index() -> dict[str, tuple[int, int]]:
    """Walk the tar headers with range requests (no full 7.6 GB download)."""
    path = CACHE / "msd_task01_index.json"
    if path.exists():
        return json.loads(path.read_text())
    index, off = {}, 0
    while True:
        h = _range(MSD_TAR, off, 512)
        name = h[:100].rstrip(b"\0").decode()
        if not name:
            break
        prefix = h[345:500].rstrip(b"\0").decode()
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        index[f"{prefix}/{name}" if prefix else name] = (off + 512, size)
        off += 512 + ((size + 511) // 512) * 512
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(index))
    return index


def fetch_brats(case: str) -> tuple[Path, Path]:
    out = CACHE / "msd"
    out.mkdir(parents=True, exist_ok=True)
    img, seg = out / f"{case}_img.nii.gz", out / f"{case}_seg.nii.gz"
    if not (img.exists() and seg.exists()):
        index = msd_index()
        for kind, dest in (("imagesTr", img), ("labelsTr", seg)):
            off, size = index[f"Task01_BrainTumour/{kind}/{case}.nii.gz"]
            dest.write_bytes(_range(MSD_TAR, off, size))
    return img, seg


def fetch_lumbar(sub: str) -> Path:
    out = CACHE / "ds008803"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sub-{sub}_ses-hip_T2w.nii.gz"
    if not path.exists():
        url = f"{OPENNEURO}/sub-{sub}/ses-hip/anat/sub-{sub}_ses-hip_T2w.nii.gz"
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    return path


def run_brain(case: str) -> dict:
    img_path, seg_path = fetch_brats(case)
    img = nib.load(img_path)
    flair = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0].astype(np.float32), img.affine)  # channel 0 = FLAIR
    vol = from_nifti_image(flair, "nifti")
    gt = np.asarray(nib.as_closest_canonical(nib.load(seg_path)).dataobj) > 0
    res = analyze_brain(vol, already_skull_stripped=True)
    labels = res["label_map"]
    hits = [f for f in res["findings"] if (gt & (labels == f["label_value"])).any()]
    misses = [f for f in res["findings"] if f not in hits]
    pred = np.isin(labels, [f["label_value"] for f in hits])
    top = res["findings"][0] if res["findings"] else None
    gt_x = np.argwhere(gt).mean(0)[0]
    mid_x = np.argwhere(vol.data > 0).mean(0)[0]
    return {
        "case": case,
        "tumour_detected": bool(hits),
        "largest_candidate_is_tumour": bool(top and top in hits),
        "dice_whole_tumour": round(float(2 * (pred & gt).sum() / (pred.sum() + gt.sum())), 3),
        "gt_volume_ml": round(float(gt.sum() * vol.voxel_volume_mm3 / 1000), 1),
        "predicted_volume_ml": round(float(pred.sum() * vol.voxel_volume_mm3 / 1000), 1),
        "gt_side": "right" if gt_x > mid_x else "left",
        "predicted_side": top["location"]["hemisphere"] if top else None,
        "other_candidates": len(misses),
        "other_candidates_ml": round(sum(f["measurements"]["volume_ml"] for f in misses), 2),
    }


def run_spine(sub: str) -> dict:
    vol = from_nifti_image(nib.load(fetch_lumbar(sub)), "nifti")
    res = analyze_spine(vol)
    return {
        "subject": sub,
        "reference_ap_mm": res["summary"].get("reference_ap_diameter_mm"),
        "discs": [(d["level"], d["canal_ap_mm"]) for d in res["discs"]],
        "findings": [(f["type"], f["location"]["level"], f.get("severity")) for f in res["findings"]],
    }


if __name__ == "__main__":
    mode, ids = sys.argv[1], sys.argv[2:]
    fn = run_brain if mode == "brain" else run_spine
    for i in ids:
        print(json.dumps(fn(i)), flush=True)
