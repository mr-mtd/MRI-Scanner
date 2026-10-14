import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import phantoms
from app.brain import analyze_brain
from app.loaders import load_uploads
from app.main import app
from app.spine import analyze_spine


@pytest.fixture(scope="module")
def brain():
    return phantoms.brain_phantom()


@pytest.fixture(scope="module")
def spine():
    return phantoms.spine_phantom()


def test_brain_detects_planted_lesions_with_accurate_volume(brain):
    vol, truth = brain
    result = analyze_brain(vol)
    assert len(result["findings"]) == len(truth)
    for t in truth:
        f = min(result["findings"],
                key=lambda f: np.linalg.norm(np.subtract(f["measurements"]["centroid_voxel"], t["centroid_voxel"])))
        assert np.linalg.norm(np.subtract(f["measurements"]["centroid_voxel"], t["centroid_voxel"])) < 2
        assert f["measurements"]["volume_mm3"] == pytest.approx(t["volume_mm3"], rel=0.15)
    sides = sorted(f["location"]["hemisphere"] for f in result["findings"])
    assert sides == ["left", "right"]


def test_brain_label_map_matches_findings(brain):
    vol, _ = brain
    result = analyze_brain(vol)
    labels = result["label_map"]
    for f in result["findings"]:
        assert int((labels == f["label_value"]).sum()) == f["measurements"]["voxel_count"]


def test_brain_without_lesions_has_no_large_candidates():
    vol, _ = phantoms.brain_phantom()
    # Replace lesion voxels with white-matter level intensity.
    healthy = vol.data.copy()
    healthy[healthy > 850] = 500
    vol.data = healthy
    result = analyze_brain(vol)
    assert all(f["measurements"]["volume_mm3"] < 200 for f in result["findings"])


def test_spine_finds_stenosis_protrusion_and_desiccation(spine):
    vol, truth = spine
    result = analyze_spine(vol)
    assert [d["level"] for d in result["discs"]][:6] == ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1"]
    by_type = {}
    for f in result["findings"]:
        by_type.setdefault(f["type"], []).append(f)
    (narrow,) = by_type["central canal narrowing"]
    assert narrow["location"]["level"] == truth["stenosis_level"]
    assert narrow["measurements"]["min_ap_diameter_mm"] == pytest.approx(truth["stenosis_ap_mm"], abs=1.5)
    (prot,) = by_type["posterior disc protrusion"]
    assert prot["location"]["level"] == truth["stenosis_level"]
    assert prot["measurements"]["protrusion_depth_mm"] >= 4
    (dark,) = by_type["disc T2 signal loss"]
    assert dark["location"]["level"] == truth["desiccated_level"]


def test_dicom_round_trip_preserves_geometry(brain):
    vol, _ = brain
    files = phantoms.volume_to_dicom_series(vol)
    loaded = load_uploads(files)
    assert loaded.shape == vol.shape
    assert np.allclose(loaded.spacing, vol.spacing, atol=1e-3)
    assert np.allclose(loaded.affine, vol.affine, atol=1e-3)
    corr = np.corrcoef(loaded.data.ravel(), vol.data.ravel())[0, 1]
    assert corr > 0.999


def test_zip_and_nifti_uploads(brain):
    vol, _ = brain
    files = phantoms.volume_to_dicom_series(vol)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files:
            zf.writestr(f"series/{name}", data)
    assert load_uploads([("study.zip", buf.getvalue())]).shape == vol.shape
    nii = phantoms.volume_to_nifti_bytes(vol)
    assert load_uploads([("brain.nii", nii)]).shape == vol.shape


def test_api_end_to_end():
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    for kind, mode in [("brain-dicom", "brain"), ("spine", "spine")]:
        meta = client.post(f"/api/demo/{kind}").json()
        sid = meta["id"]
        vol = client.get(f"/api/studies/{sid}/volume")
        assert len(vol.content) == int(np.prod(meta["shape"]))
        res = client.post(f"/api/studies/{sid}/analyze", json={"mode": mode}).json()
        assert res["findings"], res
        assert "not diagnoses" in res["disclaimer"]
        labels = client.get(f"/api/studies/{sid}/labels")
        assert len(labels.content) == int(np.prod(meta["shape"]))


def test_api_rejects_garbage_upload():
    client = TestClient(app)
    r = client.post("/api/studies", files=[("files", ("x.txt", b"not an image"))])
    assert r.status_code == 400
