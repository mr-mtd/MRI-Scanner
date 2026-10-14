import io
import zipfile

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import demo, loaders
from app.brain import BrainParams, analyze_brain
from app.main import app
from app.spine import SpineParams, analyze_spine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_brain_phantom_lesions_detected():
    nii, truth = demo.brain_phantom()
    vol = loaders.load_files([("brain.nii", nii.to_bytes())])
    mask, result = analyze_brain(vol, BrainParams())
    findings = result["findings"]
    assert len(findings) == 2
    big, small = findings
    # Volumes within 20% of ground truth.
    assert big["volumeMl"] == pytest.approx(truth["volumes_ml"][0], rel=0.2)
    assert small["volumeMl"] == pytest.approx(truth["volumes_ml"][1], rel=0.4)
    assert big["location"]["side"] == "right" and big["location"]["region"] == "frontal"
    assert small["location"]["side"] == "left"
    pred = mask > 0
    gt = truth["lesions"][0] | truth["lesions"][1]
    dice = 2 * (pred & gt).sum() / (pred.sum() + gt.sum())
    assert dice > 0.8


def test_brain_ventricles_not_flagged_in_hypo_mode_with_asymmetry():
    nii, _ = demo.brain_phantom()
    vol = loaders.load_files([("brain.nii", nii.to_bytes())])
    _, result = analyze_brain(vol, BrainParams(polarity="hypo"))
    # The dark, symmetric ventricles must not become candidates.
    assert all(f["asymmetryZ"] >= 1.0 for f in result["findings"])


def test_spine_dicom_detects_stenosis_and_desiccation():
    vol = loaders.load_files([("spine.zip", demo.spine_dicom_zip())])
    assert vol.source_format == "dicom"
    assert vol.spacing == pytest.approx((4.0, 0.75, 0.75))
    _, result = analyze_spine(vol, SpineParams())
    by_level = {f["level"]: f for f in result["findings"]}
    assert list(by_level) == demo.SPINE_LEVELS
    l45 = by_level["L4-L5"]
    assert l45["canalApMm"] < 10
    assert {f["code"] for f in l45["flags"]} >= {"stenosis_severe", "disc_displacement"}
    l5s1 = {f["code"] for f in by_level["L5-S1"]["flags"]}
    assert "low_signal" in l5s1
    for name in ["L1-L2", "L2-L3", "L3-L4"]:
        assert by_level[name]["flags"] == []
        assert by_level[name]["canalApMm"] == pytest.approx(14, abs=1.5)
    assert result["summary"]["flaggedLevels"] == ["L4-L5", "L5-S1"]


def test_dicom_and_nifti_orientations_agree():
    data, spacing, _ = demo.spine_phantom()
    from_nii = loaders.load_files([("s.nii", nib.Nifti1Image(data, demo.spine_affine(data.shape, spacing)).to_bytes())])
    from_dcm = loaders.load_files([("s.zip", demo.spine_dicom_zip())])
    assert from_nii.shape == from_dcm.shape
    np.testing.assert_allclose(from_nii.affine, from_dcm.affine, atol=1e-3)
    corr = np.corrcoef(from_nii.data.ravel(), from_dcm.data.ravel())[0, 1]
    assert corr > 0.999


def test_loose_dicom_files_upload(client):
    with zipfile.ZipFile(io.BytesIO(demo.spine_dicom_zip())) as zf:
        files = [("files", (n, zf.read(n), "application/dicom")) for n in zf.namelist()]
    r = client.post("/api/studies", files=files)
    assert r.status_code == 200, r.text
    assert r.json()["info"]["shape"] == [13, 240, 400]


def test_api_end_to_end(client):
    r = client.post("/api/demo/brain")
    assert r.status_code == 200
    sid, info = r.json()["id"], r.json()["info"]
    vol = client.get(f"/api/studies/{sid}/volume")
    assert len(vol.content) == int(np.prod(info["shape"]))
    assert client.get(f"/api/studies/{sid}/mask").status_code == 404
    res = client.post(f"/api/studies/{sid}/analyze", json={"mode": "brain"})
    assert res.status_code == 200, res.text
    assert res.json()["summary"]["candidateCount"] == 2
    mask = client.get(f"/api/studies/{sid}/mask")
    assert len(mask.content) == len(vol.content)
    nii = client.get(f"/api/studies/{sid}/mask.nii.gz")
    assert nii.status_code == 200 and nii.content[:2] == b"\x1f\x8b"


def test_upload_rejects_garbage(client):
    r = client.post("/api/studies", files=[("files", ("x.txt", b"hello", "text/plain"))])
    assert r.status_code == 400


def test_onnx_model_hook(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    # Tiny "model": 2-channel output = [0, x - 2] so argmax picks voxels with z-score > 2.
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, None, None, None])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, None, None, None])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    nodes = [
        helper.make_node("Sub", ["x", "two"], ["fg"]),
        helper.make_node("Mul", ["x", "zero"], ["bg"]),
        helper.make_node("Concat", ["bg", "fg"], ["y"], axis=1),
    ]
    graph = helper.make_graph(nodes, "t", [inp], [out], initializer=[two, zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "m.onnx"
    onnx.save(model, path)
    monkeypatch.setenv("MRI_BRAIN_ONNX", str(path))

    nii, truth = demo.brain_phantom()
    vol = loaders.load_files([("brain.nii", nii.to_bytes())])
    mask, result = analyze_brain(vol, BrainParams(engine="onnx", min_volume_ml=0.3))
    assert result["summary"]["engine"] == "onnx:m.onnx"
    assert (mask > 0)[truth["lesions"][0]].mean() > 0.8
