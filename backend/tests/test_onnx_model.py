import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402

from app import phantoms  # noqa: E402
from app.brain import analyze_brain  # noqa: E402
from app.onnx_model import OnnxSegmentationModel  # noqa: E402


def _threshold_model(path, threshold=2.5):
    """Tiny stand-in for a pretrained net: class 1 where the z-scored input exceeds a threshold."""
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, None, None, None])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, None, None, None])
    t = helper.make_tensor("t", TensorProto.FLOAT, [], [threshold])
    nodes = [
        helper.make_node("Sub", ["image", "t"], ["fg"]),
        helper.make_node("Neg", ["fg"], ["bg"]),
        helper.make_node("Concat", ["bg", "fg"], ["logits"], axis=1),
    ]
    graph = helper.make_graph(nodes, "thresh", [x], [y], initializer=[t])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # readable by older onnxruntime releases
    onnx.save(model, path)


def test_onnx_adapter_feeds_shared_measurements(tmp_path):
    _threshold_model(str(tmp_path / "m.onnx"))
    cfg = tmp_path / "model.json"
    cfg.write_text(json.dumps({"name": "test-threshold", "model_path": "m.onnx", "patch_size": [64, 64, 64],
                               "labels": {"1": "lesion"}}))
    model = OnnxSegmentationModel(str(cfg))
    vol, truth = phantoms.brain_phantom()
    result = analyze_brain(vol, model=model)
    assert result["method"]["name"] == "test-threshold"
    big = [f for f in result["findings"] if f["measurements"]["volume_mm3"] > 500]
    assert big, "the stand-in model should segment at least the large planted lesion"
    largest = max(big, key=lambda f: f["measurements"]["volume_mm3"])
    # The stand-in is a crude threshold, so only check wiring: the region lands on the planted lesion
    # and the reported voxel count matches the label map handed to the viewer.
    assert np.linalg.norm(np.subtract(largest["measurements"]["centroid_voxel"], truth[0]["centroid_voxel"])) < 3
    labels = result["label_map"]
    assert int((labels == largest["label_value"]).sum()) == largest["measurements"]["voxel_count"]
