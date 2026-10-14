"""Optional adapter for a pretrained single-sequence 3D segmentation model in ONNX format.

Point the env var MRI_BRAIN_MODEL at a JSON config, for example:

    {
      "name": "my-flair-lesion-unet",
      "model_path": "flair_unet.onnx",
      "patch_size": [96, 96, 96],
      "normalization": "zscore_nonzero",
      "output": "softmax",
      "labels": {"1": "lesion"}
    }

The model must take a float32 tensor shaped [1, 1, X, Y, Z] in canonical RAS+
voxel order at the series' native resolution and return [1, C, X, Y, Z] scores
(C classes incl. background for softmax, or C foreground channels for sigmoid).
Models trained on other inputs (several co-registered sequences, fixed 1 mm
resampling) need a matching export or preprocessing before they can be used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .volume import Volume


class OnnxSegmentationModel:
    def __init__(self, config_path: str):
        import onnxruntime as ort  # optional dependency

        cfg_file = Path(config_path)
        self.config = json.loads(cfg_file.read_text())
        model_path = Path(self.config["model_path"])
        if not model_path.is_absolute():
            model_path = cfg_file.parent / model_path
        self.name = self.config.get("name", model_path.stem)
        self.patch = [int(v) for v in self.config.get("patch_size", [96, 96, 96])]
        self.labels = {int(k): v for k, v in self.config.get("labels", {"1": "lesion"}).items()}
        self.output = self.config.get("output", "softmax")
        self.threshold = float(self.config.get("threshold", 0.5))
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def describe(self) -> dict:
        return {"patch_size": self.patch, "labels": self.labels, "output": self.output,
                "normalization": self.config.get("normalization", "zscore_nonzero")}

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self.config.get("normalization", "zscore_nonzero") == "none":
            return data.astype(np.float32)
        nz = data[data > 0]
        mu, sd = (float(nz.mean()), float(nz.std()) or 1.0) if nz.size else (0.0, 1.0)
        return ((data - mu) / sd).astype(np.float32)

    def segment(self, vol: Volume) -> tuple[np.ndarray, dict]:
        """Tile the volume into patches, run the model, and return (label map, label names)."""
        data = self._normalize(vol.data)
        shape = np.array(data.shape)
        patch = np.array(self.patch)
        padded_shape = np.ceil(shape / patch).astype(int) * patch
        padded = np.zeros(padded_shape, dtype=np.float32)
        padded[: shape[0], : shape[1], : shape[2]] = data
        labels = np.zeros(padded_shape, dtype=np.uint8)
        for i in range(0, padded_shape[0], patch[0]):
            for j in range(0, padded_shape[1], patch[1]):
                for k in range(0, padded_shape[2], patch[2]):
                    block = padded[i:i + patch[0], j:j + patch[1], k:k + patch[2]][None, None]
                    scores = self.session.run(None, {self.input_name: block})[0][0]
                    if self.output == "sigmoid":
                        prob = 1 / (1 + np.exp(-scores))
                        lab = np.where(prob.max(0) > self.threshold, prob.argmax(0) + 1, 0)
                    else:
                        lab = scores.argmax(0)
                    labels[i:i + patch[0], j:j + patch[1], k:k + patch[2]] = lab
        return labels[: shape[0], : shape[1], : shape[2]], self.labels


def load_configured_model() -> tuple[OnnxSegmentationModel | None, str | None]:
    """Load the model named by MRI_BRAIN_MODEL, returning (model, error)."""
    path = os.environ.get("MRI_BRAIN_MODEL")
    if not path:
        return None, None
    try:
        return OnnxSegmentationModel(path), None
    except Exception as exc:  # report, fall back to the baseline detector
        return None, f"Could not load pretrained model from {path}: {exc}"
