"""Optional pretrained segmentation models, loaded from ONNX files.

No weights ship with the repo: the openly available brain tumour models
(e.g. BraTS-trained MONAI bundles or nnU-Net) need four co-registered
sequences and carry research-only licences. Export the model you are licensed to use to
ONNX and point an environment variable at it:

    MRI_BRAIN_ONNX=/models/brain_lesion.onnx

Expected contract: input ``(1, 1, X, Y, Z)`` float32 in RAS+ orientation,
z-score normalised inside the volume; output ``(1, C, X, Y, Z)`` logits or
probabilities where any channel > 0 after argmax is a lesion. If the model has
a fixed spatial input size the volume is resampled to it and back.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy import ndimage

ENV_VARS = {"brain": "MRI_BRAIN_ONNX"}


class OnnxSegmenter:
    def __init__(self, path: str):
        import onnxruntime as ort  # imported lazily: optional dependency

        self.path = path
        self.name = os.path.basename(path)
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        dims = inp.shape[2:]
        self.fixed_shape = tuple(int(d) for d in dims) if all(isinstance(d, int) for d in dims) else None

    def segment(self, data: np.ndarray) -> np.ndarray:
        x = data.astype(np.float32)
        fg = x > np.percentile(x, 5)
        mu, sd = float(x[fg].mean()), float(x[fg].std() or 1.0)
        x = (x - mu) / sd
        orig_shape = x.shape
        if self.fixed_shape and self.fixed_shape != orig_shape:
            x = ndimage.zoom(x, np.array(self.fixed_shape) / np.array(orig_shape), order=1)
        out = self.session.run(None, {self.input_name: x[None, None]})[0][0]
        labels = np.argmax(out, axis=0) if out.shape[0] > 1 else (out[0] > 0.5)
        mask = labels > 0
        if mask.shape != orig_shape:
            mask = ndimage.zoom(mask.astype(np.uint8), np.array(orig_shape) / np.array(mask.shape), order=0) > 0
            mask = _fit(mask, orig_shape)
        return mask


def _fit(mask: np.ndarray, shape) -> np.ndarray:
    out = np.zeros(shape, bool)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(mask.shape, shape))
    out[sl] = mask[sl]
    return out


@lru_cache(maxsize=4)
def _load(path: str) -> OnnxSegmenter:
    return OnnxSegmenter(path)


def get_model(task: str) -> OnnxSegmenter | None:
    path = os.environ.get(ENV_VARS[task], "")
    if not path or not os.path.exists(path):
        return None
    try:
        return _load(path)
    except ImportError:
        return None


def available() -> dict:
    return {task: (get_model(task).name if get_model(task) else None) for task in ENV_VARS}
