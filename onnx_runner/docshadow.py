# No dependency on PyTorch

import numpy as np
import onnxruntime as ort
from PIL import Image


class DocShadowRunner:
    def __init__(
        self,
        onnx_path=None,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.model = ort.InferenceSession(onnx_path, providers=providers)

    def run(self, images: np.ndarray) -> np.ndarray:
        result = self.model.run(None, {"image": images})[0]
        return result

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        # image.shape == (H, W, C)
        image = np.asarray(image) / 255
        image = image[None].transpose(0, 3, 1, 2)
        image = image.astype(np.float32)
        return image
