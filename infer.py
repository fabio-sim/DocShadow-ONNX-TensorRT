import argparse

import numpy as np
from PIL import Image

from onnx_runner import DocShadowRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        default="assets/sample.jpg",
        required=False,
        help="Path to input image for inference.",
    )
    parser.add_argument(
        "--img_size",
        nargs=2,
        type=int,
        default=[512, 512],
        required=False,
        help="Image size for inference. Please provide two integers (height width). Ensure that you have enough memory.",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        required=False,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to run inference using float16 (half) ONNX model (CUDA only).",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Whether to use TensorRT. Note that the end2end ONNX model must NOT be exported with --fp16. TensorRT will perform the conversion instead. Only static input shapes are supported.",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Whether to visualize the results."
    )
    return parser.parse_args()


def infer(
    img_path="assets/sample.jpg",
    img_size=[512, 512],
    onnx_path=None,
    fp16=False,
    trt=False,
    viz=False,
):
    img = Image.open(img_path).convert("RGB")
    orig_W, orig_H = img.size

    # Handle args.
    if onnx_path is None:
        onnx_path = "weights/docshadow_sd7k.onnx"  # default path

    # Preprocessing
    H, W = img_size
    image = DocShadowRunner.preprocess(np.array(img.resize((W, H))))
    if fp16 and not trt:
        image = image.astype(np.float16)

    # Inference
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if trt:
        providers.insert(
            0,
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "weights/cache",
                },
            ),
        )

    runner = DocShadowRunner(onnx_path, providers=providers)
    result = runner.run(image)

    # Visualisation
    if viz:
        import cv2

        result_img = result[0].transpose(1, 2, 0)
        result_img = cv2.resize(result_img, (orig_W, orig_H))
        cv2.imshow("result", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    return result


if __name__ == "__main__":
    args = parse_args()
    result = infer(**vars(args))
    print(result)
    print(result.shape)
