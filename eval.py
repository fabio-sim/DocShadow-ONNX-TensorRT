import argparse
import time

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "framework",
        type=str,
        choices=["torch", "ort"],
        help="The framework to measure inference time. Options are 'torch' for PyTorch and 'ort' for ONNXRuntime.",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="assets/sample.jpg",
        required=False,
        help="Path to the root of the MegaDepth dataset.",
    )
    parser.add_argument(
        "--img_size",
        nargs=2,
        type=int,
        default=[512, 512],
        required=False,
        help="Image size for inference. Please provide two integers (height width). Ensure that you have enough memory.",
    )

    # ONNXRuntime-specific args
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        required=False,
        help="Path to ONNX model (end2end).",
    )
    # parser.add_argument(
    #     "--fp16",
    #     action="store_true",
    #     help="Whether to enable half-precision for ONNXRuntime.",
    # )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Whether to use TensorRT Execution Provider.",
    )
    return parser.parse_args()


def create_models(framework: str, fp16=False, onnx_path=None, trt=False):
    if framework == "torch":
        device = torch.device("cuda")

        model = DocShadow()
        load_checkpoint(model, "sd7k", device)
        model.eval().to(device)
    elif framework == "ort":
        if onnx_path is None:
            onnx_path = (
                f"weights/docshadow_sd7k"
                f"{'_fp16' if fp16 and not trt else ''}"
                ".onnx"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        if trt:
            providers.insert(
                0,
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": fp16,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "weights/cache",
                        "trt_builder_optimization_level": 5,
                    },
                ),
            )
        model = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers
        )

    return model


def get_inputs(framework: str, img_path, img_size, fp16, trt):
    img = Image.open(img_path).convert("RGB")
    H, W = img_size
    img = img.resize((W, H))

    if framework == "torch":
        image = to_tensor(img)[None].cuda()
    elif framework == "ort":
        image = DocShadowRunner.preprocess(np.array(img))
        if fp16 and not trt:
            image = image.astype(np.float16)

    return image


def measure_inference(framework: str, model, images, fp16) -> float:
    if framework == "torch":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.inference_mode():
            result = model(images)
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end)
    elif framework == "ort":
        model_inputs = {"image": images}
        model_outputs = ["result"]

        # Prepare IO-Bindings
        binding = model.io_binding()

        for name, arr in model_inputs.items():
            binding.bind_cpu_input(name, arr)

        for name in model_outputs:
            binding.bind_output(name, "cuda")

        # Measure only matching time
        start = time.perf_counter()
        result = model.run_with_iobinding(binding)
        end = time.perf_counter()

        return (end - start) * 1000


def evaluate(
    framework: str,
    img_path="assets/sample.jpg",
    img_size=[512, 512],
    fp16=False,
    onnx_path=None,
    trt=False,
):
    model = create_models(
        framework,
        fp16=fp16,
        onnx_path=onnx_path,
        trt=trt,
    )

    # Warmup
    for _ in tqdm(range(5)):
        images = get_inputs(framework, img_path, img_size=img_size, fp16=fp16, trt=trt)
        _ = measure_inference(framework, model, images, fp16=fp16)

    # Measure
    timings = []
    for _ in tqdm(range(1000)):
        images = get_inputs(framework, img_path, img_size=img_size, fp16=fp16, trt=trt)

        inference_time = measure_inference(framework, model, images, fp16=fp16)
        timings.append(inference_time)

    # Results
    timings = np.array(timings)
    print(timings)
    print(f"Mean inference time: {timings.mean():.2f} +/- {timings.std():.2f} ms")
    print(f"Median inference time: {np.median(timings):.2f} ms")


if __name__ == "__main__":
    args = parse_args()
    if args.framework == "torch":
        import torch
        from torchvision.transforms.functional import to_tensor

        from DocShadow.models import DocShadow
        from DocShadow.utils import load_checkpoint
    elif args.framework == "ort":
        import onnxruntime as ort

        from onnx_runner import DocShadowRunner

    evaluate(**vars(args))
