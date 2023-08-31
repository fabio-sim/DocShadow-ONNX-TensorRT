import argparse
import warnings

warnings.filterwarnings("ignore", module="onnxconverter_common.float16")

import onnx
import torch
from onnxconverter_common import float16

from DocShadow.models import DocShadow
from DocShadow.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        nargs=2,
        type=int,
        default=[256, 256],
        required=False,
        help="Sample image size for ONNX tracing. Please provide two integers (height width). Ensure that you have enough memory to run the export.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="sd7k",
        required=False,
        help="DocShadow has been trained on these datasets: ['sd7k', 'jung', 'kliger']. Defaults to 'sd7k' weights. You can also specify a local path to the weights.",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the exported ONNX model.",
    )
    parser.add_argument(
        "--dynamic_img_size",
        action="store_true",
        help="Whether to allow dynamic image sizes.",
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="Whether to allow dynamic batch size.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to also export float16 (half) ONNX model (CUDA only).",
    )

    return parser.parse_args()


def export_onnx(
    img_size=[256, 256],
    weights="sd7k",
    onnx_path=None,
    dynamic_img_size=False,
    dynamic_batch=False,
    fp16=False,
):
    # Handle args.
    H, W = img_size
    if onnx_path is None:
        onnx_path = (
            f"weights/docshadow_{weights}"
            f"{f'_{H}x{W}' if not dynamic_img_size else ''}"
            ".onnx"
        )

    # Load inputs and models.
    device = torch.device("cpu")  # Device on which to export.

    img = torch.rand(1, 3, H, W, dtype=torch.float32, device=device)

    docshadow = DocShadow()
    load_checkpoint(docshadow, weights, device)
    docshadow.eval().to(device)

    # Export.
    opset_version = 12
    dynamic_axes = {"image": {}, "result": {}}
    if dynamic_batch:
        dynamic_axes["image"].update({0: "batch_size"})
        dynamic_axes["result"].update({0: "batch_size"})
    if dynamic_img_size:
        dynamic_axes["image"].update({2: "height", 3: "width"})
        dynamic_axes["result"].update({2: "height", 3: "width"})

    torch.onnx.export(
        docshadow,
        img,
        onnx_path,
        input_names=["image"],
        output_names=["result"],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
    )
    if fp16:
        convert_fp16(onnx_path)


def convert_fp16(onnx_model_path: str):
    onnx_model = onnx.load(onnx_model_path)
    fp16_model = float16.convert_float_to_float16(onnx_model)
    onnx.save(fp16_model, onnx_model_path.replace(".onnx", "_fp16.onnx"))


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
