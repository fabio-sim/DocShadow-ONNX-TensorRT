"""Export DocShadow to CoreML."""
import coremltools as ct  # 6.3.0
import torch

from DocShadow.models import DocShadow
from DocShadow.models.backbone import LayerNorm2d
from DocShadow.models.model import WithBias_LayerNorm
from DocShadow.utils import load_checkpoint

H, W = 256, 256
weights = "sd7k"  # "jung", "kligler"


# Patches for CoreML compatibility


def WithBias_LayerNorm_forward(self, x):
    """Manually compute variance instead of using Tensor.var()"""
    mu = x.mean(-1, keepdim=True)
    sigma = (x - mu).pow(2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


WithBias_LayerNorm.forward = WithBias_LayerNorm_forward


def LayerNorm2d_forward(self, x):
    """Layer Normalization over the channels dimension only."""
    N, C, H, W = x.size()
    mu = x.mean(1, keepdim=True)
    var = (x - mu).pow(2).mean(1, keepdim=True)
    y = (x - mu) / (var + self.eps).sqrt()
    y = self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)
    return y


LayerNorm2d.forward = LayerNorm2d_forward

# Load inputs and models.
device = torch.device("cpu")  # Device on which to export.

img = torch.rand(1, 3, H, W, dtype=torch.float32, device=device)

docshadow = DocShadow()
load_checkpoint(docshadow, weights, device)
docshadow.eval().to(device)

docshadow.trans_high.spp_img.interpolation_type = "bilinear"  # bicubic unsupported

traced_docshadow = torch.jit.trace(docshadow, img)

coreml_docshadow = ct.convert(
    traced_docshadow,
    # convert_to="mlprogram",
    inputs=[ct.TensorType(shape=img.shape)],
)

coreml_docshadow.save(f"weights/docshadow_{weights}.mlmodel")
# coreml_docshadow.save(f"weights/docshadow_{weights}.mlpackage")
