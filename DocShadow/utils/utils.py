from collections import OrderedDict
from pathlib import Path

import torch

MODEL_URLS = {
    "sd7k": "https://drive.google.com/uc?export=download&confirm=t&id=1a1dGcSFYB1ocR5KohSTjXIQcpQ0w05V7",
    "jung": "https://drive.google.com/uc?export=download&confirm=t&id=19i0ms_5Cv2tOE6SmL7vyms_gCdZCZQDt",
    "kligler": "https://drive.google.com/uc?export=download&confirm=t&id=1JEmtyGeyhCNdZ9_yhhEYJSeFTgQr5XYw",
}


def load_checkpoint(model: torch.nn.Module, weights: str, device) -> None:
    # Check if local path
    if Path(weights).exists():
        checkpoint = torch.load(weights, map_location=str(device))
    else:
        # Download
        assert (
            weights.lower() in MODEL_URLS.keys()
        ), f"DocShadow has only been trained on {MODEL_URLS.keys()}"
        checkpoint = torch.hub.load_state_dict_from_url(
            MODEL_URLS[weights.lower()],
            file_name=f"{weights.lower()}.pth",
            map_location=str(device),
        )

    new_state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module"):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value

    model.load_state_dict(new_state_dict)
