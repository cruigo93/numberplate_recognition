import efficientnet_pytorch as efp
import importlib
import torch


def init_model():
    base = efp.EfficientNet.from_pretrained("efficientnet-b0")
    num_chars = 32

    CONFIG = {
        'PY': 'models',
        'ARCH': 'TextRecogModel'
    }

    module = importlib.import_module(CONFIG["PY"])
    model = getattr(module, CONFIG["ARCH"])(base, num_chars)

    model.load_state_dict(torch.load("baseline_46_weights.pt"))
    return model