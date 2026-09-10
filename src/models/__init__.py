from .regnet_y_800_mf import get_regnet_y_800_mf
from .resnet18 import get_resnet18
from .resnet34 import get_resnet34
from .shufflenet_v2_x1_0 import get_shufflenet_v2_x1_0
from .shufflenet_v2_x1_5 import get_shufflenet_v2_x1_5

__all__ = [
    "get_regnet_y_800_mf",
    "get_resnet18",
    "get_resnet34",
    "get_shufflenet_v2_x1_0",
    "get_shufflenet_v2_x1_5",
    "get_model",
]


def get_model(model_name, num_classes, **kwargs):
    if model_name == "regnet_y_800_mf":
        return get_regnet_y_800_mf(num_classes=num_classes, **kwargs)
    elif model_name == "resnet18":
        return get_resnet18(num_classes=num_classes, **kwargs)
    elif model_name == "resnet34":
        return get_resnet34(num_classes=num_classes, **kwargs)
    elif model_name == "sn0":
        return get_shufflenet_v2_x1_0(num_classes=num_classes, **kwargs)
    elif model_name == "sn5":
        return get_shufflenet_v2_x1_5(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")
