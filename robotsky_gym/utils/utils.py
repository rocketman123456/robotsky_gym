import torch
import importlib


def instantiate_cfg(entry: str):
    if entry is None:
        return None
    mod_name, attr_name = entry.split(":")
    mod = importlib.import_module(mod_name)
    cfg_cls = getattr(mod, attr_name)
    if callable(cfg_cls):
        cfg = cfg_cls()
    else:
        cfg = cfg_cls
    return cfg


class _TorchParam:
    torch_type = torch.float
    torch_device = None


def set_default_torch_type(torch_type):
    _TorchParam.torch_type = torch_type


def set_default_torch_device(torch_device):
    _TorchParam.torch_device = torch_device


def type_and_device(type=None, device=None):
    return {"dtype": type if type is not None else _TorchParam.torch_type, "device": device if device is not None else _TorchParam.torch_device}
