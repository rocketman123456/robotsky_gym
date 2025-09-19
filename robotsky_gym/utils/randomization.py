import torch
from torch import Tensor

from typing import Union, Tuple
import numpy as np

_use_random = True


def disable_randomization():
    global _use_random
    _use_random = False


def enable_randomization():
    global _use_random
    _use_random = True


def apply_randomization(tensor, params, return_noise=False) -> Tensor | Tuple[Tensor, Tensor]:
    if params == None:
        return tensor

    if params["distribution"] == "gaussian":
        mu, var = params["range"]
        noise = torch.randn_like(tensor)  # if isinstance(tensor, torch.Tensor) else np.random.randn()
        noise_val = mu + var * noise
    elif params["distribution"] == "uniform":
        lower, upper = params["range"]
        noise = torch.rand_like(tensor)  # if isinstance(tensor, torch.Tensor) else np.random.rand()
        noise_val = lower + (upper - lower) * noise
    else:
        raise ValueError(f"Invalid randomization distribution: {params['distribution']}")

    if params["operation"] == "additive":
        result = tensor + noise_val
    elif params["operation"] == "scaling":
        result = tensor * noise_val
    else:
        raise ValueError(f"Invalid randomization operation: {params['operation']}")

    if return_noise:
        return result, noise
    else:
        return result


def randomize(tensor, params, return_noise=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if _use_random:
        return apply_randomization(tensor, params, return_noise)
    else:
        if return_noise:
            noise = torch.zeros_like(tensor) if isinstance(tensor, torch.Tensor) else 0.0
            return tensor, noise
        else:
            return tensor
