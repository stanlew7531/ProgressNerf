import torch
import numpy as np
from typing import Dict, Tuple
from abc import ABC, abstractmethod


class AbstractRaysampler(ABC):
    @abstractmethod
    def __init__(self, config:Dict):
        pass  

    @abstractmethod
    def sampleRays(self, ray_origins:torch.Tensor, ray_dirs:torch.Tensor, other_info:Dict)->Tuple[torch.Tensor, torch.Tensor]:
        pass

RAYSAMPLER_REGISTRY = {}


def register_raysampler(name):
    def register_raysampler_fn(fn):
        if name in RAYSAMPLER_REGISTRY:
            raise ValueError('Cannot register duplicate raysampler ({})'.format(name))
        RAYSAMPLER_REGISTRY[name] = fn
        return fn
    return register_raysampler_fn


def get_raysampler(name):
    return RAYSAMPLER_REGISTRY.get(name, None)