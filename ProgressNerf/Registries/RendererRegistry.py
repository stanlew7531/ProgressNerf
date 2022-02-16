import torch
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod


class AbstractRenderer(ABC):
    @abstractmethod
    def __init__(self, config:Dict):
        pass  

    @abstractmethod
    def renderRays(self, raySamples: torch.Tensor, deltas: torch.Tensor, **kwargs):
        pass

RENDERER_REGISTRY = {}


def register_renderer(name):
    def register_renderer_fn(fn):
        if name in RENDERER_REGISTRY:
            raise ValueError('Cannot register duplicate renderer ({})'.format(name))
        RENDERER_REGISTRY[name] = fn
        return fn
    return register_renderer_fn


def get_renderer(name):
    return RENDERER_REGISTRY.get(name, None)