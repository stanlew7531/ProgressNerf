import torch
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod


class AbstractRaypicker(ABC):
    @abstractmethod
    def __init__(self, config:Dict):
        pass

    @abstractmethod
    def setCameraParameters(self, camera_matrix:torch.Tensor, vp_height, vp_width):
        pass        

    @abstractmethod
    def getRays(self, camera_origin:torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def getAllRays(self, camera_origin:torch.Tensor, **kwargs):
        pass

RAYPICKER_REGISTRY = {}


def register_raypicker(name):
    def register_raypicker_fn(fn):
        if name in RAYPICKER_REGISTRY:
            raise ValueError('Cannot register duplicate raypicker ({})'.format(name))
        RAYPICKER_REGISTRY[name] = fn
        return fn
    return register_raypicker_fn


def get_raypicker(name):
    return RAYPICKER_REGISTRY.get(name, None)