import torch
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod


class AbstractEncoder(ABC):
    @abstractmethod
    def __init__(self, config:Dict):
        pass

    @abstractmethod
    def encodeFeature(self, feature:torch.Tensor):
        pass        

ENCODER_REGISTRY = {}


def register_encoder(name):
    def register_encoder_fn(fn):
        if name in ENCODER_REGISTRY:
            raise ValueError('Cannot register duplicate encoder ({})'.format(name))
        ENCODER_REGISTRY[name] = fn
        return fn
    return register_encoder_fn


def get_encoder(name):
    return ENCODER_REGISTRY.get(name, None)