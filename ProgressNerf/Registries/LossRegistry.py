import torch
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod


class AbstractLoss(ABC):
    @abstractmethod
    def __init__(self, config:Dict):
        pass

    @abstractmethod
    def calculateLoss(self, gt_pixels:torch.Tensor, rendered_pixels:torch.Tensor):
        pass        

LOSS_REGISTRY = {}


def register_loss(name):
    def register_loss_fn(fn):
        if name in LOSS_REGISTRY:
            raise ValueError('Cannot register duplicate loss function ({})'.format(name))
        LOSS_REGISTRY[name] = fn
        return fn
    return register_loss_fn


def get_loss(name):
    return LOSS_REGISTRY.get(name, None)