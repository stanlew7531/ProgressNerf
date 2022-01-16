import torch
import numpy as np

MODEL_REGISTRY = {}


def register_model(name):
    def register_model_fn(fn):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        MODEL_REGISTRY[name] = fn
        return fn
    return register_model_fn


def get_model(name) -> torch.nn.Module:
    return MODEL_REGISTRY.get(name, None)