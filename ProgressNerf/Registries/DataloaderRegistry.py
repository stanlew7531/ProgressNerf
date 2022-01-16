import torch
import numpy as np

DATALOADER_REGISTRY = {}


def register_dataloader(name):
    def register_dataloader_cls(cls):
        if name in DATALOADER_REGISTRY:
            raise ValueError('Cannot register duplicate dataloader ({})'.format(name))
        DATALOADER_REGISTRY[name] = cls
        return cls
    return register_dataloader_cls


def get_dataloader(name):
    return DATALOADER_REGISTRY.get(name, None)