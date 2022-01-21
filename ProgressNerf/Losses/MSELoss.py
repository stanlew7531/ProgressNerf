import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.LossRegistry import register_loss, AbstractLoss

@register_loss("mse_loss")
class MSELoss(AbstractLoss):
    def __init__(self, config:Dict):
        self.loss_fn = torch.nn.MSELoss()
    
    def calculateLoss(self, gt_pixels:torch.Tensor, rendered_pixels:torch.Tensor, **kwargs):
        return self.loss_fn(gt_pixels, rendered_pixels)