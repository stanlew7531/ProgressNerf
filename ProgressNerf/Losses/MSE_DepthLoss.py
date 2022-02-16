import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.LossRegistry import register_loss, AbstractLoss

# this loss function is just like the standard MSELoss, except it also includes a depth component

@register_loss("mse_depth_loss")
class MSEDepthLoss(AbstractLoss):
    def __init__(self, config:Dict):
        self.loss_fn = torch.nn.MSELoss()
    
    def calculateLoss(self, gt_pixels:torch.Tensor, rendered_pixels:torch.Tensor, **kwargs):
        gt_depths = kwargs["gt_depths"]
        rendered_depths = kwargs["rendered_depths"]
        rendered_rgbd = torch.cat((rendered_pixels, rendered_depths), dim = -1)
        gt_rgbd = torch.cat((gt_pixels, gt_depths), dim = -1)
        mse_loss = self.loss_fn(gt_rgbd, rendered_rgbd)
        rgb_error = self.loss_fn(gt_pixels, rendered_pixels)
        depth_error = self.loss_fn(gt_depths, rendered_depths)
        return {"loss": mse_loss, "mse_rgb": rgb_error, "mse_depth": depth_error}