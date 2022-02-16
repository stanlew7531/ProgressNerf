import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.LossRegistry import register_loss, AbstractLoss

# this loss function is just like the standard MSELoss, except it also includes a depth component

@register_loss("mse_loss_beta_reg")
class MSEBetaRegLoss(AbstractLoss):
    def __init__(self, config:Dict):
        self.loss_fn = torch.nn.MSELoss()
        self.reg_lambda = config['beta_regularization_term']
    
    def calculateLoss(self, gt_pixels:torch.Tensor, rendered_pixels:torch.Tensor, **kwargs):
        alphas = kwargs["alphas"] #(batch_size, num_rays)
        alphas = (alphas * (1. - 1e-7)) + 0.5e-7 # use to prevent the beta dist from evaluating to inf
        reg_value = self.reg_lambda * (torch.mean(torch.log(alphas) + torch.log(1.0 - alphas)))
        rgb_error = self.loss_fn(gt_pixels, rendered_pixels)
        return {"loss": rgb_error + reg_value, "rgb_error": rgb_error, "reg_value": reg_value}