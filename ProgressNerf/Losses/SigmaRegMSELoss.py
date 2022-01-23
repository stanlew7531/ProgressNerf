import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.LossRegistry import register_loss, AbstractLoss

# this loss function is just like the standard MSELoss, except it includes a regularization term
# on the MLP sigma outputs. This encourages the network to output sigmas closer to 0.0, which 
# should assist in reducing rendering artifacts among non-supervised pixels

@register_loss("mse_loss_sigma_regularized")
class SigmaRegMSELoss(AbstractLoss):
    def __init__(self, config:Dict):
        self.loss_fn = torch.nn.MSELoss()
        self.reg_lambda = config['sigma_regularization_term']
    
    def calculateLoss(self, gt_pixels:torch.Tensor, rendered_pixels:torch.Tensor, **kwargs):
        sigmas = kwargs["sigma_vals"] # (batch_dim, height, width)
        regularization_term = self.reg_lambda * torch.mean(sigmas)
        mse_loss = self.loss_fn(gt_pixels, rendered_pixels)
        return {"loss": self.loss_fn(gt_pixels, rendered_pixels) + regularization_term, "mse": mse_loss, "sigma_reg": regularization_term}