import torch
import numpy as np
import math
from typing import Dict
from ProgressNerf.Registries.EncoderRegistry import AbstractEncoder, register_encoder

@register_encoder("positional_encoder")
class PositionalEncoder(AbstractEncoder):
    def __init__(self, config: Dict):
        self.L = config["levels"]
        self.includeOriginal = config["includeOrig"] if("includeOrig" in config.keys()) else False

    def encodeFeature(self, feature: torch.Tensor):
        '''
        feature: tensor of arbitrary size/dims: (dim_0,dim_1,...,dim_n,N)
        the tensor will be reshaped to a 2D tensor of size (dim_0*...*dim_n,N)
        before encoding is applied, then reshaped back to its original 0-n dims

        return:
        samples: (dim_0, dim_1,...,dim_n, 2 * N * self.L)
        '''
        og_shape = feature.shape

        reshape_d0 = 1
        for i in range(len(og_shape) - 1):
            reshape_d0 *= og_shape[i]

        coefficients = torch.linspace(0, self.L, device=feature.device, dtype=feature.dtype, steps = self.L) # create the powers
        coefficients = torch.pow(2 * torch.ones_like(coefficients), coefficients) # raise 2 by the created powers
        coefficients = coefficients * math.pi # add the pi component
        coefficients = coefficients.unsqueeze(0).repeat((reshape_d0, og_shape[-1], 1)) # match the size of the input tensor

        trig_fn_inputs = coefficients * feature.reshape((reshape_d0, og_shape[-1],1))
        all_results = torch.cat((torch.sin(trig_fn_inputs), torch.cos(trig_fn_inputs)), dim=1)

        new_shape = list(og_shape)
        new_shape[-1] = new_shape[-1] * self.L * 2
            
        new_shape = torch.Size(new_shape)

        restored_results = all_results.reshape(new_shape).contiguous()

        if(self.includeOriginal):
            restored_results = torch.cat((feature, restored_results), dim = -1)

        return restored_results