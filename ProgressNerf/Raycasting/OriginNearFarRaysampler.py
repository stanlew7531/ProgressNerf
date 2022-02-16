import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.RaysamplerRegistry import register_raysampler, AbstractRaysampler

@register_raysampler("origin_nearfarsampler")
class OriginNearFarRaysampler(AbstractRaysampler):
    def __init__(self, config: Dict):
        self.length = config["plus_minus_length"]
        self.num_samples = config["num_samples"]
    
    def sampleRays(self, ray_origins: torch.Tensor, ray_dirs: torch.Tensor, other_info:Dict=None):
        '''
        ray_origins: (batch_size, num_rays, 3)
        ray_dirs: (batch_size, num_rays, 3)

        return:
        samples: (batch_size, num_rays, num_samples, 3)
        distances: (batch_size, num_rays, num_samples)
        '''
        batch_size, num_rays, _ = ray_origins.shape
        dot_prod = torch.sum(ray_dirs * (-1 * ray_origins), dim=-1) #(batch_size, num_rays)
        mag = dot_prod #(batch_size, num_rays) # TODO: add math here to support ray_dirs that are not already magnitude 1
        mag = mag.unsqueeze(-1).repeat((1,1,self.num_samples)) #(batch_size, num_rays, num_samples)
        distances = torch.linspace(-1 * self.length, self.length, self.num_samples, device=ray_origins.device, dtype=ray_origins.dtype) # (self.num_samples)
        distances = distances + mag #(batch_size, num_rays, num_samples)
        distances = distances.unsqueeze(-1).repeat((1, 1, 1, 3)) # (batch_size, num_rays, num_samples, 3)
        ray_origins = ray_origins.unsqueeze(2).repeat(1,1, self.num_samples, 1) # (batch_size, num_rays, num_samples, 3)
        ray_dirs = ray_dirs.unsqueeze(2).repeat(1,1, self.num_samples, 1) # (batch_size, num_rays, num_samples, 3)
        samples = (ray_dirs * distances)  + ray_origins
        distances = distances[:,:,:,0]
        return samples, distances