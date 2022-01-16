import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.RaysamplerRegistry import register_raysampler, AbstractRaysampler

@register_raysampler("nearfarsampler")
class NearFarRaysampler(AbstractRaysampler):
    def __init__(self, config: Dict):
        self.near = config["near_plane"]
        self.far = config["far_plane"]
        self.num_samples = config["num_samples"]
    
    def sampleRays(self, ray_origins: torch.Tensor, ray_dirs: torch.Tensor, other_info:Dict):
        '''
        ray_origins: (batch_size, num_rays, 3)
        ray_dirs: (batch_size, num_rays, 3)

        return:
        samples: (batch_size, num_rays, num_samples, 3)
        distances: (batch_size, num_rays, num_samples)
        '''
        batch_size, num_rays, _ = ray_origins.shape

        distances = torch.linspace(self.near, self.far, self.num_samples, device=ray_origins.device, dtype=ray_origins.dtype) # (self.num_samples)
        distances = distances.unsqueeze(1).repeat(1, 3) # (self.num_samples, 3)
        ray_origins = ray_origins.unsqueeze(2).repeat(1,1, self.num_samples, 1) # (batch_size, num_rays, num_samples, 3)
        ray_dirs = ray_dirs.unsqueeze(2).repeat(1,1, self.num_samples, 1) # (batch_size, num_rays, num_samples, 3)
        samples = (ray_dirs * distances)  + ray_origins
        distances = distances[:,0].unsqueeze(0).unsqueeze(1).repeat(batch_size, num_rays, 1)
        return samples, distances