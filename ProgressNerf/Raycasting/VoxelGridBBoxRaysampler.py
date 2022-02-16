import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.RaysamplerRegistry import register_raysampler, AbstractRaysampler

@register_raysampler("voxel_grid_bbox")
class VoxelGridBBoxRaysampler(AbstractRaysampler):
    def __init__(self, config: Dict):
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
        voxel_grid = other_info['voxel_grid']
        min_pts, max_pts ,min_taus, max_taus = voxel_grid.getRayGridIntersections(ray_origins.flatten(0,-2), ray_dirs.flatten(0,-2)) # min_taus & max_taus: (batch_size*num_rays)
        deltas = (max_taus - min_taus).reshape(batch_size, num_rays, 1).repeat((1,1,self.num_samples))
        alphas = torch.linspace(0.0, 1.0, self.num_samples, device=ray_origins.device, dtype=ray_origins.dtype)
        deltas = deltas * alphas
        distances = min_taus.reshape(batch_size, num_rays,1).repeat((1,1,self.num_samples)) + deltas
        distances = distances.unsqueeze(-1).repeat((1, 1, 1, 3)) # (batch_size, num_rays, num_samples, 3)
        ray_origins = ray_origins.unsqueeze(2).repeat(1,1, self.num_samples, 1) # (batch_size, num_rays, num_samples, 3)
        ray_dirs = ray_dirs.unsqueeze(2).repeat(1,1, self.num_samples, 1) # (batch_size, num_rays, num_samples, 3)
        samples = (ray_dirs * distances)  + ray_origins
        distances = distances[:,:,:,0]
        return samples, distances