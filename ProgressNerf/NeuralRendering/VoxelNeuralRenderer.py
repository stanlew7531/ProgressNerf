import torch
import numpy as np
from typing import Dict
from ProgressNerf.NeuralRendering.NeuralRenderer import NeuralRenderer
from ProgressNerf.Registries.RendererRegistry import register_renderer, AbstractRenderer

@register_renderer("voxel_neural_renderer")
class VoxelNeuralRenderer(AbstractRenderer):
    def __init__(self, config: Dict):
        self.base_renderer = NeuralRenderer(config)

    def renderRays(self, raySamples: torch.Tensor, distances: torch.Tensor, **kwargs):
        '''
        raySamples: (batch_size, num_rays, num_samples, 4)
        distances: (batch_size, num_rays, num_samples)

        kwargs contains:
        voxels: ProgressNerf.Utils.VoxelGrid
        sample_locations: (batch_size, num_rays, num_samples, 3)

        return: (batch_size, num_rays, 4)
        '''
        voxel_grid = kwargs['voxels']
        sample_locs = kwargs['sample_locations']
        batch_size, num_rays, num_samples, _ = sample_locs.shape
        # resize the sample locations to match what the voxel grid helper functions want
        sample_locs = sample_locs.reshape((batch_size * num_rays * num_samples, 3))
        in_bounds_results = voxel_grid.are_voxels_xyz_in_bounds(sample_locs) # (batch_size * num_rays * num_samples)
        occupied_results = in_bounds_results.clone()
        # only include rays which intersect the bounding volume, and only include samples inside non-pruned voxels
        if(torch.any(in_bounds_results).item()):
            voxel_vals = voxel_grid.get_voxels_xyz(sample_locs[in_bounds_results, :])
            occupied_results[in_bounds_results] = (voxel_vals[:, 0] > 0) # (batch_size * num_rays * num_samples)
        occupied_results = occupied_results.reshape((batch_size, num_rays, num_samples)) # (batch_size * num_rays * num_samples)
        
        return self.base_renderer.renderRays(raySamples = raySamples[...,:] * occupied_results[...,None], distances = distances)
