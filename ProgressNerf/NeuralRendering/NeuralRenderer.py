import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.RendererRegistry import register_renderer, AbstractRenderer

@register_renderer("neural_renderer")
class NeuralRenderer(AbstractRenderer):
    def __init__(self, config: Dict):
        pass

    def renderRays(self, raySamples: torch.Tensor, distances: torch.Tensor, **kwargs):
        '''
        raySamples: (batch_size, num_rays, num_samples, 4)
        distances: (batch_size, num_rays, num_samples)

        return: (batch_size, num_rays, 4)
        '''
        batch_size, num_rays, num_samples = distances.shape

        deltas = distances[:, :, 1:] - distances[:, :, :-1]  # (batch_size, num_rays, num_samples-1)
        dist_far = torch.empty(size=(batch_size, num_rays, 1), dtype=torch.float32, device=distances.device).fill_(1e10)  # (batch_size, num_rays, 1)
        deltas = torch.cat([deltas, dist_far], dim=2)  # (batch_size, num_rays, num_samples)

        rgb_samples = raySamples[:,:,:,0:3] # (batch_size, num_rays, num_samples, 3)
        sigmas = raySamples[:,:,:,3].relu() # (batch_size, num_rays, num_samples, 1)
        alphas = 1 - torch.exp(-1.0 * sigmas * deltas) # (batch_size, num_rays, num_samples)

        Tis = torch.roll(torch.exp(-1 * torch.cumsum(input=(sigmas * deltas), dim = 2)), shifts=1, dims=2) # (batch_size, num_rays, num_samples)
        Tis[:,:,0] = 1.0 # (batch_size, num_rays, num_samples)

        weights = alphas * Tis # (batch_size, num_rays, num_samples)
        
        rgb_rendered = torch.sum(weights.unsqueeze(3) * rgb_samples, dim=2).sigmoid() # (batch_size, num_rays, 3)
        depth_rendered = torch.sum(weights * distances, dim=2)  # (batch_size, num_rays)

        render_result = {
            "rgb" : rgb_rendered, # (batch_size, num_rays, 3)
            #"weights" : weights, # (batch_size, num_rays, num_samples)
            "depth" : depth_rendered, # (batch_size, num_rays)
        }

        return render_result