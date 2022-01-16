from google.protobuf.descriptor import Error
import ProgressNerf.Raycasting.csrc as _C
from typing import Dict
import torch
from ProgressNerf.Registries.RaysamplerRegistry import register_raysampler, AbstractRaysampler

@register_raysampler("weighted_raysampler")
class WeightedRaysampler(AbstractRaysampler):
    def __init__(self, config: Dict):
        self.num_samples = config["num_samples"]
    
    def sampleRays(self, ray_origins: torch.Tensor, ray_dirs: torch.Tensor, other_info:Dict=None):
        '''
        ray_origins: (batch_size, num_rays, 3)
        ray_dirs: (batch_size, num_rays, 3)

        other_info: needs to contain keys:
            'sigmas': (batch_size, num_rays, num_samples)
            'distances': (batch_size, num_rays, num_samples)
        these come from the previous sampling step (or are generated manually) and represent the (weight, distance) pair to resample from

        return:
        samples: (batch_size, num_rays, num_samples, 3)
        distances: (batch_size, num_rays, num_samples)
        '''

        distances = other_info['distances']
        sigmas = other_info['sigmas']
        weighted_resamples, weighted_resample_distances = _C.DoWeightedResampling(sigmas, distances, ray_origins, ray_dirs, self.num_samples)

        return weighted_resamples, weighted_resample_distances
