import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.RaypickerRegistry import register_raypicker, AbstractRaypicker
from ProgressNerf.Utils.CameraUtils import ComputeCameraEpipolars
import cv2 as cv

@register_raypicker("random_raypicker")
class RandomRaypicker(AbstractRaypicker):
    def __init__(self, config:Dict):
        self.camera_initialized = False
        self.baseRays = None
        self.num_rays = config['num_rays']

    def setCameraParameters(self, camera_matrix: torch.Tensor, vp_height, vp_width):
        self.camera_matrix = camera_matrix
        self.width = vp_width
        self.height = vp_height
        self.baseRays = ComputeCameraEpipolars(self.camera_matrix, self.height, self.width)
        self.camera_initialized=True

    def getRays(self, camera_tfs: torch.Tensor, **kwargs):
        '''
        camera_tfs: batch_dim, 4, 4
        '''
        if(not self.camera_initialized):
            raise ValueError("getRays requires camera parameters to be initialized first!")
        if(self.num_rays is None):
            raise ValueError("num_rays must be specified first!")

        rays_to_pick = self.baseRays\
            .reshape((self.width * self.height, 3)).contiguous()\
                .to(dtype = camera_tfs.dtype, device = camera_tfs.device) # (W*H, 3)

        batch_size = camera_tfs.shape[0]
                
        picked_rays_cache = torch.randperm(rays_to_pick.shape[0])[:self.num_rays] # (self.num_rays, 1)
        picked_js = torch.remainder(picked_rays_cache, self.height).to(dtype=torch.int64)
        picked_is = torch.floor(torch.divide(picked_rays_cache, self.height)).to(dtype=torch.int64)
        picked_ijs = torch.cat((picked_is.unsqueeze(1), picked_js.unsqueeze(1)), dim=1).unsqueeze(0).repeat((batch_size,1,1))
        picked_rays = rays_to_pick[picked_rays_cache] # (self.num_rays, 3)
        camera_tfs = camera_tfs.unsqueeze(1) # (batch_dim, 1, 4, 4)

        # extract the ray origins from the camera pose
        ray_origins = camera_tfs[:,:,0:3,3].expand(-1, self.num_rays, -1)  # (batch_dim, self.num_rays, 3)

        ray_rotations = camera_tfs[:,:,0:3,0:3] # (batch_dim, 1, 3, 3)
        picked_rays = picked_rays.unsqueeze(0).unsqueeze(3) # (1, self.num_rays, 3, 1)
        ray_directions = torch.matmul(ray_rotations, picked_rays).squeeze(3) # (batch_dim, self.num_rays, 3)

        return ray_origins, ray_directions, picked_ijs

    def getAllRays(self, camera_tfs: torch.Tensor):
        '''
        camera_tfs: batch_dim, 4, 4
        '''
        if(not self.camera_initialized):
            raise ValueError("getRays requires camera parameters to be initialized first!")
        if(self.num_rays is None):
            raise ValueError("num_rays must be specified first!")

        camera_tfs = camera_tfs.unsqueeze(1) # (batch_dim, 1, 4, 4)

        # extract the ray origins from the camera pose
        ray_origins = camera_tfs[:,:,0:3,3].expand(-1, self.width * self.height, -1)  # (batch_dim, width*height, 3)

        ray_rotations = camera_tfs[:,:,0:3,0:3] # (batch_dim, 1, 3, 3)
        picked_rays = self.baseRays.reshape((self.width * self.height, 3)).unsqueeze(0).unsqueeze(3).to(device=camera_tfs.device) # (1, width * height, 3, 1)
        ray_directions = torch.matmul(ray_rotations, picked_rays).squeeze(3) # (batch_dim, width*height, 3)

        return ray_origins, ray_directions
