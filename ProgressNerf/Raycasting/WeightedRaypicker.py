import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.RaypickerRegistry import register_raypicker, AbstractRaypicker
from ProgressNerf.Utils.CameraUtils import ComputeCameraEpipolars
import cv2 as cv

@register_raypicker("weighted_raypicker")
class WeightedRaypicker(AbstractRaypicker):
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

        kwargs will need to contain a Tensor of size (batch_dim, height, width) with key 'ray_weights'
        that is used for randomly selecting rays
        '''
        if(not self.camera_initialized):
            raise ValueError("getRays requires camera parameters to be initialized first!")
        if("ray_weights" not in kwargs.keys()):
            raise ValueError("getRays in WeightedRaypicker requires a ray weights parameter!")
        if(self.num_rays is None):
            raise ValueError("num_rays must be specified first!")

        ray_weights = kwargs["ray_weights"] # (batch_dim, height, width)
        ray_weights = ray_weights.transpose(-1,-2).contiguous()
        batch_size = ray_weights.shape[0]

        rays_to_pick = self.baseRays\
            .reshape((self.width * self.height, 3))\
                .repeat(batch_size, 1, 1).contiguous()\
                .to(dtype = camera_tfs.dtype, device = camera_tfs.device)# (batch_dim, W*H, 3)

        picked_rays = torch.multinomial(ray_weights.reshape((batch_size, self.width * self.height)), self.num_rays, replacement=True) # (batch_dim, self.num_rays)
        picked_js = torch.remainder(picked_rays, self.height).to(dtype=torch.int64) #(batch_dim, self.num_rays)
        picked_is = torch.floor(torch.divide(picked_rays, self.height)).to(dtype=torch.int64) #(batch_dim, self.num_rays)
        picked_ijs = torch.cat((picked_is.unsqueeze(2), picked_js.unsqueeze(2)), dim=2) #(batch_dim, self.num_rays, 2)
        picked_rays = torch.gather(rays_to_pick,1,picked_rays.unsqueeze(-1).repeat((1,1,3))) # (batch_dim, self.num_rays, 3)
        camera_tfs = camera_tfs.unsqueeze(1) # (batch_dim, 1, 4, 4)

        # extract the ray origins from the camera pose
        ray_origins = camera_tfs[:,:,0:3,3].expand(-1, self.num_rays, -1)  # (batch_dim, self.num_rays, 3)

        ray_rotations = camera_tfs[:,:,0:3,0:3] # (batch_dim, 1, 3, 3)
        picked_rays = picked_rays.unsqueeze(3) # (batch_dim, self.num_rays, 3, 1)
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