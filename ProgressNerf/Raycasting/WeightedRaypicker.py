import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.RaypickerRegistry import register_raypicker, AbstractRaypicker
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
        self.baseRays = self.__initBaseRays()
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

    def __initBaseRays(self):
        """Compute base ray directions in camera coordinates, which only depends on intrinsics.
        These will be further converted to world coordinates later, using camera poses.
        :return: (W, H, 3) torch.float32
        """
        y, x = torch.meshgrid(torch.arange(self.height, dtype=torch.float32),
                            torch.arange(self.width, dtype=torch.float32))  # (H, W)

        # Use OpenCV coordinate in 3D:
        #   +x points to right
        #   +y points to down
        #   +z points to forward
        #
        # The coordinate of the top left corner of an image should be (-0.5W, -0.5H, 1.0), then normed to be magnitude 1.0
        dirs_x = (x - 0.5*self.width) / self.camera_matrix[0,0]  # (H, W)
        dirs_y = (y - 0.5*self.height) / self.camera_matrix[1,1]  # (H, W)
        dirs_z = torch.ones((self.height, self.width), dtype=torch.float32)  # (H, W)
        rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
        rays_dir = rays_dir.transpose(0,1).contiguous() # put into row-major format
        rays_dir = torch.reshape(rays_dir, (self.height * self.width, 3)) # (H*W, 3)
        norms = torch.linalg.norm(rays_dir, dim=1, keepdim=True) # (H*W, 1)
        rays_dir = rays_dir / norms # (H*W, 3)
        rays_dir = torch.reshape(rays_dir, (self.width, self.height, 3)).contiguous() # (W, H, 3)
        return rays_dir