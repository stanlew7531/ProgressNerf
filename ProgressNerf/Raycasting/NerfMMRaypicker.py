import torch
import numpy as np
from typing import Dict
from ProgressNerf.Registries.RaypickerRegistry import register_raypicker, AbstractRaypicker, get_raypicker
from ProgressNerf.Utils.CameraUtils import ComputeCameraEpipolars
from ProgressNerf.Utils.LieUtils import SE3, SE3Exp
import cv2 as cv

@register_raypicker("nerfmm_raypicker")
class NerfMMRaypicker(AbstractRaypicker):
    def __init__(self, config:Dict):
        self.camera_initialized = False
        self.baseRays = None
        self.base_raypicker = get_raypicker(config['base_raypicker'])(config[config['base_raypicker']])
        self.num_rays = self.base_raypicker.num_rays
        self.pose_modifications = None

    # make sure to use the base raypicker's setCameraParameters functionality
    def setCameraParameters(self, camera_matrix: torch.Tensor, vp_height, vp_width):
        self.base_raypicker.setCameraParameters(camera_matrix, vp_height, vp_width)
        self.camera_initialized = self.base_raypicker.camera_initialized

    def setDynamicParameters(self, **kwargs):
        self.initPoseModifications(kwargs['num_images'], kwargs['device'])

    # method used for initializing the memory to store the pose modifications
    def initPoseModifications(self, num_images: int, device: torch.device):
        self.pose_modifications = torch.zeros((num_images, 6), device = device, dtype = torch.float32, requires_grad = True)

    def getRays(self, camera_tfs:torch.Tensor, **kwargs):
        '''
        camera_tfs: batch_dim, 4, 4
        '''
        # get the data index and get the pose modifications
        idxs = kwargs['data_idx'].squeeze() # batch_dim
        #print(idxs)
        # if this is first time, allocate the memory for the modifications
        #if(self.pose_modifications is None):
        #    self.initPoseModifications(kwargs['num_images'], camera_tfs.device)
        delta_poses = self.pose_modifications[idxs] # batch_dim, 6

        #print(delta_poses.shape)
        #raise Exception("stopping to print")
        tfs = SE3Exp(delta_poses) # batch_dim, 4, 4
        refined_tfs = torch.matmul(tfs, camera_tfs)
        return self.base_raypicker.getRays(refined_tfs, **kwargs)

    def getAllRays(self, camera_tfs:torch.Tensor, **kwargs):
        return self.base_raypicker.getAllRays(camera_tfs, **kwargs)

    def getLearningParams(self):
        return self.pose_modifications
