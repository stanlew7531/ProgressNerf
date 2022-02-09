from importlib.machinery import FrozenImporter
import torch
import numpy as np
import cv2 as cv
from ProgressNerf.Utils.CameraUtils import CamPoseDepthToXYZ
from ProgressNerf.Utils.VoxelGrid import VoxelGrid

class ParticleField(object):
    def __init__(self, cam_matrix:torch.Tensor, output_height:int, output_width:int):
        self.points = torch.empty((0,4), device=cam_matrix.device, dtype=torch.float32) # (N points, 4) (4 to make homogenous)
        self.cam_matrix = cam_matrix
        self.output_height = output_height
        self.output_width = output_width

    def to(self, device:torch.device):
        self.points = self.points.to(device)
        self.cam_matrix = self.cam_matrix.to(device)

    # cam_poses: (batch_size, 4,4)
    # segmentation: (batch_size, width, height)
    # depth: (batch_size, width, height)
    def appendPoints(self, cam_poses:torch.Tensor, segmentation:torch.Tensor, depth:torch.Tensor):
        # from the cam pose, project the depth points in the segmentation into R3 and add to list
        new_points = CamPoseDepthToXYZ(self.cam_matrix, cam_poses, depth)
        new_points = new_points[segmentation == True, :] #( segmentation_N, 3)
        to_homogenous = torch.ones((new_points.shape[0],1), device=new_points.device, dtype=new_points.dtype) # (segmentation_N, 1)
        new_points = torch.cat((new_points, to_homogenous), dim=-1) # (segmentation_N, 4)
        self.points = torch.cat((self.points, new_points)) #(N points + segmentation_N, 4)
        # make sure to eliminate duplicate points
        self.points = torch.unique(self.points, dim = 0) #(new N points, 4)

    def filterByVoxelGrid(self, voxel_grid:VoxelGrid):
        in_bounds_results = voxel_grid.are_voxels_xyz_in_bounds(self.points[:,0:3])
        self.points = self.points[in_bounds_results]

    # cam_poses: (batch_size, 4,4)
    def getPointsSegmentation(self, cam_poses:torch.Tensor): 
        # put the existing point cloud into the new camera's frame
        points_cam_frame = torch.matmul(torch.linalg.inv(cam_poses).unsqueeze(1), self.points.unsqueeze(-1)).squeeze(-1) #(batch_size, N points, 4)

        # apply the camera matrix to get i,js
        ijs = torch.matmul(self.cam_matrix.unsqueeze(0).unsqueeze(0), points_cam_frame[:,:,:-1].unsqueeze(-1)).squeeze(-1) # (batch_size, N points, 3)
        ijs = torch.floor((ijs[:,:,0:2] / ijs[:,:,2:])).to(dtype=torch.long) #(batch_size, N_Points, 2)
        to_return = torch.zeros((cam_poses.shape[0], self.output_width, self.output_height), device=cam_poses.device, dtype=torch.bool)
        for batch_idx in range(cam_poses.shape[0]):
            batch = to_return[batch_idx]
            batch[ijs[batch_idx,...,0], ijs[batch_idx,...,1]] = True
            to_return[batch_idx] = batch
        return to_return
