from tracemalloc import start
import torch
import numpy as np
import cv2 as cv

class VoxelGrid(object):
    # axesMinMax: (2,3) - (min,max) x (x,y,z axes)
    def __init__(self, axesMinMax:torch.Tensor, voxelSize:torch.double, stored_data_size:int):
        self.volume_bounds = axesMinMax
        self.axesLengths = axesMinMax[1,:] - axesMinMax[0,:]
        self.voxelSize = voxelSize
        self.shape = torch.ceil(self.axesLengths / voxelSize).to(torch.int)
        # update the axes length to reflect the increased volume from the ceil operation
        self.axesLengths = self.shape * voxelSize
        self.voxels = torch.zeros((self.shape[0].item(),self.shape[1].item(),self.shape[2].item(),stored_data_size), device=axesMinMax.device)

    def to(self, dvc:torch.device):
        self.axesLengths = self.axesLengths.to(dvc)
        self.shape = self.shape.to(dvc)
        self.voxels = self.voxels.to(dvc)

    # allows for directly getting voxel values by index (not 3d location)
    # is primarily used internally, but is exposed externally because seems like no reason not to
    def __getitem__(self, key):
        return self.voxels[key]

    # allows for directly setting voxel values by index (not 3d location)
    # is primarily used internally, but is exposed externally because seems like no reason not to
    def __setitem__(self, key, value):
        self.voxels[key] = value

    def size(self, dim=None):
        return self.voxels.size(dim)

    # xyz_locations: (N, 3) 3d locations to get the voxel values for
    # returns: (N, stored_data_size)
    def get_voxels_xyz(self, xyz_locations:torch.Tensor):
        voxel_locations = torch.floor((xyz_locations[:] - self.volume_bounds[0,:]) / self.voxelSize).to(torch.long)
        return self[voxel_locations[:,0], voxel_locations[:,1], voxel_locations[:,2]]

    # xyz_locations: (N, 3) 3d locations to get the voxel values for
    # values: (N, stored_data_size) data to assign to each voxel location
    def set_voxels_xyz(self, xyz_locations:torch.Tensor, values:torch.Tensor):
        voxel_locations = torch.floor((xyz_locations[:] - self.volume_bounds[0,:]) / self.voxelSize).to(torch.long)
        self[voxel_locations[:,0], voxel_locations[:,1], voxel_locations[:,2]] = values[:]

    # xyz_locations: (N, 3) 3d locations to determine if they are inside the top level bounding box
    def are_voxels_xyz_in_bounds(self, xyz_locations:torch.Tensor):
        voxel_locations = torch.floor((xyz_locations[:] - self.volume_bounds[0,:]) / self.voxelSize).to(torch.long)
        is_in_bounds = torch.all(torch.logical_and(voxel_locations >= torch.zeros(3,device = self.voxels.device), voxel_locations < self.shape), dim = -1) # (N)
        #in_bound_voxels = voxel_locations[is_in_bounds].reshape((int(is_in_bounds.shape[0] / 3), int(3)))
        return is_in_bounds

    
    def subdivideGrid(self):
        start_size = self.voxels.shape
        self.voxels = self.voxels.repeat_interleave(2, dim = 0)\
            .repeat_interleave(2, dim = 1)\
                .repeat_interleave(2, dim = 2)

        self.voxelSize = self.voxelSize * 0.5
        self.shape = self.voxels.shape

