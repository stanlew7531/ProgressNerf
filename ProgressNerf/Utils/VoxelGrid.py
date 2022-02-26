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
        self.shape = torch.floor(self.axesLengths / voxelSize).to(torch.int) + 1
        # update the axes length to reflect the increased volume from the ceil operation
        self.axesLengths = self.shape * voxelSize
        self.voxels = torch.zeros((self.shape[0].item(),self.shape[1].item(),self.shape[2].item(),stored_data_size), device=axesMinMax.device)
        self.shape = torch.cat((self.shape, torch.Tensor([stored_data_size]).to(device=self.shape.device))) #update with the stored_data_size

        self.points = torch.zeros((8,3), dtype=torch.float32, device=axesMinMax.device)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.points[i*4 + j*2 + k] = torch.Tensor([self.volume_bounds[i,0], self.volume_bounds[j,1], self.volume_bounds[k,2]]).to(device=axesMinMax.device)

    def save(self, file_path:str):
        previous_device = self.voxels.device
        self.to("cpu")
        to_write_dict = {'voxel_size': self.voxelSize, 'volume_bounds': self.volume_bounds, 'voxel_data': self.voxels}
        torch.save(to_write_dict, file_path)
        self.to(previous_device)

    @staticmethod
    def load(file_path:str):
        saved_dict = torch.load(file_path)
        voxel_size = saved_dict['voxel_size']
        volume_bounds = saved_dict['volume_bounds']
        voxel_data = saved_dict['voxel_data']
        toReturn = VoxelGrid(volume_bounds, voxel_size, voxel_data.shape[-1])
        toReturn.voxels = voxel_data

        return toReturn

    def to(self, dvc:torch.device):
        self.axesLengths = self.axesLengths.to(dvc)
        self.volume_bounds = self.volume_bounds.to(dvc)
        self.shape = self.shape.to(dvc)
        self.voxels = self.voxels.to(dvc)
        self.points = self.points.to(dvc)

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

    def get_voxel_locs(self, xyz_locations:torch.Tensor):
        return torch.floor((xyz_locations[:] - self.volume_bounds[0,:]) / self.voxelSize).to(torch.long)

    # xyz_locations: (N, 3) 3d locations to get the voxel values for
    # returns: (N, stored_data_size)
    def get_voxels_xyz(self, xyz_locations:torch.Tensor):
        voxel_locations = self.get_voxel_locs(xyz_locations) #torch.floor((xyz_locations[:] - self.volume_bounds[0,:]) / self.voxelSize).to(torch.long)
        return self[voxel_locations[:,0], voxel_locations[:,1], voxel_locations[:,2]]

    # xyz_locations: (N, 3) 3d locations to get the voxel values for
    # values: (N, stored_data_size) data to assign to each voxel location
    def set_voxels_xyz(self, xyz_locations:torch.Tensor, values:torch.Tensor):
        voxel_locations = self.get_voxel_locs(xyz_locations)
        self[voxel_locations[:,0], voxel_locations[:,1], voxel_locations[:,2]] = values[:]

    # xyz_locations: (N, 3) 3d locations to determine if they are inside the top level bounding box
    def are_voxels_xyz_in_bounds(self, xyz_locations:torch.Tensor):
        voxel_locations = self.get_voxel_locs(xyz_locations)
        is_in_bounds = torch.all(torch.logical_and(voxel_locations >= torch.zeros(3,device = self.voxels.device), voxel_locations < self.shape[0:3]), dim = -1) # (N)
        return is_in_bounds

    
    def subdivideGrid(self):
        self.voxels = self.voxels.repeat_interleave(2, dim = 0)\
            .repeat_interleave(2, dim = 1)\
                .repeat_interleave(2, dim = 2)

        self.voxelSize = self.voxelSize * 0.5
        self.shape[0:3] = self.shape[0:3] * 2

    # cam_matrix: (3, 3)
    # cam_poses: (N, 4, 4)
    def getVoxelGridBBox(self, cam_matrix, cam_poses, render_height, render_width):
        points_homo = torch.cat([self.points, torch.ones((8,1), device=self.points.device)], dim = -1).unsqueeze(-1) # (8, 4, 1)
        points_cam_frame = torch.matmul(torch.linalg.inv(cam_poses).unsqueeze(1), points_homo)# (N, 8, 4, 1)
        ijzs = torch.matmul(cam_matrix.unsqueeze(0), points_cam_frame[...,0:3,:].unsqueeze(-3)).squeeze() # (N, 8, 3)
        ijzs = ijzs[...,0:2]/ijzs[...,2:] # (N, 8, 2)
        to_return = torch.zeros((ijzs.shape[0], render_height, render_width), device = self.points.device) # (N, H, W)        
        for idx in range(ijzs.shape[0]):
            max_i = torch.max(ijzs[idx,:,0]).to(dtype=torch.long)
            min_i = torch.min(ijzs[idx,:,0]).to(dtype=torch.long)
            max_j = torch.max(ijzs[idx,:,1]).to(dtype=torch.long)
            min_j = torch.min(ijzs[idx,:,1]).to(dtype=torch.long)
            to_return[idx,min_j:max_j, min_i:max_i] = 1.0
        return to_return, ijzs


    # ray_origins: (N, 3)
    # ray_dirs: (N, 3)
    def getRayGridIntersections(self, ray_origins, ray_dirs):

        # compute the taus which represent the distance for intersecting each of the 6 bounding planes
        x0 = self.volume_bounds[0,0] # (1)
        x1 = self.volume_bounds[1,0] # (1)
        y0 = self.volume_bounds[0,1] # (1)
        y1 = self.volume_bounds[1,1] # (1)
        z0 = self.volume_bounds[0,2] # (1)
        z1 = self.volume_bounds[1,2] # (1)

        taus_x0 = ((x0 - ray_origins[...,0]) / ray_dirs[...,0]).unsqueeze(0)  # (N,1)
        taus_x1 = ((x1 - ray_origins[...,0]) / ray_dirs[...,0]).unsqueeze(0)  # (N,1)
        taus_y0 = ((y0 - ray_origins[...,1]) / ray_dirs[...,1]).unsqueeze(0)  # (N,1)
        taus_y1 = ((y1 - ray_origins[...,1]) / ray_dirs[...,1]).unsqueeze(0)  # (N,1)
        taus_z0 = ((z0 - ray_origins[...,2]) / ray_dirs[...,2]).unsqueeze(0)  # (N,1)
        taus_z1 = ((z1 - ray_origins[...,2]) / ray_dirs[...,2]).unsqueeze(0)  # (N,1)

        # concat and sort taus from least to greatest
        all_taus = torch.cat((taus_x0, taus_x1, taus_y0, taus_y1, taus_z0, taus_z1), dim=0)   # (6, N)
        all_taus = all_taus.transpose(1,0).contiguous()
        all_taus, _ = torch.sort(all_taus, dim=-1) #(N, 6)

        # ray_origins: (N, 3)
        # ray_dirs: (N, 3)
        points = ray_origins.unsqueeze(1).repeat((1,6,1)) + (all_taus.unsqueeze(-1).repeat((1,1,3)) * ray_dirs[...,:].unsqueeze(1).repeat((1,6,1))) # (N, 6, 3)

        # use eps on boundary check to avoid floating point issues
        eps = 1e-6
        points_x_in_bounds = torch.logical_and(points[...,0] >= x0 - eps, points[...,0] <= x1 + eps)
        points_y_in_bounds = torch.logical_and(points[...,1] >= y0 - eps, points[...,1] <= y1 + eps)
        points_z_in_bounds = torch.logical_and(points[...,2] >= z0 - eps, points[...,2] <= z1 + eps)

        points_in_bounds = torch.logical_and(torch.logical_and(points_x_in_bounds, points_y_in_bounds), points_z_in_bounds)
        _, min_sorted_points_indices = torch.sort(points_in_bounds.to(dtype=torch.int8), dim=-1, descending=True)
        _, max_sorted_points_indices = torch.sort(torch.flip(points_in_bounds, dims=[-1]).to(dtype=torch.int8), dim=-1, descending=True)
        min_indices = min_sorted_points_indices[...,0]
        max_indices = 5 - max_sorted_points_indices[...,0] # have to do (5 - x) operation due to the previous flipping


        # gotta do the unsqueeze/repeat in order to make sizes match
        min_points = torch.gather(input=points, dim=1, index=min_indices.unsqueeze(-1).unsqueeze(-1).repeat((1,1,3))).squeeze()
        max_points = torch.gather(input=points, dim=1, index=max_indices.unsqueeze(-1).unsqueeze(-1).repeat((1,1,3))).squeeze()
        min_taus = torch.gather(input=all_taus, dim=1, index=min_indices.unsqueeze(-1)).squeeze()
        max_taus = torch.gather(input=all_taus, dim=1, index=max_indices.unsqueeze(-1)).squeeze()

        return min_points, max_points, min_taus, max_taus



