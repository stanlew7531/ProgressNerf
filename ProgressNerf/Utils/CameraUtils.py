from cv2 import rotate
import torch
import numpy as np

def BuildCameraMatrix(fx, fy, tx, ty):
    toRet = torch.eye(3)
    toRet[0,0] = fx
    toRet[1,1] = fy
    toRet[0,2] = tx
    toRet[1,2] = ty
    return toRet

# cam_matrix: (3, 3)
# camera_tfs: (batch_dim, 4, 4)
# depth: (batch_dim, width, height)
def CamPoseDepthToXYZ(cam_matrix:torch.Tensor, camera_tfs:torch.Tensor, depth:torch.Tensor):
    height = depth.shape[-1]
    width = depth.shape[-2]
    batch_dim = depth.shape[0]
    rays = ComputeCameraEpipolars(cam_matrix, height, width).repeat((batch_dim,1,1,1)) # (batch_dim, W, H, 3) 
    depth_scales = depth.unsqueeze(-1).repeat((1,1,1,3)) # (batch_dim, W, H, 3)
    points_camera_frame = rays * depth_scales # (batch_dim, W, H, 3)
    points_cf_homo = torch.cat((points_camera_frame, torch.ones((batch_dim, width, height, 1), device=cam_matrix.device)), dim=-1)

    points_world_frame = torch.matmul(camera_tfs.unsqueeze(1), points_cf_homo.reshape((batch_dim, width * height, 4, 1))).reshape((batch_dim, width, height, 4))[:,:,:,0:3]
    return points_world_frame


def ComputeCameraEpipolars(camera_matrix, height, width):
        """Compute base ray directions in camera coordinates, which only depends on intrinsics.
        These will be further converted to world coordinates later, using camera poses.
        :return: (W, H, 3) torch.float32
        """
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32, device=camera_matrix.device),
                            torch.arange(width, dtype=torch.float32, device=camera_matrix.device))  # (H, W)

        # Use OpenCV coordinate in 3D:
        #   +x points to right
        #   +y points to down
        #   +z points to forward
        #
        # The coordinate of the top left corner of an image should be (-0.5W, -0.5H, 1.0), then normed to be magnitude 1.0
        dirs_x = (x - 0.5*width) / camera_matrix[0,0]  # (H, W)
        dirs_y = (y - 0.5*height) / camera_matrix[1,1]  # (H, W)
        dirs_z = torch.ones((height, width), dtype=torch.float32, device=camera_matrix.device)  # (H, W)
        rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
        rays_dir = rays_dir.transpose(0,1).contiguous() # put into row-major format
        rays_dir = torch.reshape(rays_dir, (height * width, 3)) # (H*W, 3)
        norms = torch.linalg.norm(rays_dir, dim=1, keepdim=True) # (H*W, 1)
        rays_dir = rays_dir / norms # (H*W, 3)
        rays_dir = torch.reshape(rays_dir, (width, height, 3)).contiguous() # (W, H, 3)
        return rays_dir