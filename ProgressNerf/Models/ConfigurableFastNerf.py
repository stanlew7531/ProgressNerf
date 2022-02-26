from cgi import test
from re import X
from turtle import forward, pos
from typing import Dict
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import math
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as R
from zmq import device
from ProgressNerf.Registries.ModelRegistry import register_model
from ProgressNerf.Utils.VoxelGrid import VoxelGrid

@register_model('config_fast_nerf')
class ConfigurableFastNerf(nn.Module):

    """
    constructors - one from config Dict, other from raw params
    """
    def __init__(self, config:Dict):
        super(ConfigurableFastNerf, self).__init__()
        include_orig_pos_in = config['includeOrigPos'] if 'includeOrigPos' in config.keys() else False
        include_orig_dir_in = config['includeOrigDir'] if 'includeOrigDir' in config.keys() else False
        include_orig_conf_in = config['includeOrigConf'] if 'includeOrigConf' in config.keys() else False
        conf_vector_d = int(config['configDegOfFreedom'])

        if(include_orig_pos_in):
            pos_in_dims = (config['pos_enc_levels'] * 2 + 1) * 3 # (L * 2) + 1 for sin/cos functions while including original values
        else:
            pos_in_dims = config['pos_enc_levels'] * 2 * 3 # (L * 2) for sin/cos functions
            
        if(include_orig_dir_in):
            dir_in_dims = (config['dir_enc_levels'] * 2 + 1) * 3 # (L * 2) + 1 for sin/cos functions while including original values
        else:
            dir_in_dims = config['dir_enc_levels'] * 2 * 3 # (L * 2) for sin/cos functions

        if(include_orig_conf_in):
            self.conf_in_dims = (config['conf_enc_levels'] * 2 + 1) * conf_vector_d # (L * 2) + 1 for sin/cos functions while including original values
        else:
            self.conf_in_dims = config['conf_enc_levels'] * 2 * conf_vector_d # (L * 2) for sin/cos functions


        pos_hidden = config['hidden_units_pos']
        dir_hidden = config['hidden_units_dir']
        conf_hidden = config['hidden_units_conf']
        pos_layers = config['layers_pos']
        dir_layers = config['layers_dir']
        conf_layers = config['layers_conf']
        self.D = config['D']

        self.initialize(pos_in_dims, dir_in_dims, self.conf_in_dims, pos_hidden, dir_hidden, conf_hidden, pos_layers, dir_layers, conf_layers, self.D)
    
    def initialize(self, pos_in_dims, dir_in_dims, conf_in_dims, hidden_units_pos, hidden_units_dir, hidden_units_conf, layers_pos, layers_dir, layers_conf, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param conf_in_dims: scalar, number of channels of encoded configuration vector
        :param hidden_units_pos:  scalar, number of hidden layer units for pos
        :param hidden_units_dir:  scalar, number of hidden layer units for dir
        :param hidden_units_conf:  scalar, number of hidden layer units for configuration
        :param layers_pos:  scalar, number of hidden layers for pos
        :param layers_dir:  scalar, number of hidden layers for dir
        :param layers_conf:  scalar, number of hidden layers for configuration
        :param D:           scalar, number of hidden dimensions for the final dot product
        """
        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims
        self.D = D

        # layers_pos_modules = []
        # layers_pos_modules.append(nn.Linear(pos_in_dims, hidden_units_pos))
        # layers_pos_modules.append(nn.ReLU())
        # for _ in range(layers_pos):
        #     layers_pos_modules.append(nn.Linear(hidden_units_pos, hidden_units_pos))
        #     layers_pos_modules.append(nn.ReLU())
        
        # final_pos_layer = nn.Linear(hidden_units_pos, (D * 3) + 1)
        # final_pos_layer.bias.data = torch.cat((torch.tensor([0.02]).float().repeat(3*D), torch.tensor([0.1]).float()))
        # layers_pos_modules.append(final_pos_layer)
        # layers_pos_modules.append(nn.ReLU())
        # self.layers_pos = nn.Sequential(*layers_pos_modules)

        self.layers_pos0 = nn.Sequential(
            nn.Linear(pos_in_dims + conf_in_dims, hidden_units_pos), nn.ReLU(),
            nn.Linear(hidden_units_pos, hidden_units_pos), nn.ReLU(),
            nn.Linear(hidden_units_pos, hidden_units_pos), nn.ReLU(),
            nn.Linear(hidden_units_pos, hidden_units_pos), nn.ReLU(),
        # self.layers_conf0 = nn.Sequential(
        #     nn.Linear(pos_in_dims + conf_in_dims, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        # )

        # self.layers_conf1 = nn.Sequential(
        #     nn.Linear(hidden_units_conf + pos_in_dims + conf_in_dims, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, 3),
        # )
        )

        self.layers_pos1 = nn.Sequential(
            nn.Linear(hidden_units_pos + pos_in_dims + conf_in_dims, hidden_units_pos), nn.ReLU(),
            nn.Linear(hidden_units_pos, hidden_units_pos), nn.ReLU(),
            nn.Linear(hidden_units_pos, hidden_units_pos), nn.ReLU(),
            nn.Linear(hidden_units_pos, hidden_units_pos), nn.ReLU(),
            nn.Linear(hidden_units_pos, (D * 3) + 1),
        )

        # self.layers_conf0 = nn.Sequential(
        #     nn.Linear(pos_in_dims + conf_in_dims, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        # )

        # self.layers_conf1 = nn.Sequential(
        #     nn.Linear(hidden_units_conf + pos_in_dims + conf_in_dims, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, hidden_units_conf),
        #     nn.Linear(hidden_units_conf, 3),
        # )

        layers_dir_modules = []
        layers_dir_modules.append(nn.Linear(dir_in_dims, hidden_units_dir))
        layers_dir_modules.append(nn.ReLU())
        for _ in range(layers_dir):
            layers_dir_modules.append(nn.Linear(hidden_units_dir, hidden_units_dir))
            layers_dir_modules.append(nn.ReLU())
        final_dir_layer = nn.Linear(hidden_units_dir, D)
        final_dir_layer.bias.data = torch.tensor([0.1]).float().repeat(D)
        layers_dir_modules.append(final_dir_layer)
        layers_dir_modules.append(nn.ReLU())
        self.layers_dir = nn.Sequential(*layers_dir_modules)

        self.sigma_layer = nn.Linear(self.D, 1)

    # def forward_conf_def(self, pos_enc, conf_enc):
    #     # concat position & configuration data
    #     x = torch.cat([pos_enc, conf_enc], dim=-1) # (H, W, N_sample, pos_in_dims + conf_in_dims)

    #     # run first stage
    #     x_prime = self.layers_conf0(x) # (H, W, N_sample, hidden_units_conf)

    #     # run second skip stage
    #     x_prime = self.layers_conf1(torch.cat([x_prime, x], dim=-1)) # (H, W, N_sample, 3)

    #     return x_prime

    def forward_nn(self, pos_enc, dir_enc, only_uvws=False, only_betas=False):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims + conf_in_dims) encoded positions/configurations
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        # validate inputs
        assert(not(only_betas and only_uvws))
        assert(not(only_uvws and (pos_enc == None)))
        assert(not(only_betas and (dir_enc == None)))

        x = None
        if(not only_betas):
            x = self.layers_pos0(pos_enc)  # (H, W, N_sample, hidden_units_pos)
            x = self.layers_pos1(torch.cat([pos_enc, x], dim=-1)) # (H, W, N_sample, (D * 3 + 1)
        if(only_uvws):
            return x

        betas = self.layers_dir(dir_enc) # (H, W, N_sample, D)
        if only_betas:
            return betas

        uvw = x[:,:,:,:-1] # (H, W, N_sample, D * 3)
        density = x[:,:,:,-1:]  # (H, W, N_sample, 1)

        og_shape = uvw.shape
        uvw = uvw.view(og_shape[0], og_shape[1], og_shape[2], 3, self.D)# (H, W, N_sample, 3, D)
        betas = betas.view(og_shape[0], og_shape[1], og_shape[2], 1, self.D) # (H, W, N_sample, 1, D)

        rgb = (uvw * betas).sum(dim=-1).sigmoid() # (H, W, N_sample, 3)

        # add regularization noise as per Mildenhall et al. (training details section)
        if(self.training):
            density_reg = torch.randn_like(density)
            density = density + density_reg
            
        rgb_den = torch.cat([rgb, density.relu()], dim=-1)  # (H, W, N_sample, 4)

        return rgb_den

    def forward_cache(self, pos, dir, uvws_cache:VoxelGrid, beta_cache:torch.tensor, config_cache_meta = {}):
        """
        :param pos_enc: (H, W, N_sample, 3 + n_conf_dims) xyz positions & configuration vector
        :param dir_enc: (H, W, N_sample, 3) xyz directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        # its 0130 when I wrote this - no judging allowed
        conf = pos[...,3:].flatten(end_dim=-2) # (N, n_art_dims)
        pos = pos[...,:3]
        in_bounds_results = uvws_cache.are_voxels_xyz_in_bounds(pos.view(-1, 3)) # (N)
        uvws_temp = uvws_cache.get_voxels_xyz(pos.view(-1, 3)[in_bounds_results])
        uvws = torch.zeros((in_bounds_results.shape[0], uvws_temp.shape[-1]), device = uvws_temp.device, dtype=torch.float32) # (N, config_D ** num_art_dims * (D*3 + 1))
        uvws[in_bounds_results] = uvws_temp

        num_config_dims = config_cache_meta['num_config_dims']
        config_cache_D = config_cache_meta['config_cache_D']
        config_bounds = config_cache_meta['config_bounds'] # (2, num_config_dims)
        config_widths = config_bounds[1] - config_bounds[0]
        config_steps = config_widths / config_cache_D

        view_dims = [-1]
        for _ in range(num_config_dims):
            view_dims.append(config_cache_D )
        view_dims.append(-1)
        conf_idx = torch.floor((conf - config_bounds[0])/config_steps).squeeze().to(torch.long)
        
        ogShape = uvws.shape
        dir = dir.unsqueeze(-2).repeat(1,1,pos.shape[-2],1)
        l = beta_cache.shape[0]
        thetas = torch.atan2(dir[...,1], dir[...,0]) # arctan(y/x)
        phis = torch.acos(dir[...,2]) # arccos(z/1) assume dirs to be normed already
        idx_i = torch.clamp(((thetas.view(-1,1) + math.pi) * (l / (2*math.pi))).to(dtype=torch.long), min = 0, max = l-1)
        idx_j = torch.clamp(((phis.view(-1,1) + math.pi) * (l / (2*math.pi))).to(dtype=torch.long), min = 0, max = l-1)
        idx_betas = torch.cat((idx_i,idx_j), dim = -1)
        betas = beta_cache[idx_betas[...,0],idx_betas[...,1]].view(ogShape[0], 1, -1) # (N, 1, D)
        view_dims = [ogShape[0]]
        for _ in range(num_config_dims):
            view_dims.append(config_cache_D )
        view_dims.append(-1)
        reshaped = uvws.view(*view_dims)[torch.arange(0,conf_idx.shape[0],dtype=torch.long, device=conf_idx.device), conf_idx]
        uvw = reshaped[...,:-1].view(ogShape[0], 3, int((reshaped.shape[-1]-1)/3)) # (N, 3, D)
        s = uvws[...,-1:] # (N, 1)
        rgb = (uvw * betas).sum(dim=-1).sigmoid()
        rgb_den = torch.cat([rgb,s.relu()], dim=-1)
        inPosShape = pos.shape
        return rgb_den.reshape(inPosShape[0],inPosShape[1],inPosShape[2],4)

    def forward(self, pos, dir,\
         only_uvws=False, only_betas=False, use_cache=False,\
         uvws_cache:VoxelGrid=None, beta_cache:torch.tensor=None,\
            config_cache_meta={}):
        if(not use_cache):
            return self.forward_nn(pos, dir, only_uvws, only_betas)
        else:
            return self.forward_cache(pos, dir, uvws_cache, beta_cache, config_cache_meta)

    def populate_grid(self, uvws_cache:VoxelGrid, beta_cache:torch.Tensor, pos_encoder, dir_encoder, config_encoder, config_bounds, config_D):
        with torch.no_grad():
            # populate the uvws cache
            bounds = uvws_cache.volume_bounds
            xs = torch.linspace(bounds[0,0],bounds[1,0], int(uvws_cache.shape[0].item()), dtype=torch.float32)
            ys = torch.linspace(bounds[0,1],bounds[1,1], int(uvws_cache.shape[1].item()), dtype=torch.float32)
            zs = torch.linspace(bounds[0,2],bounds[1,2], int(uvws_cache.shape[2].item()), dtype=torch.float32)
            configs = torch.cartesian_prod(*[torch.linspace(config_bounds[0,i], config_bounds[1,i], config_D, dtype=torch.float32) for i in range(config_bounds.shape[-1])]) # (config_D ** num_art_dims, num_art_dims)
            if(config_bounds.shape[-1] == 1):
                configs = configs.unsqueeze(-1)
            configs = config_encoder.encodeFeature(configs).to(beta_cache.device) # (config_D ** num_art_dims , 2 * 3 * config_encode_L)
            x_size = xs[1] - xs[0]
            y_size = ys[1] - ys[0]
            z_size = zs[1] - zs[0]
            test_points = torch.rand((100,3), device = beta_cache.device)
            test_points[:,0] *= x_size
            test_points[:,1] *= y_size
            test_points[:,2] *= z_size
            # TODO: vectorize when Im not feeling lazy
            for x in tqdm(xs,leave=False):
                for y in tqdm(ys,leave=False):
                    for z in tqdm(zs,leave=False):

                        inputs = configs.unsqueeze(1).repeat((1,100,1))# (config_D ** num_art_dims, 100, 2 * 3 * config_encode_L)

                        position = torch.Tensor([[x,y,z]]).to(beta_cache.device)
                        positions = test_points[:,:] + position[0,:] # (100, 3)
                        positions = pos_encoder.encodeFeature(positions) # (100, 2 * 3 * pos_encode_L)
                        positions = positions.unsqueeze(0).repeat(inputs.shape[0],1,1) # (config_D ** num_art_dims, 100, 2 * 3 * pos_encode_L)
                        inputs = torch.cat((inputs, positions), dim = -1) # (config_D * num_art_dims, 100, (2 * 3 * config_encode_L) + (2 * 3 * pos_encode_L))
                        uvws_result = self.forward(inputs, None, only_uvws=True) #(config_D * num_art_dims, 100, (cache_D * 3 + 1))
                        _, max_s_idx = torch.max(uvws_result[:, :, -1:], dim=-2)
                        to_add = torch.gather(uvws_result, 1, max_s_idx.unsqueeze(-1).repeat(1,1,uvws_result.shape[-1])) # (config_D ** num_art_dims, (cache_D * 3 + 1))
                        uvws_cache.set_voxels_xyz(position, to_add.flatten())

            # populate the betas cache
            l = beta_cache.shape[0]
            radian_samples = torch.linspace(-1 * math.pi, math.pi, l)
            point = torch.zeros((1,4))
            point[0,3] = 1 # set cam dir looking straight ahead (down the z axis) initially
            # TODO: vectorize when Im not feeling lazy
            for i in tqdm(range(l), leave=False):
                for j in tqdm(range(l), leave=False):
                    sphere_rotation = torch.eye(4)
                    elevation = radian_samples[i]
                    azimuth = radian_samples[j]
                    sphere_rotation[0:3,0:3] = torch.from_numpy(R.from_euler('xyx',[elevation, azimuth, 0]).as_matrix())
                    new_point = torch.matmul(sphere_rotation, point.transpose(0,1)).transpose(0,1).contiguous().to(beta_cache.device)
                    encoded = dir_encoder.encodeFeature(new_point[0,0:3])
                    beta_result = self.forward(None, encoded.view(1,1,1,-1), only_betas=True)
                    beta_cache[i,j] = beta_result