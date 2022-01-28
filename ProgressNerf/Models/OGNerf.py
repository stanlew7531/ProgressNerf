from typing import Dict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from ProgressNerf.Registries.ModelRegistry import register_model

@register_model('OGNerf')
class OGNerf(nn.Module):

    """
    constructors - one from config Dict, other from raw params
    """
    def __init__(self, config:Dict):
        super(OGNerf, self).__init__()
        include_orig_pos_in = config['includeOrigPos'] if 'includeOrigPos' in config.keys() else False
        include_orig_dir_in = config['includeOrigDir'] if 'includeOrigDir' in config.keys() else False

        if(include_orig_pos_in):
            pos_in_dims = (config['pos_enc_levels'] * 2 + 1) * 3 # (L * 2) + 1 for sin/cos functions while including original values
        else:
            pos_in_dims = config['pos_enc_levels'] * 2 * 3 # (L * 2) for sin/cos functions
            
        if(include_orig_dir_in):
            dir_in_dims = (config['dir_enc_levels'] * 2 + 1) * 3 # (L * 2) + 1 for sin/cos functions while including original values
        else:
            dir_in_dims = config['dir_enc_levels'] * 2 * 3 # (L * 2) for sin/cos functions

        self.initialize(pos_in_dims, dir_in_dims, config['hidden_dims'])
    
    def initialize(self, pos_in_dims, dir_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(OGNerf, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_rgb = nn.Linear(D//2, 3)

        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_rgb.bias.data = torch.tensor([0.02, 0.02, 0.02]).float()


    def forward(self, pos_enc, dir_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=3)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        density = self.fc_density(x)

        # add regularization noise as per Mildenhall et al. (training details section)
        if(self.training):
            density_reg = torch.randn_like(density)
            density = density + density_reg

        density = density.relu()  # (H, W, N_sample, 1)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)  # (H, W, N_sample, 3)

        rgb_den = torch.cat([rgb, density], dim=3)  # (H, W, N_sample, 4)
        return rgb_den