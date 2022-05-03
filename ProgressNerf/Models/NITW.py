from email.mime import base
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from ProgressNerf.Registries.ModelRegistry import register_model

@register_model('NITW')
class NITW(nn.Module):

    """
    constructors - one from config Dict, other from raw params
    """
    def __init__(self, config:Dict):
        super(NITW, self).__init__()
        base_layers = config['base_layers']
        static_layers = config['static_layers']
        transient_layers = config['transient_layers']
        base_hidden_units = config['base_hidden_units']
        static_hidden_units = config['static_hidden_units']
        transient_hidden_units = config['transient_hidden_units']

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

        appearence_embedding_dim = config['appear_embed_dim']
        transient_embedding_dim = config['transient_embed_dim']


        base_params = (base_layers, base_hidden_units)
        static_params = (static_layers, static_hidden_units)
        transient_params = (transient_layers, transient_hidden_units)

        self.initialize(pos_in_dims, dir_in_dims, base_params, static_params, transient_params, appearence_embedding_dim, transient_embedding_dim)
    
    def initialize(self, pos_in_dims, dir_in_dims, base_params, static_params, transient_params, appear_emb_d, trans_embed_d):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param base_params: tuple, 2 tuple of # of base MLP layers, hidden dimensions
        :param static_params: tuple, 2 tuple of # of static MLP layers, hidden dimensions
        :param transient_params: tuple, 2 tuple of # of transient MLP layers, hidden dimensions
        :param appear_emb_d: scalar, dimensionality of appearance embedding
        :param trans_embed_d: scalar, dimensionality of transient embedding
        """
        super(NITW, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims

        base_layer_count = base_params[0]
        base_hidden_units = base_params[1]
        base_modules = [nn.Linear(pos_in_dims, base_hidden_units), nn.ReLU]
        for _ in range(base_layer_count - 1):
            base_modules.append(nn.Linear(base_hidden_units, base_hidden_units))
            base_modules.append(nn.ReLU())
        
        static_layer_count = static_params[0]
        static_hidden_units = static_params[1]
        static_modules = [nn.Linear(appear_emb_d + dir_in_dims + base_hidden_units, static_hidden_units), nn.ReLU]
        for _ in range(static_layer_count - 1):
            static_modules.append(nn.Linear(static_hidden_units, static_hidden_units))
            static_modules.append(nn.ReLU())
        
        transient_layer_count = transient_params[0]
        transient_hidden_units = transient_params[1]
        transient_modules = [nn.Linear(trans_embed_d + base_hidden_units, transient_hidden_units), nn.ReLU]
        for _ in range(transient_layer_count - 1):
            transient_modules.append(nn.Linear(transient_hidden_units, transient_hidden_units))
            transient_modules.append(nn.ReLU())

        self.base_layers = nn.Sequential(*base_modules)
        self.static_layers = nn.Sequential(*static_modules)
        self.transient_layers = nn.Sequential(*transient_modules)

        self.fc_rgb_static = nn.Linear(static_hidden_units, 3)
        self.fc_rgb_transient = nn.Linear(transient_hidden_units, 3)

        self.fc_beta = nn.Linear(transient_hidden_units, 1)
        self.fc_density_static = nn.Linear(base_hidden_units, 1)
        self.fc_density_transient = nn.Linear(transient_hidden_units, 1)


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
        rgb = self.fc_rgb(x).sigmoid()  # (H, W, N_sample, 3)

        rgb_den = torch.cat([rgb, density], dim=3)  # (H, W, N_sample, 4)
        return rgb_den