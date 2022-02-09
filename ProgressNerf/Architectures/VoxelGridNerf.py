# dependency imports
import cv2 as cv
from datetime import datetime
from doctest import OutputChecker
import math
import numpy as np
from numpy.core.numeric import full
import os
from os.path import exists
from pyrsistent import b
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Dict
from re import L, S
import sys
import yaml

# internal imports
from ProgressNerf.Registries.ModelRegistry import get_model
from ProgressNerf.Registries.DataloaderRegistry import get_dataloader
from ProgressNerf.Registries.RaypickerRegistry import get_raypicker
from ProgressNerf.Registries.RaysamplerRegistry import get_raysampler
from ProgressNerf.Registries.EncoderRegistry import get_encoder
from ProgressNerf.Registries.RendererRegistry import get_renderer
from ProgressNerf.Registries.LossRegistry import get_loss
from ProgressNerf.Utils.CameraUtils import BuildCameraMatrix
from ProgressNerf.Utils.FolderUtils import last_epoch_from_output_dir
from ProgressNerf.Utils.VoxelGrid import VoxelGrid
from ProgressNerf.Utils.ParticleField import ParticleField
# import the supported arch dataloaders here
import ProgressNerf.Dataloading.ToolsPartsDataloader

# import the supported arch raypickers here
import ProgressNerf.Raycasting.RandomRaypicker
import ProgressNerf.Raycasting.WeightedRaypicker

# import the supported arch raysamplers here
import ProgressNerf.Raycasting.NearFarRaysampler
import ProgressNerf.Raycasting.WeightedRaysampler
import ProgressNerf.Raycasting.PerturbedRaysampler

# import the supported arch encoders here
import ProgressNerf.Encoders.PositionalEncoder

# import the supported arch models here
import ProgressNerf.Models.OGNerf

# import the supported arch renderers here
import ProgressNerf.NeuralRendering.NeuralRenderer
import ProgressNerf.NeuralRendering.VoxelNeuralRenderer

# import the supported arch loss functions here
import ProgressNerf.Losses.MSELoss
import ProgressNerf.Losses.MSE_DepthLoss
import ProgressNerf.Losses.BetaRegMSELoss

# this architecture represents the original NeRF paper as proposed by Mildenhall et al.
# see https://arxiv.org/abs/2003.08934 for further details
class VoxelGridNerf(object):
    def __init__(self, configFile:str) -> None:
        super().__init__()
        print("loading config at {0}".format(configFile))
        config = yaml.load(open(configFile,'r'), yaml.FullLoader)

        # pos enc levels = 10
        # dir enc levels = 4
        # parse the configuration & populate values
        print("parsing config")
        self.parseConfig(config)

        self.particle_field = ParticleField(self.cam_matrix.to(self.device), self.render_height, self.render_width)

        if(self.train_loader is not None):
            # if we are training, we init the NN models and the optimizer
            print("initializing optimizer")
            learning_params = list(self.nn.parameters())
            self.nn.to(self.device)
            if(self.nn_fine is not None):
                self.nn_fine.to(self.device)
                learning_params = learning_params + list(self.nn_fine.parameters())

            # setup optimizer and loss function
            self.optimizer = torch.optim.Adam(learning_params, lr=self.lr)

            if(self.start_epoch is not None):
                load_checkpoint_dir = ""
                if(self.start_epoch <= 0):
                    latest_epoch = last_epoch_from_output_dir(self.base_dir)
                    if(latest_epoch < 0):
                        print("no previous saved data found - starting training from epoch 0")
                        self.start_epoch = 0
                    else:
                        load_checkpoint_dir = os.path.join(self.base_dir, "epoch_{0}".format(latest_epoch))
                        self.start_epoch = latest_epoch + 1
                        print("loading model  & optimizer params from {0}".format(load_checkpoint_dir))
                        self.loadTrainState(load_checkpoint_dir)
                        print("resuming training from epoch {0}...".format(self.start_epoch))
                else:
                    load_checkpoint_dir= os.path.join(self.base_dir, "epoch_{0}".format(self.start_epoch))
                    self.start_epoch = self.start_epoch + 1
                    print("loading model  & optimizer params from {0}".format(load_checkpoint_dir))
                    self.loadTrainState(load_checkpoint_dir)
                    print("resuming training from epoch {0}...".format(self.start_epoch))
            else:
                print("starting training from epoch 0...")
                self.start_epoch = 0
        else:
            # otherwise, only init the NN models
            self.nn.to(self.device)
            if(self.nn_fine is not None):
                self.nn_fine.to(self.device)

            if(self.start_epoch is not None):
                load_checkpoint_dir = ""
                if(self.start_epoch <= 0):
                    latest_epoch = last_epoch_from_output_dir(self.base_dir)
                    load_checkpoint_dir = os.path.join(self.base_dir, "epoch_{0}".format(latest_epoch))
                    self.start_epoch = latest_epoch
                else:
                    load_checkpoint_dir= os.path.join(self.base_dir, "epoch_{0}".format(self.start_epoch))
                print("loading ONLY model params from {0}".format(load_checkpoint_dir))
                self.loadNNModels(load_checkpoint_dir)
            else:
                raise Exception("load_from_epoch must be defined to load models when in evaluation mode!")


    # extract this architecture's parameters from the configuration YAML file
    def parseConfig(self, config:Dict):
        self.base_dir = config['baseDir']

        self.testPeriod = config['testPeriod'] if 'testPeriod' in config.keys() else None
        self.savePeriod = config['save_period']  if 'save_period' in config.keys() else None

        self.train_loader = None
        self.test_loader = None
        if('train_dataloader' in config.keys()):
            train_loader_config = config['train_dataloader']
            self.train_loader = get_dataloader(train_loader_config['dataloader'])(train_loader_config[train_loader_config['dataloader']])
        if('test_dataloader' in config.keys()):
            test_loader_config = config['test_dataloader']
            self.test_loader = get_dataloader(test_loader_config['dataloader'])(test_loader_config[test_loader_config['dataloader']])

        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else None
        self.num_workers = config['num_workers'] if 'num_workers' in config.keys() else None
        self.device = config['device']

        self.cam_f_t = [config['camera_fx'],config['camera_fy'],config['camera_tx'],config['camera_ty']]
        self.cam_matrix = BuildCameraMatrix(*self.cam_f_t)

        self.render_width = config['render_resoluion'][0]
        self.render_height = config['render_resoluion'][1]
        self.raypicker = get_raypicker(config['raypicker'])(config[config['raypicker']])
        self.raypicker.setCameraParameters(self.cam_matrix, self.render_height, self.render_width)

        self.eval_subBatchSize = config['eval_subbatch_size']

        self.raysampler = get_raysampler(config['raysampler'])(config[config['raysampler']])

        self.train_epochs = config['train_epochs'] if 'train_epochs' in config.keys() else None
        self.lr = config['optim_lr'] if 'optim_lr' in config.keys() else None

        pos_enc_dict = config['posEncoder']
        self.pos_encoder = get_encoder(pos_enc_dict['encoder'])(pos_enc_dict[pos_enc_dict['encoder']])

        dir_enc_dict = config['dirEncoder']
        self.dir_encoder = get_encoder(dir_enc_dict['encoder'])(dir_enc_dict[dir_enc_dict['encoder']])

        # create a voxel grid that only stores 1 axes (e.g. is voxel occupied or not)
        axes_min_max = torch.Tensor([float(val) for val in config['boundary_min_maxes']]).reshape((2,3)).to(dtype=torch.float32, device=self.device)
        voxel_size = float(config['init_voxel_size'])
        self.voxel_grid = VoxelGrid(axes_min_max, voxel_size, 2)
        self.voxel_grid.voxels = self.voxel_grid.voxels + 1. # mark all voxels as occupied to start with
        # schedule for reducing the voxel grid size and for pruning as necessary
        self.voxel_halving_schedule = [int(val) for val in config['half_voxels_schedule']]
        self.prune_voxels_schedule = [int(val) for val in config['prune_voxels_schedule']]
        self.prune_voxels_sample_count = int(config['prune_voxels_sample_count'])
        self.prune_voxels_gamma = float(config['prune_voxels_gamma'])
        self.voxel_visit_after = int(config['voxel_visit_after_epoch'])
        self.voxel_visit_depth_stoppage = float(config['voxel_visit_depth_stoppage']) if 'voxel_visit_depth_stoppage' in config.keys() else None
        self.voxel_visit_factor = float(config['voxel_visit_factor'])

        self.build_weighting_cloud = config['build_weighting_cloud']
        self.weighting_cloud = None

        self.renderer = get_renderer(config['renderer'])(config[config['renderer']])

        train_loss_config = config['train_loss']
        self.train_loss = get_loss(train_loss_config['loss_fn'])(train_loss_config[train_loss_config['loss_fn']])
        test_loss_config = config['test_loss']
        self.test_loss = get_loss(test_loss_config['loss_fn'])(test_loss_config[test_loss_config['loss_fn']])

        coarse_config = config['coarse_model']
        self.nn = get_model(coarse_config['nnModel'])(coarse_config[coarse_config['nnModel']])

        if('fine_model' in config.keys()):
            fine_config = config['fine_model']

            self.nn_fine = get_model(fine_config['nnModel'])(fine_config[fine_config['nnModel']])
            self.raysampler_fine = get_raysampler(fine_config['raysampler'])(fine_config[fine_config['raysampler']])
        else:
            self.nn_fine = None
            self.raysampler_fine = None

        self.tool = config['desired_tool']
        self.masks = None
        if('apply_masks' in config.keys()):
            self.masks = config['apply_masks']

        self.tb_writer = None

        self.start_epoch = None
        if('load_from_epoch' in config.keys()):
            self.start_epoch = config['load_from_epoch']

    # helper function for saving the state of the training process
    def saveTrainState(self, checkpoint_dir:str = "latest"):
        output_dir = os.path.join(self.base_dir, checkpoint_dir)
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir)
        torch.save(self.nn.state_dict(), os.path.join(output_dir, "mlp_dict.ptr"))
        if(self.nn_fine is not None):
            torch.save(self.nn_fine.state_dict(), os.path.join(output_dir, "mlp_dict_fine.ptr"))
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer_dict.ptr"))
        self.voxel_grid.save(os.path.join(output_dir, "voxel_grid.ptr"))

    def loadNNModels(self, checkpoint_dir:str = "latest"):
        input_dir = os.path.join(self.base_dir, checkpoint_dir)
        self.nn.load_state_dict(torch.load(os.path.join(input_dir, "mlp_dict.ptr")))
        if(self.nn_fine is not None):
            self.nn_fine.load_state_dict(torch.load(os.path.join(input_dir,"mlp_dict_fine.ptr")))
        self.nn = self.nn.to(self.device)
        self.nn_fine = self.nn.to(self.device)

    def loadTrainState(self, checkpoint_dir:str = "latest"):
        input_dir = os.path.join(self.base_dir, checkpoint_dir)
        self.loadNNModels(checkpoint_dir)
        self.optimizer.load_state_dict(torch.load(os.path.join(input_dir, "optimizer_dict.ptr")))
        self.voxel_grid = VoxelGrid.load(os.path.join(checkpoint_dir, "voxel_grid.ptr"))
        self.voxel_grid.to(self.device)

    # performs the per-ray sampling and final rendering
    # this is where the raysampler is called, as well as the renderer
    # produces rgb and est depth outputs from the provided ray origins and dirs
    # ray_origins: (batch_size, num_rays, 3)
    # ray_dirs: (batch_size, num_rays, 3)
    def render(self, ray_origins, ray_dirs, mark_visited_voxels = False):
        # sampled_locations: (batch_size, num_rays, num_samples, 3)
        # sampled_distances: (batch_size, num_rays, num_samples)
        sampled_locations, sampled_distances = self.raysampler.sampleRays(ray_origins, ray_dirs)

        encoded_locs = self.pos_encoder.encodeFeature(sampled_locations)
        encoded_dirs = self.dir_encoder.encodeFeature(ray_dirs)
        encoded_dirs_coarse = encoded_dirs.unsqueeze(2).repeat(1,1,encoded_locs.shape[2],1)
        
        mlp_outputs = self.nn.forward(encoded_locs, encoded_dirs_coarse) #(batch_size, num_rays, num_samples, 4)
        rendered_output = self.renderer.renderRays(mlp_outputs, sampled_distances, voxels=self.voxel_grid, sample_locations=sampled_locations)

        # mark voxels as visited if necessary
        if(mark_visited_voxels):
            in_bounds_results = self.voxel_grid.are_voxels_xyz_in_bounds(sampled_locations) # (batch_size, num_rays, num_samples)
            rendered_depth = rendered_output['depth'] # (batch_size, num_rays)
            mark_visited = in_bounds_results
            if(self.voxel_visit_depth_stoppage is not None):
                is_past_depth = (sampled_distances[:,:,:] > (rendered_depth[:,:, None] + self.voxel_visit_depth_stoppage))
                mark_visited = torch.bitwise_and(mark_visited, is_past_depth)
            voxel_vals = self.voxel_grid.get_voxels_xyz(sampled_locations[mark_visited, :])
            voxel_vals[:, 1] = voxel_vals[:, 1] * self.voxel_visit_factor # update the 'non-visited' prior # TODO: make this factor configurable
            self.voxel_grid.set_voxels_xyz(sampled_locations[mark_visited, :], voxel_vals)

        if(self.nn_fine is not None):
            weighted_resampling_other_info = {'distances': sampled_distances, 'sigmas': mlp_outputs[:,:,:,3]}
            _, resampled_distances = self.raysampler_fine.sampleRays(ray_origins, ray_dirs, other_info=weighted_resampling_other_info)
            resampled_distances, _ = torch.cat((sampled_distances, resampled_distances), dim = -1).sort()
            resampled_locations = ray_origins[...,None,:] + ray_dirs[..., None, :] * resampled_distances[...,:,None]
            encoded_re_locs = self.pos_encoder.encodeFeature(resampled_locations)
            encoded_dirs_fine = encoded_dirs.unsqueeze(2).repeat(1,1,encoded_re_locs.shape[2],1)
            fine_mlp_outputs = self.nn_fine.forward(encoded_re_locs, encoded_dirs_fine)
            rendered_output_fine = self.renderer.renderRays(fine_mlp_outputs, resampled_distances, voxels=self.voxel_grid, sample_locations=resampled_locations)
            rendered_output['rgb'] = (rendered_output['rgb'] + rendered_output_fine['rgb']) / 2.0
            rendered_output['depth'] = (rendered_output['depth'] + rendered_output_fine['depth']) / 2.0

        return rendered_output

    def getSegementationWeighting(self, sample_batched):
        segmentation_img = sample_batched['segmentation'].to(self.device) # (batch_size, W, H)

        # create the ray weights tensor based on the provided segmentation tools/parts
        # note that this is not always used by all raypickers (e.g. RandomRaypicker ignores this input)
        ray_weights = torch.zeros_like(segmentation_img).to(self.device) #(batch_size, W, H)
        if(self.masks is not None):
            for mask_name in self.masks:
                mask_seg_label = sample_batched[mask_name + "_label"] #(batch_size)
                for batch_idx in range(mask_seg_label.shape[0]):
                    ray_weights[batch_idx] = torch.logical_or(ray_weights[batch_idx], segmentation_img[batch_idx] == mask_seg_label[batch_idx])

        return ray_weights

    def doVoxelPruning(self):
        self.nn.eval()
        if(self.nn_fine is not None):
            self.nn_fine.eval()
        with torch.no_grad():
            sizes = self.voxel_grid.shape.to(dtype=torch.int)
            bbox_bounds = self.voxel_grid.volume_bounds
            voxel_size = self.voxel_grid.voxelSize

            random_samples = torch.rand((self.prune_voxels_sample_count, 6)) # (sample_count, 6) -> final dim will be (x,y,z,view_x, view_y, view_z)
            random_samples[:, 0:3] = random_samples[:, 0:3] * voxel_size # scale x,y,z to fit inside a voxel
            random_samples[:, 3:] = random_samples[:, 3:] / torch.linalg.norm(random_samples[:, 3:], dim = 1).unsqueeze(-1).repeat(1,3) # norm all of the ray_dirs to be length 1

            random_samples = random_samples.to(self.device)
            offset = torch.Tensor([0.0,0.0,0.0]).to(self.device)

            total_voxels = sizes[0] * sizes[1] * sizes[2]
            pruned_voxels = 0
            pruned_no_visit = 0
            occ_voxels = 0
            min_vals = torch.zeros_like(self.voxel_grid.voxels)

            # TODO: make this vectorized
            for x_idx in tqdm(range(sizes[0]), leave = False):
                offset[0] = x_idx * voxel_size + bbox_bounds[0,0]
                for y_idx in tqdm(range(sizes[1]), leave = False):
                    offset[1] = y_idx * voxel_size + bbox_bounds[0,1]
                    for z_idx in tqdm(range(sizes[2]), leave = False):
                        # only test if not already pruned
                        if(self.voxel_grid[x_idx, y_idx, z_idx,0] > 0.0):
                            offset[2] = z_idx * voxel_size + bbox_bounds[0,2]
                            occ_voxels = occ_voxels + 1
                            # note - unsqueezes are just to make the nn.forward play nice
                            encoded_locs = self.pos_encoder.encodeFeature(random_samples[:, 0:3] + offset[None, :]).unsqueeze(0).unsqueeze(0) #(1, 1, prune_voxels_sample_count, N_feature_encodings)
                            encoded_dirs_coarse = self.dir_encoder.encodeFeature(random_samples[:, 3:]).unsqueeze(0).unsqueeze(0) #(1, 1, prune_voxels_sample_count, N_feature_encodings)
                            if(self.nn_fine is not None):
                                mlp_outputs = self.nn_fine.forward(encoded_locs, encoded_dirs_coarse) #(1, 1, prune_voxels_sample_count, 4)
                            else:
                                mlp_outputs = self.nn.forward(encoded_locs, encoded_dirs_coarse) #(1, 1, prune_voxels_sample_count, 4)
                            sigmas = mlp_outputs[:,:,:,3]
                            pruning_vals = torch.exp(-1 * sigmas)
                            min_val = torch.min(pruning_vals)
                            min_vals[x_idx, y_idx, z_idx] = min_val
                            # condition under which to prune voxels
                            if(min_val > self.prune_voxels_gamma):
                                self.voxel_grid[x_idx, y_idx, z_idx,0] = 0.0
                                pruned_voxels =  pruned_voxels + 1
                            # prune if we never visited this location
                            if(self.voxel_grid[x_idx, y_idx, z_idx,1] >= 0.9):
                                self.voxel_grid[x_idx, y_idx, z_idx,0] = 0.0
                                pruned_no_visit =  pruned_no_visit + 1

            
            tqdm.write("Done pruning voxels. Original voxel count: {0}, orig occupied voxel count: {1}, sigma pruned: {2}, no visit pruned: {3}".format(total_voxels, occ_voxels, pruned_voxels, pruned_no_visit))


    # performs the ray picking step and sends the results to the renderer
    # this is where the raypicker is called
    def doTrainRendering(self, sample_batched, mark_visited_voxels=True):
        # these are H,W ordere here, but get transposed in the render,etc. stages as necessary
        train_imgs = sample_batched['image'].to(self.device) # (batch_size, H, W, 3)
        train_depths = sample_batched['depth'].to(self.device) # (batch_size, H, W)

        # get the camera poses
        cam_poses = torch.linalg.inv(sample_batched['{0}_pose'.format(self.tool)]).to(self.device) # (batch_size, 4, 4)

        # create the ray weights tensor based on the provided segmentation tools/parts
        # note that this is not always used by all raypickers (e.g. RandomRaypicker ignores this input)
        points_segmentation = torch.bitwise_not(self.particle_field.getPointsSegmentation(cam_poses))
        train_imgs.transpose(1,2)[points_segmentation, :] *= 0.0
        train_imgs.transpose(1,2)[points_segmentation, :] += 1.0
        train_depths.transpose(1,2)[points_segmentation] = 100.0
        points_segmentation = points_segmentation.to(torch.float32) # (batch_size, W, H)

        ray_weights = self.getSegementationWeighting(sample_batched) # (batch_size, W, H)

        background_points = points_segmentation.sum(dim=(1,2))
        foreground_points = ray_weights.sum(dim=(1,2))

        ray_weights *= ((background_points / (foreground_points + background_points)) * 1).unsqueeze(-1).unsqueeze(-1).repeat((1, self.render_height, self.render_width))
        points_segmentation *= (foreground_points / (foreground_points + background_points)).unsqueeze(-1).unsqueeze(-1).repeat((1, self.render_width, self.render_height))

        ray_weights += points_segmentation.transpose(-1,-2)

        # run the raypicker & render for the object rays
        ray_origins, ray_dirs, ijs = self.raypicker.getRays(cam_poses, ray_weights = ray_weights)
        render_result = self.render(ray_origins, ray_dirs, mark_visited_voxels=mark_visited_voxels)

        # TODO: find out how to properly vectorize this indexing
        ijs_label = torch.zeros_like(train_imgs)
        # populate object render results
        train_pixels = torch.zeros_like(render_result['rgb'])
        train_pixel_depths = torch.zeros_like(render_result['depth'])
        for i in range(train_pixels.shape[0]):
            train_pixels[i] = train_imgs[i, ijs[i,:,1], ijs[i,:,0], :]
            train_pixel_depths[i] = train_depths[i, ijs[i,:,1], ijs[i,:,0]]
            ijs_label[i, ijs[i,:,1], ijs[i,:,0], :] = train_imgs[i, ijs[i,:,1], ijs[i,:,0], :]
            
        return render_result, train_pixels, train_pixel_depths.unsqueeze(-1)

    # performs a rendering of the full image
    # unlike doTrainRendering, the raypicker is not run (except to the the totality to rays created). Instead, we
    # extract sub-batches of all rays for rendering to prevent out-of-memory issues
    def doTestRendering(self, sample_batched):
        test_imgs = sample_batched['image'].to(self.device) # (batch_size, H, W, 3)
        test_depths = sample_batched['depth'].to(self.device).unsqueeze(-1) # (batch_size, H, W, 1)
        cam_poses = torch.linalg.inv(sample_batched['{0}_pose'.format(self.tool)]).to(self.device) # (batch_size, 4, 4)
        full_rendering = self.doFullRender(cam_poses)
        return full_rendering, test_imgs, test_depths

    # performs a rendering at an arbitrary camera pose for the entire image.
    # this method has no dependency on the underlying dataloader, so it can be called in applications
    # or other packages which are disconnected from the original training process
    # note that it is called in doTestRendering (for use during the training process), so some care is required
    # before making extensive modifications
    # camera_poses should be a Tensor of size (N, 4, 4) - comprised of N, 4x4 homogenous camera poses (in OpenCV/RH coords)
    def doEvalRendering(self, camera_poses:torch.Tensor):
        with torch.no_grad():
            rendering_output = self.doFullRender(camera_poses)
            rgb_output = rendering_output['rgb'].contiguous().reshape((camera_poses.shape[0], self.render_width, self.render_height, 3)).transpose(1,2).contiguous()
            return rgb_output

    def doFullRender(self, camera_poses:torch.Tensor):
        ray_origins, ray_dirs = self.raypicker.getAllRays(camera_poses.to(self.device)) # (batch_dim, width*height, 3), (batch_dim, width*height, 3)
        total_sample_size = ray_origins.shape[1]
        num_subBatches = math.ceil(total_sample_size / self.eval_subBatchSize)
        full_rendering = {}

        for subBatch_idx in range(num_subBatches):
            start_idx = subBatch_idx * self.eval_subBatchSize
            end_idx = min((subBatch_idx + 1) * self.eval_subBatchSize, total_sample_size)
            subBatch_ray_origins = ray_origins[:,start_idx:end_idx,:]
            subBatch_ray_dirs = ray_dirs[:,start_idx:end_idx,:]
            subBatch_rendering = self.render(subBatch_ray_origins, subBatch_ray_dirs)
            if('rgb' in full_rendering.keys()):
                full_rendering['rgb'] = torch.cat((full_rendering['rgb'], subBatch_rendering['rgb'].clone()), dim=1)
            else:
                full_rendering['rgb'] = subBatch_rendering['rgb'].clone()

            if('depth' in full_rendering.keys()):
                full_rendering['depth'] = torch.cat((full_rendering['depth'], subBatch_rendering['depth'].clone()), dim=1)
            else:
                full_rendering['depth'] = subBatch_rendering['depth'].clone()

        return full_rendering

    def populateParticleField(self):
        # verify the training conditions are met in the provided config
        if(self.train_loader is None):
            raise Exception("dataloaders for train and test sets are required for populating particle field!")

        # create dataloaders
        self.train_dataloader = DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        tqdm.write("Populating the particle field")

        for i_batch, sample_batched in tqdm(enumerate(self.train_dataloader), leave = True):
            depths = sample_batched['depth'].transpose(-1,-2).to(self.device) # (batch_size, W, H)
            cam_poses = torch.linalg.inv(sample_batched['{0}_pose'.format(self.tool)]).to(self.device) # (batch_size, 4, 4)
            center_ij = torch.matmul(self.cam_matrix.to(self.device), sample_batched['{0}_pose'.format(self.tool)][0,0:3,3].to(self.device))
            center_ij = (center_ij / center_ij[2]).to(torch.long)[0:2]
            self.particle_field.appendPoints(cam_poses, self.getSegementationWeighting(sample_batched).transpose(-1,-2), depths)
            self.particle_field.filterByVoxelGrid(self.voxel_grid)


    def train(self):

        # verify the training conditions are met in the provided config
        if(self.train_loader is None or self.test_loader is None):
            raise Exception("dataloaders for train and test sets are required for training!")

        # create the Tensorboard Writer
        self.tb_writer = SummaryWriter(log_dir=str(self.base_dir))

        # create train/test dataloaders
        self.train_dataloader = DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.test_loader, batch_size=1, shuffle=False, num_workers=self.num_workers)

        # set the main models to train, move the GPU if necessary, and extract the optimization parameters
        self.nn.train()
        self.nn.to(self.device)
        self.voxel_grid.to(self.device)
        if(self.nn_fine is not None):
            self.nn_fine.train()
            self.nn_fine.to(self.device)
            self.nn_fine.to(self.device)

        tqdm.write("Starting the main training loop")
        # main training loop
        for epoch in tqdm(range(self.start_epoch, self.train_epochs), leave = True):

            # if needed, half voxels
            if(epoch in self.voxel_halving_schedule):
                tqdm.write("Halving voxel sizes on epoch {0}".format(epoch))
                self.voxel_grid.subdivideGrid()

            # if needed, prune empty voxels
            if(epoch in self.prune_voxels_schedule):
                tqdm.write("Pruning voxel sizes on epoch {0}".format(epoch))
                self.doVoxelPruning()

            # Do training step
            losses = {}
            self.nn.train()

            outputString = "Epoch:{0}".format(epoch)
            if(self.nn_fine is not None):
                self.nn_fine.train()
            for i_batch, sample_batched in tqdm(enumerate(self.train_dataloader), leave = False):
                rendered_output, train_pixels, train_depths = self.doTrainRendering(sample_batched, epoch >= self.voxel_visit_after)
                losses_i = self.train_loss.calculateLoss(train_pixels, rendered_output['rgb'],\
                    rendered_depths=rendered_output["depth"].unsqueeze(-1),\
                    gt_depths=train_depths,\
                    alphas=rendered_output['alphas'])
                losses_i['loss'].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                for key in losses_i.keys():
                    if(key in losses.keys()):
                        losses[key].append(losses_i[key].item())
                    else:
                        losses[key] = list([losses_i[key].item()])
            
            # compute & log metrics
            for key in losses.keys():
                avg = np.mean(losses[key])
                self.tb_writer.add_scalar('train/{0}'.format(key), avg, epoch)
                outputString = outputString + ' train/{0}:{1:.4f}'.format(key,avg)

            self.optimizer.zero_grad()

            # if we need to do evaluation this epoch
            if(not(epoch == 0) and (epoch % self.testPeriod == 0)):
                # Do test/eval
                self.nn.eval()
                if(self.nn_fine is not None):
                    self.nn_fine.eval()
                # render the entire image instead of only the sampled pixels, compute loss & send to tensorboard
                losses_test = {}
                with torch.no_grad():
                    for i_batch, sample_batched in tqdm(enumerate(self.test_dataloader), leave = False):
                        rendered_output, test_images, test_depths = self.doTestRendering(sample_batched)
                        rgb_output = rendered_output['rgb'].reshape((1, self.render_width, self.render_height, 3)).transpose(1,2)
                        depth_output = rendered_output['depth'].reshape((1, self.render_width, self.render_height, 1)).transpose(1,2)
                        losses_i = self.test_loss.calculateLoss(test_images, rgb_output, rendered_depths=depth_output, gt_depths=test_depths)
                        for key in losses_i.keys():
                            if(key in losses_test.keys()):
                                losses_test[key].append(losses_i[key].item())
                            else:
                                losses_test[key] = list([losses_i[key].item()])
                        # swap dimensions since tensorboard expects shape (batches, channels, w, h)
                        # but test_images, etc. comes out as (batches, w, h, channels)
                        test_images = torch.transpose(test_images, 3, 1).transpose(2,3)
                        rgb_output = torch.transpose(rgb_output, 3, 1).transpose(2,3)
                        self.tb_writer.add_images("test/gtImage/{0}".format(i_batch), test_images, epoch)
                        self.tb_writer.add_images("test/rendered/{0}".format(i_batch), rgb_output, epoch)
                for key in losses_test.keys():
                    avg = np.mean(losses_test[key])
                    self.tb_writer.add_scalar('test/{0}'.format(key), avg, epoch)
                    outputString = outputString + ' test/{0}:{1:.4f}'.format(key,avg)
            tqdm.write(outputString)
            # output the arch's data on requisite epochs
            if(not(epoch == 0) and (epoch % self.savePeriod == 0)):
                self.saveTrainState("epoch_{0}".format(epoch))
        
        # final end-of-training outputs test_loss
        losses_test = {}
        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(self.test_dataloader), leave = True):
                rendered_output, test_images, test_depths = self.doTestRendering(sample_batched)
                rgb_output = rendered_output['rgb'].reshape((1, self.render_width, self.render_height, 3)).transpose(1,2)
                depth_output = rendered_output['depth'].reshape((1, self.render_width, self.render_height, 1)).transpose(1,2)
                losses = self.test_loss.calculateLoss(test_images, rgb_output, rendered_depths=depth_output, gt_depths=test_depths)
                for key in losses_i.keys():
                    if(key in losses_test.keys()):
                        losses_test[key].append(losses_i[key].item())
                    else:
                        losses_test[key] = list([losses_i[key].item()])
                # swap dimensions since tensorboard expects shape (batches, channels, w, h)
                # but test_images, etc. comes out as (batches, w, h, channels)
                test_images = torch.transpose(test_images, 3, 1).transpose(2,3)
                rgb_output = torch.transpose(rgb_output, 3, 1).transpose(2,3)
                self.tb_writer.add_images("test/gtImage/{0}".format(i_batch), test_images, self.train_epochs)
                self.tb_writer.add_images("test/rendered/{0}".format(i_batch), rgb_output, self.train_epochs)
        outputString = "Epoch:{0}".format(self.train_epochs)
        for key in losses_test.keys():
            avg = np.mean(losses_test[key])
            self.tb_writer.add_scalar('test/{0}'.format(key), avg, epoch)
            outputString = outputString + ' test/{0}:{1:.4f}'.format(key,avg)
        tqdm.write(outputString)
        self.saveTrainState("epoch_{0}".format(self.train_epochs))

if __name__=="__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    configFile = "./configs/VoxelGridNerf/toolPartsCoarseFinePerturbedMasked.yml"
    if(len(sys.argv) == 2):
        configFile = str(sys.argv[1])
    arch = VoxelGridNerf(configFile)
    arch.populateParticleField()
    arch.train()