# dependency imports
import cv2 as cv
from datetime import datetime
from doctest import OutputChecker
import math
import numpy as np
from numpy.core.numeric import full
import os
from os.path import exists
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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
from ProgressNerf.Utils.VoxelGrid import VoxelGrid

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
                        self.start_epoch = latest_epoch
                        print("loading model  & optimizer params from {0}".format(load_checkpoint_dir))
                        self.loadTrainState(load_checkpoint_dir)
                        print("resuming training from epoch {0}...".format(self.start_epoch))
                else:
                    load_checkpoint_dir= os.path.join(self.base_dir, "epoch_{0}".format(self.start_epoch))
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
        axes_min_max = torch.Tensor([float(val) for val in config['boundary_min_maxes']]).reshape((2,3)).to(dtype=torch.float64, device=self.device)
        voxel_size = float(config['init_voxel_size'])
        self.voxel_grid = VoxelGrid(axes_min_max, voxel_size, 1)
        self.voxel_grid.voxels = self.voxel_grid.voxels + 1. # mark all voxels as occupied to start with

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

    def loadNNModels(self, checkpoint_dir:str = "latest"):
        input_dir = os.path.join(self.base_dir, checkpoint_dir)
        self.nn.load_state_dict(torch.load(os.path.join(input_dir, "mlp_dict.ptr")))
        if(self.nn_fine is not None):
            self.nn_fine.load_state_dict(torch.load(os.path.join(input_dir,"mlp_dict_fine.ptr")))

    def loadTrainState(self, checkpoint_dir:str = "latest"):
        input_dir = os.path.join(self.base_dir, checkpoint_dir)
        self.loadNNModels(checkpoint_dir)
        self.optimizer.load_state_dict(torch.load(os.path.join(input_dir, "optimizer_dict.ptr")))

    # performs the per-ray sampling and final rendering
    # this is where the raysampler is called, as well as the renderer
    # produces rgb and est depth outputs from the provided ray origins and dirs
    # ray_origins: (batch_size, num_rays, 3)
    # ray_dirs: (batch_size, num_rays, 3)
    def render(self, ray_origins, ray_dirs):
        # sampled_locations: (batch_size, num_rays, num_samples, 3)
        # sampled_distances: (batch_size, num_rays, num_samples)
        sampled_locations, sampled_distances = self.raysampler.sampleRays(ray_origins, ray_dirs)

        encoded_locs = self.pos_encoder.encodeFeature(sampled_locations)
        encoded_dirs = self.dir_encoder.encodeFeature(ray_dirs)
        encoded_dirs_coarse = encoded_dirs.unsqueeze(2).repeat(1,1,encoded_locs.shape[2],1)
        
        mlp_outputs = self.nn.forward(encoded_locs, encoded_dirs_coarse) #(batch_size, num_rays, num_samples, 4)
        rendered_output = self.renderer.renderRays(mlp_outputs, sampled_distances, voxels=self.voxel_grid, sample_locations=sampled_locations)

        if(self.nn_fine is not None):
            weighted_resampling_other_info = {'distances': sampled_distances, 'sigmas': mlp_outputs[:,:,:,3].relu()}
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

    # performs the ray picking step and sends the results to the renderer
    # this is where the raypicker is called
    def doTrainRendering(self, sample_batched):
        train_imgs = sample_batched['image'].to(self.device) # (batch_size, W, H, 3)
        train_depths = sample_batched['depth'].to(self.device) # (batch_size, W, H)
        # create the ray weights tensor based on the provided segmentation tools/parts
        # note that this is not always used by all raypickers (e.g. RandomRaypicker ignores this input)
        ray_weights = self.getSegementationWeighting(sample_batched)
        # get the camera poses & run the raypicker
        cam_poses = torch.linalg.inv(sample_batched['{0}_pose'.format(self.tool)]).to(self.device) # (batch_size, 4, 4)
        ray_origins, ray_dirs, ijs = self.raypicker.getRays(cam_poses, ray_weights = ray_weights)
        ijs_label = torch.zeros_like(train_imgs)
        render_result = self.render(ray_origins, ray_dirs)
        # TODO: find out how to properly vectorize this indexing
        train_pixels = torch.zeros_like(render_result['rgb'])
        for i in range(train_pixels.shape[0]):
            train_pixels[i] = train_imgs[i, ijs[i,:,1], ijs[i,:,0], :]
            ijs_label[i, ijs[i,:,1], ijs[i,:,0], :] = train_imgs[i, ijs[i,:,1], ijs[i,:,0], :]

        return render_result, train_pixels

    # performs a rendering of the full image
    # unlike doTrainRendering, the raypicker is not run (except to the the totality to rays created). Instead, we
    # extract sub-batches of all rays for rendering to prevent out-of-memory issues
    def doTestRendering(self, sample_batched):
        test_imgs = sample_batched['image'].to(self.device) # (batch_size, W, H, 3)
        cam_poses = torch.linalg.inv(sample_batched['{0}_pose'.format(self.tool)]).to(self.device) # (batch_size, 4, 4)
        full_rendering = self.doFullRender(cam_poses)
        return full_rendering, test_imgs

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

        return full_rendering

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
        if(self.nn_fine is not None):
            self.nn_fine.train()
            self.nn_fine.to(self.device)

        # main training loop
        for epoch in tqdm(range(self.start_epoch, self.train_epochs)):
            losses = []
            losses_test = []
            # Do training step
            self.nn.train()
            if(self.nn_fine is not None):
                self.nn_fine.train()
            for i_batch, sample_batched in tqdm(enumerate(self.train_dataloader)):
                rendered_output, train_pixels = self.doTrainRendering(sample_batched)
                losses_i = self.train_loss.calculateLoss(train_pixels, rendered_output['rgb'])
                losses_i['loss'].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(losses_i['loss'].item())
            
            # compute & log metrics
            avg_loss = np.mean(losses)
            self.tb_writer.add_scalar('train/mse', avg_loss, epoch)
            self.optimizer.zero_grad()

            # if we need to do evaluation this epoch
            if(not(epoch == 0) and (epoch % self.testPeriod == 0)):
                # Do test/eval
                self.nn.eval()
                if(self.nn_fine is not None):
                    self.nn_fine.eval()
                # render the entire image instead of only the sampled pixels, compute loss & send to tensorboard
                with torch.no_grad():
                    for i_batch, sample_batched in tqdm(enumerate(self.test_dataloader)):
                        rendered_output, test_images = self.doTestRendering(sample_batched)
                        rgb_output = rendered_output['rgb'].contiguous().reshape((1, self.render_width, self.render_height, 3)).transpose(1,2).contiguous()
                        losses = self.test_loss.calculateLoss(test_images, rgb_output)
                        losses_test.append(losses['loss'].item())
                        # swap dimensions since tensorboard expects shape (batches, channels, w, h)
                        # but test_images, etc. comes out as (batches, w, h, channels)
                        test_images = torch.transpose(test_images, 3, 1).transpose(2,3).contiguous()
                        rgb_output = torch.transpose(rgb_output, 3, 1).transpose(2,3).contiguous()
                        self.tb_writer.add_images("test/gtImage/{0}".format(i_batch), test_images, epoch)
                        self.tb_writer.add_images("test/rendered/{0}".format(i_batch), rgb_output, epoch)
                avg_loss_test = np.mean(losses_test)
                self.tb_writer.add_scalar('test/mse', avg_loss_test, epoch)
                tqdm.write("Epoch: {0}, Avg RGB MSE loss: {1:.4f}, Test Avg RGB MSE loss: {2:.4f}".format(epoch, avg_loss, avg_loss_test))
            else:
                tqdm.write("Epoch: {0}, Avg RGB MSE loss: {1:.4f}".format(epoch, avg_loss))

            # output the arch's data on requisite epochs
            if(not(epoch == 0) and (epoch % self.savePeriod == 0)):
                self.saveTrainState("epoch_{0}".format(epoch))
        
        # final end-of-training outputstest_loss
        losses_test = []
        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(self.test_dataloader)):
                rendered_output, test_images = self.doTestRendering(sample_batched)
                rgb_output = rendered_output['rgb'].reshape((1, self.render_width, self.render_height, 3)).transpose(1,2)
                losses = self.test_loss.calculateLoss(test_images, rgb_output)
                losses_test.append(losses['loss'].item())
                # swap dimensions since tensorboard expects shape (batches, channels, w, h)
                # but test_images, etc. comes out as (batches, w, h, channels)
                test_images = torch.transpose(test_images, 3, 1).transpose(2,3)
                rgb_output = torch.transpose(rgb_output, 3, 1).transpose(2,3)
                self.tb_writer.add_images("test/gtImage/{0}".format(i_batch), test_images, self.train_epochs)
                self.tb_writer.add_images("test/rendered/{0}".format(i_batch), rgb_output, self.train_epochs)
        avg_loss_test = np.mean(losses_test)
        self.tb_writer.add_scalar('test/mse', avg_loss_test, self.train_epochs)
        tqdm.write("Epoch: {0}, Avg RGB loss: {1:.4f}, Test Avg RGB loss: {2:.4f}".format(self.train_epochs, avg_loss, avg_loss_test))

if __name__=="__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    configFile = "./configs/VoxelGridNerf/toolPartsCoarseFinePerturbedMasked.yml"
    if(len(sys.argv) == 2):
        configFile = str(sys.argv[1])
    arch = VoxelGridNerf(configFile)
    arch.train()