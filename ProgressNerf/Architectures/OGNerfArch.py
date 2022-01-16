from datetime import datetime
from re import L
from typing import Dict
from numpy.core.numeric import full
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
from tqdm import tqdm
import numpy as np
import os
import cv2 as cv
from ProgressNerf.Registries.ModelRegistry import get_model
from ProgressNerf.Registries.DataloaderRegistry import get_dataloader
from ProgressNerf.Registries.RaypickerRegistry import get_raypicker
from ProgressNerf.Registries.RaysamplerRegistry import get_raysampler
from ProgressNerf.Registries.EncoderRegistry import get_encoder
from ProgressNerf.Registries.RendererRegistry import get_renderer
import ProgressNerf.Dataloading.ToolsPartsDataloader
import ProgressNerf.Raycasting.RandomRaypicker
import ProgressNerf.Raycasting.NearFarRaysampler
import ProgressNerf.Raycasting.WeightedRaypicker
import ProgressNerf.Raycasting.WeightedRaysampler
import ProgressNerf.Raycasting.PerturbedRaysampler
import ProgressNerf.Encoders.PositionalEncoder
import ProgressNerf.Models.OGNerf
import ProgressNerf.NeuralRendering.NeuralRenderer
from ProgressNerf.Utils.CameraUtils import BuildCameraMatrix
import yaml

# this architecture represents the original NeRF paper as proposed by Mildenhall et al.
# see https://arxiv.org/abs/2003.08934 for further details
class OGNerfArch(object):
    def __init__(self, configFile:str) -> None:
        super().__init__()
        
        config = yaml.load(open(configFile,'r'), yaml.FullLoader)

        # pos enc levels = 10
        # dir enc levels = 4
        # parse the configuration & populate values
        self.parseConfig(config)

    # extract this architecture's parameters from the configuration YAML file
    def parseConfig(self, config:Dict):
        self.base_dir = config['baseDir']

        self.evalPeriod = config['eval_period']
        self.savePeriod = config['save_period'] 

        train_loader_config = config['train_dataloader']
        self.train_loader = get_dataloader(train_loader_config['dataloader'])(train_loader_config[train_loader_config['dataloader']])
        test_loader_config = config['test_dataloader']
        self.test_loader = get_dataloader(test_loader_config['dataloader'])(test_loader_config[test_loader_config['dataloader']])

        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.device = config['device']

        self.cam_f_t = [config['camera_fx'],config['camera_fy'],config['camera_tx'],config['camera_ty']]
        self.cam_matrix = BuildCameraMatrix(*self.cam_f_t)

        self.render_width = config['render_resoluion'][0]
        self.render_height = config['render_resoluion'][1]
        self.raypicker = get_raypicker(config['raypicker'])(config[config['raypicker']])
        self.raypicker.setCameraParameters(self.cam_matrix, self.render_height, self.render_width)

        self.raysampler = get_raysampler(config['raysampler'])(config[config['raysampler']])

        self.train_epochs = config['train_epochs']
        self.lr = config['optim_lr']

        pos_enc_dict = config['posEncoder']
        self.pos_encoder = get_encoder(pos_enc_dict['encoder'])(pos_enc_dict[pos_enc_dict['encoder']])

        dir_enc_dict = config['dirEncoder']
        self.dir_encoder = get_encoder(dir_enc_dict['encoder'])(dir_enc_dict[dir_enc_dict['encoder']])

        self.renderer = get_renderer(config['renderer'])(config[config['renderer']])

        coarse_config = config['coarse_model']
        self.nn = get_model(coarse_config['nnModel'])(coarse_config[coarse_config['nnModel']])

        if('fine_model' in config.keys()):
            fine_config = config['fine_model']

            self.nn_fine = get_model(fine_config['nnModel'])(fine_config[fine_config['nnModel']])
            self.raysampler_fine = get_raysampler(fine_config['raysampler'])(config[config['raysampler']])

        self.tool = config['desired_tool']
        self.masks = None
        if('apply_masks' in config.keys()):
            self.masks = config['apply_masks']

        self.tb_writer = None

    # helper function for saving the state of the training process
    def saveTrainState(self, checkpoint_dir:str = "latest"):
        output_dir = os.path.join(self.base_dir, checkpoint_dir)
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir)
        torch.save(self.nn.state_dict(), os.path.join(output_dir, "mlp_dict.ptr"))
        if(self.nn_fine is not None):
            torch.save(self.nn_fine.state_dict(), os.path.join(output_dir, "mlp_dict_fine.ptr"))
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer_dict.ptr"))

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
        rendered_output = self.renderer.renderRays(mlp_outputs, sampled_distances)

        if(self.nn_fine is not None):
            weighted_resampling_other_info = {'distances': sampled_distances, 'sigmas': mlp_outputs[:,:,:,3].relu()}
            _, resampled_distances = self.raysampler_fine.sampleRays(ray_origins, ray_dirs, other_info=weighted_resampling_other_info)
            resampled_distances, _ = torch.cat((sampled_distances, resampled_distances), dim = -1).sort()
            resampled_locations = ray_origins[...,None,:] + ray_dirs[..., None, :] * resampled_distances[...,:,None]
            encoded_re_locs = self.pos_encoder.encodeFeature(resampled_locations)
            encoded_dirs_fine = encoded_dirs.unsqueeze(2).repeat(1,1,encoded_re_locs.shape[2],1)
            fine_mlp_outputs = self.nn_fine.forward(encoded_re_locs, encoded_dirs_fine)
            rendered_output_fine = self.renderer.renderRays(fine_mlp_outputs, resampled_distances)
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

        render_result = self.render(ray_origins, ray_dirs)

        train_pixels = train_imgs[:, ijs[:,1], ijs[:,0], :]

        return render_result, train_pixels

    # performs a rendering of the full image
    # unlike doTrainRendering, the raypicker is not run (except to the the totality to rays created). Instead, we
    # extract sub-batches of all rays for rendering to prevent out-of-memory issues
    def doEvalRendering(self, sample_batched):
        test_imgs = sample_batched['image'].to(self.device) # (batch_size, W, H, 3)
        cam_poses = torch.linalg.inv(sample_batched['{0}_pose'.format(self.tool)]).to(self.device) # (batch_size, 4, 4)
        ray_origins, ray_dirs = self.raypicker.getAllRays(cam_poses) # (batch_dim, width*height, 3), (batch_dim, width*height, 3)
        total_sample_size = ray_origins.shape[1]
        eval_subBatchSize = 1024
        num_subBatches = math.ceil(total_sample_size / eval_subBatchSize)
        full_rendering = {}

        for subBatch_idx in range(num_subBatches):
            start_idx = subBatch_idx * eval_subBatchSize
            end_idx = min((subBatch_idx + 1) * eval_subBatchSize, total_sample_size)
            subBatch_ray_origins = ray_origins[:,start_idx:end_idx,:]
            subBatch_ray_dirs = ray_dirs[:,start_idx:end_idx,:]
            subBatch_rendering = self.render(subBatch_ray_origins, subBatch_ray_dirs)
            if('rgb' in full_rendering.keys()):
                full_rendering['rgb'] = torch.cat((full_rendering['rgb'], subBatch_rendering['rgb'].clone()), dim=1)
            else:
                full_rendering['rgb'] = subBatch_rendering['rgb'].clone()

        return full_rendering, test_imgs

    def train(self):
        # create the Tensorboard Writer
        self.tb_writer = SummaryWriter(log_dir=str(self.base_dir))

        # create train/test dataloaders
        self.train_dataloader = DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.test_loader, batch_size=1, shuffle=False, num_workers=self.num_workers)

        # set the main models to train, move the GPU if necessary, and extract the optimization parameters
        self.nn.train()
        self.nn.to(self.device)
        learning_params = list(self.nn.parameters())
        if(self.nn_fine is not None):
            self.nn_fine.train()
            self.nn_fine.to(self.device)
            learning_params = learning_params + list(self.nn_fine.parameters())

        # setup optimizer and loss function
        self.optimizer = torch.optim.Adam(learning_params, lr=self.lr)
        loss_rgb = torch.nn.MSELoss()

        # main training loop
        for epoch in tqdm(range(self.train_epochs)):
            losses = []
            losses_test = []
            # Do training step
            self.nn.train()
            if(self.nn_fine is not None):
                self.nn_fine.train()
            for i_batch, sample_batched in tqdm(enumerate(self.train_dataloader)):
                rendered_output, train_pixels = self.doTrainRendering(sample_batched)
                loss_rgb_i = loss_rgb(train_pixels, rendered_output['rgb'])
                loss_rgb_i.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss_rgb_i.item())
            
            # compute & log metrics
            avg_mse = np.mean(losses)
            self.tb_writer.add_scalar('train/mse', avg_mse, epoch)
            self.optimizer.zero_grad()

            # if we need to do evaluation this epoch
            if(not(epoch == 0) and (epoch % self.evalPeriod == 0)):
                # Do test/eval
                self.nn.eval()
                if(self.nn_fine is not None):
                    self.nn_fine.eval()
                # render the entire image instead of only the sampled pixels, compute loss & send to tensorboard
                with torch.no_grad():
                    for i_batch, sample_batched in tqdm(enumerate(self.test_dataloader)):
                        rendered_output, test_images = self.doEvalRendering(sample_batched)
                        rgb_output = rendered_output['rgb'].contiguous().reshape((1, self.render_width, self.render_height, 3)).transpose(1,2).contiguous()
                        loss_rgb_i_test = loss_rgb(test_images, rgb_output)
                        losses_test.append(loss_rgb_i_test.item())
                        # swap dimensions since tensorboard expects shape (batches, channels, w, h)
                        # but test_images, etc. comes out as (batches, w, h, channels)
                        test_images = torch.transpose(test_images, 3, 1).transpose(2,3).contiguous()
                        rgb_output = torch.transpose(rgb_output, 3, 1).transpose(2,3).contiguous()
                        self.tb_writer.add_images("test/gtImage/{0}".format(i_batch), test_images, epoch)
                        self.tb_writer.add_images("test/rendered/{0}".format(i_batch), rgb_output, epoch)
                avg_mse_test = np.mean(losses_test)
                self.tb_writer.add_scalar('test/mse', avg_mse_test, epoch)
                tqdm.write("Epoch: {0}, Avg RGB MSE loss: {1:.4f}, Test Avg RGB MSE loss: {2:.4f}".format(epoch, avg_mse, avg_mse_test))
            else:
                tqdm.write("Epoch: {0}, Avg RGB MSE loss: {1:.4f}".format(epoch, avg_mse))

            # output the arch's data on requisite epochs
            if(not(epoch == 0) and (epoch % self.savePeriod == 0)):
                self.saveTrainState("epoch_{0}".format(epoch))
        
        # final end-of-training outputs
        self.saveTrainState("final_epoch_{0}".format(self.train_epochs))
        # Do one final test/eval after final training run
        self.nn.eval()
        losses_test = []
        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(self.test_dataloader)):
                rendered_output, test_images = self.doEvalRendering(sample_batched)
                rgb_output = rendered_output['rgb'].reshape((1, self.render_width, self.render_height, 3)).transpose(1,2)
                loss_rgb_i_test = loss_rgb(test_images, rgb_output)
                losses_test.append(loss_rgb_i_test.item())
                # swap dimensions since tensorboard expects shape (batches, channels, w, h)
                # but test_images, etc. comes out as (batches, w, h, channels)
                test_images = torch.transpose(test_images, 3, 1).transpose(2,3)
                rgb_output = torch.transpose(rgb_output, 3, 1).transpose(2,3)
                self.tb_writer.add_images("test/gtImage/{0}".format(i_batch), test_images, self.train_epochs)
                self.tb_writer.add_images("test/rendered/{0}".format(i_batch), rgb_output, self.train_epochs)
        avg_mse_test = np.mean(losses_test)
        self.tb_writer.add_scalar('test/mse', avg_mse_test, self.train_epochs)
        tqdm.write("Epoch: {0}, Avg RGB MSE loss: {1:.4f}, Test Avg RGB MSE loss: {2:.4f}".format(self.train_epochs), avg_mse, avg_mse_test)

if __name__=="__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    arch = OGNerfArch("/home/stanlew/src/ProgressNerf/configs/OGNerfArch/toolPartsCoarseFinePerturbed.yml")
    arch.train()