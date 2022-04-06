from typing import Dict
import torch
import cv2 as cv
import os, re
import numpy as np
import yaml
import glob
from scipy.spatial.transform import Rotation as R
from ProgressNerf.Registries.DataloaderRegistry import register_dataloader, get_dataloader

@register_dataloader("semiSynDataloader")
class SemisynDataloader(torch.utils.data.Dataset):
    def __init__(self, config: Dict):
        self.baseDir = config['baseDataDir']
        self.datasetType = config['datasetType']
        self.numToLoad = config['samplesLimit']
        if('label' in config):
            self.tool_label = config['label']
        else:
            self.tool_label = 'main'
        if('rel_tools' in config):
            self.rel_tools = config['rel_tools']
        else:
            self.tool_label = ['main']

        self.scene_images = []
        self.scene_depths = []
        self.scene_poses = []
        self.scene_configurations = []
        self.scene_segs = []
        for res in glob.glob(os.path.join(self.baseDir, self.datasetType, "rgb", "image_full_*.png")):
            self.scene_images.append(res)
            self.scene_depths.append(self.change_img_depth(res))
            self.scene_poses.append(self.change_img_poses(res))
            self.scene_configurations.append(self.change_img_configs(res))
            self.scene_segs.append(self.change_img_segs(res))


        if(self.numToLoad is not None and self.numToLoad > 0):
            indices = np.random.choice(len(self.scene_images), self.numToLoad)
            self.scene_images = [self.scene_images[idx] for idx in indices]
            self.scene_depths = [self.scene_depths[idx] for idx in indices]
            self.scene_poses = [self.scene_poses[idx] for idx in indices]
            self.scene_configurations = [self.scene_configurations[idx] for idx in indices]
            self.scene_segs = [self.scene_segs[idx] for idx in indices]
        

    def get_scene_images(self, scene_name, base_dir=None, dataset=None):
        if(dataset is None):
            dataset = self.datasetType
        if(base_dir is None):
            base_dir = self.baseDir
        base_img_dir = os.path.join(base_dir, dataset, "rgb") + "/"
        files = glob.glob("image_full_*.png", root_dir=base_img_dir)
        return files

    def change_img_depth(self, img):
        return img.replace("rgb", "depth").replace('image_full_', "depth_")
    
    def change_img_poses(self, img):
        return img.replace("rgb", "pose").replace("image_full_", "pose_").replace(".png", ".pth")

    def change_img_configs(self, img):
        return img.replace("rgb", "configuration").replace("image_full_", "configuration_").replace(".png", ".yml")

    def change_img_segs(self, img):
        return img.replace("rgb", "segmentation").replace("image_full_", "seg_")

    def __len__(self):
        return len(self.scene_images)

    def __getitem__(self, idx):
        img_file = self.scene_images[idx]
        depth_file = self.scene_depths[idx]
        pose_file = self.scene_poses[idx]
        configuration_file = self.scene_configurations[idx]
        seg_file = self.scene_segs[idx]

        img_data = np.ascontiguousarray(cv.imread(img_file, cv.IMREAD_COLOR)[:,:,::-1]) / 255.0 # swap from BGR to RGB and normalize
        depth_data = torch.from_numpy(np.int32(cv.imread(depth_file, cv.IMREAD_UNCHANGED))) # convert to int32 so that from_numpy works below
        depth_data = depth_data.to(torch.float64)/1000.0
        pose_mat = torch.load(pose_file)[self.rel_tools[0]].squeeze()
        pose_data = {self.rel_tools[0]: {"pose" : pose_mat, "label" : 1}}

        seg_data = torch.from_numpy(np.int32(cv.imread(seg_file, cv.IMREAD_UNCHANGED)))
        configuration = np.array(yaml.load(open(configuration_file,'r'), yaml.FullLoader)["configuration"])
        configuration = torch.from_numpy(configuration).to(dtype=torch.float32)
        toReturn = {\
            "image" : torch.from_numpy(img_data).to(dtype=torch.float32),\
            "depth" : depth_data.to(dtype=torch.float32),\
            "segmentation" : seg_data.to(dtype=torch.float32),\
            "configuration" : configuration.unsqueeze(0),\
            }

        for obj_key in pose_data:
            if(obj_key in self.rel_tools):
                pose_key = obj_key + "_pose"
                label_key = obj_key + "_label"
                pose = pose_data[obj_key]["pose"]
                # pose_r = R.from_quat(pose_quat[1:4] + [pose_quat[0]]).as_matrix()
                # pose = np.eye(4)
                # pose[0:3, 0:3] = pose_r
                # pose[0:3, 3] = np.transpose(pose_t)
                toReturn[pose_key] = torch.linalg.inv(pose).to(dtype=torch.float32)
                toReturn[label_key] = pose_data[obj_key]["label"]

        return toReturn
