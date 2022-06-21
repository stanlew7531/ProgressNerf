from operator import index
from curses import meta
from typing import Dict
import torch
import cv2 as cv
import os, re
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat
from ProgressNerf.Registries.DataloaderRegistry import register_dataloader, get_dataloader

@register_dataloader("YCBVideoDataloader")
class YCBVideoDataloader(torch.utils.data.Dataset):
    def __init__(self, config: Dict):
        self.baseDir = config['baseDataDir']
        self.scenes = config['scenes']
        self.datasetType = config['datasetType']
        self.numToLoad = config['samplesLimit']
        if('label' in config):
            self.tool_label = config['label']
        if('rel_tools' in config):
            self.rel_tools = config['rel_tools']

        self.scene_images = []
        self.scene_depths = []
        self.scene_poses = []
        self.scene_segmentations = []
        self.scene_ids = []
        for scene in self.scenes:
            for res in self.get_scene_images(scene):
                self.scene_ids.append(scene)
                self.scene_images.append(res)
                self.scene_depths.append(self.change_img_depth(res))
                self.scene_poses.append(self.change_img_poses(res))
                self.scene_segmentations.append(self.change_img_labels(res))

        if(self.numToLoad is not None and self.numToLoad > 0):
            indices = np.random.choice(len(self.scene_images), self.numToLoad)
            self.scene_images = [self.scene_images[idx] for idx in indices]
            self.scene_depths = [self.scene_depths[idx] for idx in indices]
            self.scene_poses = [self.scene_poses[idx] for idx in indices]
            self.scene_segmentations = [self.scene_segmentations[idx] for idx in indices]
        

    def get_scene_images(self, scene_name, base_dir=None, dataset=None):
        if(dataset is None):
            dataset = self.datasetType
        if(base_dir is None):
            base_dir = self.baseDir
        base_img_dir = os.path.join(base_dir, dataset, scene_name) + "/"
        files = os.listdir(base_img_dir)
        regex = re.compile("[0-9]{6}(-color.png){1}")
        for file in files:
            if(regex.match(file)):
                yield os.path.join(base_img_dir, file)

    def change_img_depth(self, img):
        return img.replace('-color', "-depth")
    
    def change_img_poses(self, img):
        return img.replace("-color.png", "-meta.mat")

    def change_img_labels(self, img):
        return img.replace("-color.png", "-label.png")

    def __len__(self):
        return len(self.scene_images)

    def __getitem__(self, idx):
        img_file = self.scene_images[idx]
        pose_file = self.scene_poses[idx]
        seg_file = self.scene_segmentations[idx]
        scene_id = self.scene_ids[idx]

        img_data = np.ascontiguousarray(cv.imread(img_file, cv.IMREAD_COLOR)[:,:,::-1]) / 255.0 # swap from BGR to RGB and normalize
        meta_data = loadmat(pose_file)
        indexes = meta_data['cls_indexes'].squeeze()
        poses = meta_data['poses']
        seg_data = cv.imread(seg_file, cv.IMREAD_UNCHANGED)
        image_toReturn = torch.from_numpy(img_data).to(dtype=torch.float32)
        toReturn = {\
            "image" : image_toReturn,\
            "depth" : torch.zeros_like(image_toReturn)[...,0],\
            "segmentation" : torch.from_numpy(seg_data).to(dtype=torch.float32),\
            "idx" : torch.Tensor([idx]).to(dtype=int),\
            "scene" : torch.Tensor([scene_id]).to(dtype=int),\
            }

        for idx in range(len(indexes)):
            obj_key = str(indexes[idx]).zfill(2)
            if(obj_key in self.rel_tools):
                pose_key = obj_key + "_pose"
                label_key = obj_key + "_label"
                pose = poses[:,:, idx]
                pose = np.concatenate((pose, np.array([[0.0,0.0,0.0,1.0]])), axis=0)
                toReturn[pose_key] = torch.from_numpy(pose).to(dtype=torch.float32)
                toReturn[label_key] = indexes[idx]

        return toReturn
