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
import h5py
import re

@register_dataloader("ProgLabelDataloader")
class ProgLabelDataloader(torch.utils.data.Dataset):
    def __init__(self, config: Dict):
        self.baseDir = config['baseDataDir']
        self.objectName = config['object_label']
        self.numToLoad = config['samplesLimit']

        self.scene_images = list(self.get_object_images())
        self.poses = [self.get_poses(img) for img in self.scene_images]

        if(self.numToLoad is not None and self.numToLoad > 0):
            indices = np.random.choice(len(self.scene_images), self.numToLoad)
            self.scene_images = [self.scene_images[idx] for idx in indices]
            self.poses = [self.poses[idx] for idx in indices]
        

    def get_object_images(self, base_dir=None, objectName=None):
        if(objectName is None):
            objectName = self.objectName
        if(base_dir is None):
            base_dir = self.baseDir

        base_img_dir = os.path.join(base_dir, objectName, "rgb") + "/"
        files = os.listdir(base_img_dir)
        regex = re.compile("[0-9]{19}(.png){1}")
        
        for file in files:
            if(regex.match(file)):
                yield os.path.join(base_img_dir, file)

    def change_img_mask(self, img):
        return img > 0.0
    
    def get_poses(self, img_file):
        pose_file = img_file.replace("rgb","pose").replace(".png",".txt")
        pose = np.loadtxt(pose_file)
        return pose

    def __len__(self):
        return len(self.scene_images)

    def __getitem__(self, idx):
        img_file = self.scene_images[idx]

        img_data = np.ascontiguousarray(cv.imread(img_file, cv.IMREAD_COLOR)[:,:,::-1]) / 255.0 # swap from BGR to RGB and normalize
        seg_data = np.all((img_data > 0), axis = -1) * 1.0
        img_data[seg_data == 0.0 , ...] = 1.0
        image_toReturn = torch.from_numpy(img_data).to(dtype=torch.float32)
        toReturn = {\
            "image" : image_toReturn,\
            "depth" : torch.zeros_like(image_toReturn)[...,0],\
            "segmentation" : torch.from_numpy(seg_data).to(dtype=torch.float32),\
            "idx" : torch.Tensor([idx]).to(dtype=int),\
            "scene" : torch.Tensor([0]).to(dtype=int),\
            }

        obj_key = str(self.objectName)
        pose_key = obj_key + "_pose"
        label_key = obj_key + "_label"
        pose = self.poses[idx]
        toReturn[pose_key] = torch.linalg.inv(torch.from_numpy(pose).to(dtype=torch.float32))
        toReturn[label_key] = 1.0

        return toReturn
