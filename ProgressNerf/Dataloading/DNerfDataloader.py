import json
from typing import Dict
import torch
import cv2 as cv
import os, re
import numpy as np
import yaml
import glob
from scipy.spatial.transform import Rotation as R
from ProgressNerf.Registries.DataloaderRegistry import register_dataloader, get_dataloader

@register_dataloader("DNerfDataloader")
class DNerfDataloader(torch.utils.data.Dataset):
    def __init__(self, config: Dict):
        self.baseDir = config['baseDataDir']
        self.datasetType = config['datasetType']
        self.numToLoad = config['samplesLimit']

        self.tool_label = 'main'
        self.rel_tools = ['main']

        self.scene_images = []

        for res in glob.glob(os.path.join(self.baseDir, self.datasetType, "r_*.png")):
            self.scene_images.append(res)

        if(self.numToLoad is not None and self.numToLoad > 0):
            indices = np.random.choice(len(self.scene_images), self.numToLoad)
            self.scene_images = [self.scene_images[idx] for idx in indices]

        self.transforms_file = os.path.join(self.baseDir,"transforms_{0}.json".format(self.datasetType))
        self.transforms_data = json.load(open(self.transforms_file, 'r'))
        

    def get_scene_images(self, scene_name, base_dir=None, dataset=None):
        if(dataset is None):
            dataset = self.datasetType
        if(base_dir is None):
            base_dir = self.baseDir
        base_img_dir = os.path.join(base_dir, dataset, "rgb") + "/"
        files = glob.glob("image_full_*.png", root_dir=base_img_dir)
        return files


    def __len__(self):
        return len(self.scene_images)

    def getTransformDataForImage(self, imgFileName):
        lookup_key = imgFileName.replace(self.baseDir, ".").replace(".png", "")
        frames = self.transforms_data["frames"]
        for frame in frames:
            if(frame["file_path"] == lookup_key):
                return frame
        raise Exception("lookup key not found! Original file: {0}, lookup key: {1}".format(imgFileName, lookup_key))

    def __getitem__(self, idx):
        img_file = self.scene_images[idx]

        img_data = np.ascontiguousarray(cv.imread(img_file, cv.IMREAD_COLOR)[:,:,::-1]) / 255.0 # swap from BGR to RGB and normalize

        tf_data = self.getTransformDataForImage(img_file)

        pose_mat = torch.Tensor(tf_data["transform_matrix"])
        pose_data = {self.rel_tools[0]: {"pose" : pose_mat, "label" : 1}}

        seg_data = torch.ones((img_data.shape[0], img_data.shape[1]))

        configuration = torch.Tensor([tf_data["time"]]).to(dtype=torch.float32)

        toReturn = {\
            "image" : torch.from_numpy(img_data).to(dtype=torch.float32),\
            "segmentation" : seg_data.to(dtype=torch.float32),\
            "configuration" : configuration.unsqueeze(0),\
            }
            
        
        for obj_key in pose_data:
            if(obj_key in self.rel_tools):
                pose_key = obj_key + "_pose"
                label_key = obj_key + "_label"
                pose = pose_data[obj_key]["pose"]
                toReturn[pose_key] = pose.to(dtype=torch.float32)
                toReturn[label_key] = pose_data[obj_key]["label"]

        return toReturn
