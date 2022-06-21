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

@register_dataloader("YCBObjectDataloader")
class YCBObjectDataloader(torch.utils.data.Dataset):
    def __init__(self, config: Dict):
        self.baseDir = config['baseDataDir']
        self.objectName = config['object_label']
        self.numToLoad = config['samplesLimit']

        calibration_file = h5py.File("/home/stanlew/data/ycb/ycb/005_tomato_soup_can/calibration.h5",'r')
        self.calibration_data = [calibration_file['H_N1_from_NP5'][:],\
                                    calibration_file['H_N2_from_NP5'][:],\
                                    calibration_file['H_N3_from_NP5'][:],\
                                    calibration_file['H_N4_from_NP5'][:],\
                                    calibration_file['H_N5_from_NP5'][:]]
        calibration_file.close()

        self.scene_images = []
        self.camera_ns = []
        self.tt_degs = []
        for cam_n, tt_deg, scene_img in self.get_object_images():
            self.scene_images.append(scene_img)
            self.camera_ns.append(int(cam_n))
            self.tt_degs.append(int(tt_deg))

        self.poses = [self.get_poses(idx) for idx in range(len(self.scene_images))]
        self.scene_segmentations = [self.change_img_mask(image) for image in self.scene_images]

        if(self.numToLoad is not None and self.numToLoad > 0):
            indices = np.random.choice(len(self.scene_images), self.numToLoad)
            self.scene_images = [self.scene_images[idx] for idx in indices]
            self.camera_ns = [self.camera_ns[idx] for idx in indices]
            self.tt_degs = [self.tt_degs[idx] for idx in indices]
            self.poses = [self.poses[idx] for idx in indices]
            self.scene_segmentations = [self.scene_segmentations[idx] for idx in indices]
        

    def get_object_images(self, base_dir=None, objectName=None):
        if(objectName is None):
            objectName = self.objectName
        if(base_dir is None):
            base_dir = self.baseDir

        base_img_dir = os.path.join(base_dir, objectName) + "/"
        files = os.listdir(base_img_dir)
        regex = re.compile("(N){1}[0-9]{1}(_){1}[0-9]{1,3}(.jpg){1}")
        
        for file in files:
            if(regex.match(file)):
                split = list(filter(bool, re.split("N|_|.jpg", file)))
                camera_n = split[0]
                tt_deg = split[1]
                yield camera_n, tt_deg, os.path.join(base_img_dir, file)

    def change_img_mask(self, img):
        return img.replace(self.objectName, self.objectName + "/masks").replace(".jpg","_mask.pbm")
    
    def get_poses(self, i):
        image = self.scene_images[i]
        camera_n = self.camera_ns[i]
        camera_tf = self.calibration_data[camera_n - 1]

        pose_file = image.replace(self.objectName, self.objectName + "/poses").replace(".jpg","_pose.h5").replace("N{0}".format(camera_n), "NP5")
        with h5py.File(pose_file,'r') as base_pose_file:
            base_pose = base_pose_file['H_table_from_reference_camera'][...]

        cam_pose = np.linalg.inv(camera_tf) @ base_pose
        return cam_pose

    def __len__(self):
        return len(self.scene_images)

    def __getitem__(self, idx):
        img_file = self.scene_images[idx]
        seg_file = self.scene_segmentations[idx]

        img_data = np.ascontiguousarray(cv.imread(img_file, cv.IMREAD_COLOR)[:,:,::-1]) / 255.0 # swap from BGR to RGB and normalize

        seg_data = cv.imread(seg_file, cv.IMREAD_UNCHANGED)
        
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
        toReturn[pose_key] = torch.from_numpy(pose).to(dtype=torch.float32)
        toReturn[label_key] = 0.0

        return toReturn
