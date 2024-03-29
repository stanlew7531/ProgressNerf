from typing import Dict
import torch
import cv2 as cv
import os, re
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from ProgressNerf.Registries.DataloaderRegistry import register_dataloader, get_dataloader

@register_dataloader("pybulletDataloader")
class PyBulletDataloader(torch.utils.data.Dataset):
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
        for scene in self.scenes:
            for res in self.get_scene_images(scene):
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

        self.tf_OGL_OCV = np.array([[1., 0., 0., 0.],\
                                    [0., -1., 0., 0.],\
                                    [0., 0., -1., 0.],\
                                    [0., 0., 0., 1.]])
        

    def get_scene_images(self, scene_name, base_dir=None, dataset=None):
        if(dataset is None):
            dataset = self.datasetType
        if(base_dir is None):
            base_dir = self.baseDir
        base_img_dir = os.path.join(base_dir, scene_name, dataset, "images") + "/"
        files = os.listdir(base_img_dir)
        regex = re.compile("[0-9]{6}(_rgb.png){1}")
        for file in files:
            if(regex.match(file)):
                yield os.path.join(base_img_dir, file)

    def change_img_depth(self, img):
        return img.replace('_rgb', "_depth")
    
    def change_img_poses(self, img):
        return img.replace("images", "annotations").replace("_rgb.png", "_poses.yml")

    def change_img_labels(self, img):
        return img.replace("images", "annotations").replace("_rgb.png", "_labels.png")

    def __len__(self):
        return len(self.scene_images)

    def __getitem__(self, idx):
        img_file = self.scene_images[idx]
        depth_file = self.scene_depths[idx]
        pose_file = self.scene_poses[idx]
        seg_file = self.scene_segmentations[idx]

        img_data = np.ascontiguousarray(cv.imread(img_file, cv.IMREAD_COLOR)[:,:,::-1]) / 255.0 # swap from BGR to RGB and normalize
        depth_data = torch.from_numpy(np.int32(cv.imread(depth_file, cv.IMREAD_UNCHANGED))) # convert to int32 so that from_numpy works below
        depth_data = depth_data.to(torch.float64)/1000.0
        annotation_data = yaml.load(open(pose_file,'r'), yaml.FullLoader)
        seg_data = cv.imread(seg_file, cv.IMREAD_UNCHANGED)

        # get the camera pose from the pybullet output
        # note that opengl is column major (vs row major for opencv) hence the transpose
        # we also perform a transform here to rotate the camera, such that its axes are aligned with
        # what we expect an OpenCV camera to have vis-a-vis rendering
        camera_pose = np.array(annotation_data['camera_pose']).reshape(4,4).transpose()
        camera_pose = np.matmul(self.tf_OGL_OCV, camera_pose)
        part_names = annotation_data['mask_labes'].keys()

        toReturn = {\
            "image" : torch.from_numpy(img_data).to(dtype=torch.float32),\
            "depth" : depth_data.to(dtype=torch.float32),\
            "segmentation" : torch.from_numpy(seg_data).to(dtype=torch.float32),\
            "idx" : torch.Tensor([idx]).to(dtype=int)\
            }
        for obj_key in part_names:
            if(obj_key in self.rel_tools):
                pose_key = obj_key + "_pose"
                label_key = obj_key + "_label"
                pose_quat = annotation_data['part_poses_world'][obj_key]["rot_q_xyzw"]
                pose_t = annotation_data['part_poses_world'][obj_key]["trans"]
                pose_r = R.from_quat(pose_quat).as_matrix()
                pose = np.eye(4)
                pose[0:3, 0:3] = pose_r
                pose[0:3, 3] = np.transpose(pose_t)

                pose = np.matmul(pose, np.linalg.inv(camera_pose))
                
                toReturn[pose_key] = torch.from_numpy(pose).to(dtype=torch.float32)
                toReturn[label_key] = annotation_data['mask_labes'][obj_key]

        return toReturn
