import torch
import numpy as np

def BuildCameraMatrix(fx, fy, tx, ty):
    toRet = torch.eye(3)
    toRet[0,0] = fx
    toRet[1,1] = fy
    toRet[0,2] = tx
    toRet[1,2] = ty
    return toRet