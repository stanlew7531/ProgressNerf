import numpy as np
import torch

def GetPCDPoints(file_path:str):
    data = torch.from_numpy(np.genfromtxt(file_path, delimiter=' ', skip_header=11))
    return data

def GetPCDMaxMinAxes(data:torch.Tensor):
    max_vals = torch.max(data, dim=0).values
    min_vals = torch.min(data, dim=0).values
    return torch.vstack((max_vals,min_vals))

def GetPCDBoundingBox(file_path:str):
    return GetPCDMaxMinAxes(GetPCDPoints(file_path))