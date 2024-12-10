import os
import sys
import copy
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from utils import image,file

class ImgDateSetForSemeticSegmentation(Dataset):
    def __init__(self, img_paths, mask_paths, transform_img=None,transform_mask=None):
   
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img__ndarr = image.load_image(img_path)
        mask_ndarr = image.load_image(mask_path)
        img__tensor = torch.from_numpy(img__ndarr)
        mask__tensor = torch.from_numpy(mask_ndarr)
        if img__tensor.dim() == 2:
            img__tensor = img__tensor.unsqueeze(2)
        if mask__tensor.dim() == 2:
            mask__tensor = mask__tensor.unsqueeze(2)
        img__tensor = img__tensor.permute(2,0,1)        
        mask__tensor = mask__tensor.permute(2,0,1)
        if self.transform_img:
            img__tensor = self.transform_img(img__tensor)
        if self.transform_mask:
            mask__tensor = self.transform_mask(mask__tensor)
        if not isinstance(img__tensor, torch.Tensor):
            raise TypeError(f"Expected output transform_img to be a Tensor, but got {type(img__tensor)}.")
        if not isinstance(mask__tensor, torch.Tensor):
            raise TypeError(f"Expected output transform_mask to be a Tensor, but got {type(mask__tensor)}.")
        return img__tensor.float(), mask__tensor.int()
