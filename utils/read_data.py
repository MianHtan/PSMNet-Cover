import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import logging
import os

from pathlib import Path

from PIL import Image


def read_img(filename, resize):
    ext = os.path.splitext(filename)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.jpg' or ext == '.tif' or ext == ".tiff":
        img = Image.open(filename)
        w, h = img.size
        if resize[0] != w:
            img = img.resize(resize)
        img = np.array(img)
    return img

def read_disp_dfc(filename, resize, min_disp, max_disp):
    '''
    @para: resize -> [w, h]
    '''
    disp = Image.open(filename)
    w, h = disp.size
    if resize[0] != w:
        disp = disp.resize(resize)
    disp = np.array(disp)
      
    disp[np.isnan(disp)] = 0

    # img has been resized, thus the disparity should resized
    if resize[0] != w:
        scale = w / resize[0]
        disp = disp / scale

    # generate mask
    valid = np.logical_and((disp != 0), (disp < max_disp))
    valid = np.logical_and(valid, (disp > min_disp))

    # valid = np.ones_like(disp)
    return disp, valid


def read_disp_whu(filename, resize, min_disp, max_disp):
    '''
    @para: resize -> [w, h]
    '''
    disp = Image.open(filename)
    w, h = disp.size
    if resize[0] != w:
        disp = disp.resize(resize)
    disp = np.array(disp)

    # img has been resized, thus the disparity should resized
    if resize[0] != w:
        scale = w / resize[0]
        disp = disp / scale

    # generate mask
    valid = np.logical_and((disp < max_disp), (disp > min_disp))
    # valid = np.ones_like(disp)
    return disp, valid

def img_norm(img:np.array) -> torch.tensor :
    img = img/255
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
    img = transform(img)
    img = img.float()

    return img