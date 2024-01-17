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

def read_disp_dfc(filename, resize):
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
        scale = w/resize[0]
        disp = disp / scale

    # generate mask
    valid = (disp != 0)
    # valid = np.ones_like(disp)
    return disp, valid


def read_disp_whu(filename, resize):
    '''
    @para: resize -> [w, h]
    '''
    disp = Image.open(filename)
    w, h = disp.size
    if resize[0] != w:
        disp = disp.resize(resize)
    disp = np.array(disp)

    # generate mask
    valid = (disp > -128)

    # img has been resized, thus the disparity should resized
    if resize[0] != w:
        scale = w/resize[0]
        disp = disp / scale
    # valid = np.ones_like(disp)
    return disp, valid

def img_norm(img:np.array) -> torch.tensor :
    img = img/255
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
    img = transform(img)
    img = img.float()

    return img