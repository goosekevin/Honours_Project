import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from random import shuffle
import scipy.misc
from PIL import Image

import cv2
import os
import glob


def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model    


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

def difference(tensor1, tensor2, threshhold=1):

    bat = tensor1.shape[0]
    c = tensor1.shape[1]
    l = tensor1.shape[2]
    w = tensor1.shape[3]
    new_tensor = np.zeros((bat, c, l, w), dtype=np.float)
    temp_tensor1 = Variable(tensor1, requires_grad=False)
    temp_tensor2 = Variable(tensor2, requires_grad=False)
    temp_tensor1 = temp_tensor1.cpu().numpy()
    temp_tensor2 = temp_tensor2.cpu().numpy()
    for b in range(bat):
        for x in range(l):
            for y in range(w):
                for z in range(c):
                    if abs(temp_tensor1.item((b, z, x, y)) - temp_tensor2.item((b, z, x, y))) <= threshhold:
                        new_tensor[b,z,x,y] = 0
                    else:
                        new_tensor[b,z,x,y] = temp_tensor1.item((b, z, x, y))

    new_tensor =  torch.from_numpy(new_tensor)
    new_tensor = new_tensor.float().cuda(0)
    new_tensor = Variable(new_tensor, requires_grad=True)
    return new_tensor

