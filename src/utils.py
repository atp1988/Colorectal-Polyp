
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
from torchvision import transforms

import os
import cv2
from glob import glob
import numpy as np
import shutil
from PIL import Image
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style
from absl import app, flags, logging
from absl.flags import FLAGS


####---------fastAPI Utils-----------------
'''FastAPI Utils'''
def _load_image(image):
    # input_image = Image.open(BytesIO(image)).convert("RGB")
    
    # contents = await image.read()
    # image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (352,352))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    torch_tensor = _totensor(image)
    torch_tensor = torch_tensor.unsqueeze(dim=0)
    return torch_tensor

def _load_mask(image):
    # contents = await image.read()
    # image = Image.open(BytesIO(contents))
    img = np.array(image)
    img = cv2.resize(img, (352,352))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
    return np.expand_dims(mask, axis=-1)

def _totensor(tensor):
    ## convert NHWC to NCHW and resacaled to [0,1]
    totensor = transforms.Compose([transforms.ToTensor()]) 
    return totensor(tensor)

####---------fastAPI Utils-----------------


'''metrics and losses'''
## IoU (Jaccard) Metric
class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection.sum()

        jaccard = ((intersection+smooth)/(union + smooth))

        return jaccard

## Dice Metric
class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return dice
    
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


'''Saving weights and paramteres'''
class Checkpoint(object):
    def __init__(self, model_name, net):
        '''dice = Dice'''
        self.best_dice = 0.
        self.folder = 'checkpoint'
        self.net = net
        self.model_name = model_name
        os.makedirs(self.folder, exist_ok=True)
    def save(self, dice, epoch=-1):
        if dice > self.best_dice:
            print(Fore.LIGHTRED_EX + 'INFO: Saving checkpoint...')
            state = {
                'net': self.net.state_dict(),
                'dice': dice,
                'epoch': epoch,
            }
            path = os.path.join(os.path.abspath(self.folder), self.model_name + '.pth')
            torch.save(state, path)
            self.best_dice = dice
    def load(self, net):
        pass
