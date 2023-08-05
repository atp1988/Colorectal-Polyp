import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms
from torchvision import transforms

import os
import random
import pandas as pd
import cv2
from glob import glob
import numpy as np
import shutil
from PIL import Image
from matplotlib import pyplot as plt
import albumentations as A
from absl import app, flags, logging
from absl.flags import FLAGS


from src.utils import _totensor, Dice
from src.loss import *
from src.net import Polyp_Net
from src.dataloader import DataLoader

flags.DEFINE_string('image_path', 'sessile-main-Kvasir-SEG/images/cju0qoxqj9q6s0835b43399p4.jpg', 'the image path')
flags.DEFINE_string('mask_path', 'sessile-main-Kvasir-SEG/masks/cju0qoxqj9q6s0835b43399p4.jpg', 'the mask path')
flags.DEFINE_string('weight_path', 'ckpt_pvt2_Decoder_2.pth', 'the weight path')
flags.DEFINE_string('device', 'cpu', 'device: cuda or cpu')
flags.DEFINE_integer('image_size', 352, 'image size')


def main(_argv):

    model = Polyp_Net()
    model.load_state_dict(torch.load(FLAGS.weight_path, map_location=FLAGS.device)['net'])
    model.eval()

    orginal_image = cv2.imread(FLAGS.image_path)
    orginal_image = cv2.resize(orginal_image, (FLAGS.image_size,FLAGS.image_size))
    image = cv2.cvtColor(orginal_image, cv2.COLOR_BGR2RGB)
    torch_tensor = _totensor(image)
    torch_tensor = torch_tensor.unsqueeze(dim=0)

    mask = cv2.imread(FLAGS.mask_path)
    mask = cv2.resize(mask, (FLAGS.image_size,FLAGS.image_size))
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
    target = np.expand_dims(mask, axis=-1)
    target = _totensor(target)
    target = target.unsqueeze(dim=0)   ## is needed to calculate Dice Score

    out = model(torch_tensor)
    out = torch.sigmoid(out)

    dice = Dice()
    dice_score = dice(out, target.type(torch.int64))
    print(f'Dice: {dice_score}')
    print('\n')
    ###========================plot pred and mask==========================================
    ## convert torch tensor to numpy array
    pred = torch.permute(out, (0,2,3,1)) ## NCHW
    pred = pred.detach().numpy()
    pred = np.squeeze(pred, axis=0)      ## HWC
    pred = pred*255
    _,pred = cv2.threshold(pred,200,255,cv2.THRESH_BINARY)
    pred = pred.astype(np.uint8)

    # mask = mask/255.
    mask = torch.permute(target, (0,2,3,1)) ## NCHW
    mask = mask.detach().numpy()
    mask = np.squeeze(mask, axis=0)      ## HWC
    mask = mask*255
    mask = mask.astype(np.uint8)

    plt.figure(figsize=(14,14))
    plt.subplot(1,3,1)
    plt.imshow(orginal_image)
    plt.title('polyp-image')
    plt.subplot(1,3,2)
    plt.imshow(mask)
    plt.title('mask')
    plt.subplot(1,3,3)
    plt.imshow(pred)
    plt.title('prediction')
    plt.show()



if __name__ == "__main__":
    app.run(main)
