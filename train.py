import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
from torchvision import transforms

# from torch.utils.tensorboard import SummaryWriter


import argparse
import os
import cv2
from glob import glob
import numpy as np
import shutil
from PIL import Image
from matplotlib import pyplot as plt
import albumentations as A
from tabulate import tabulate
from colorama import Fore, Back, Style
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS

from utils import *
from loss import *
from net import Polyp_Net
from dataloader import DataLoader

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

flags.DEFINE_string('dataset', '', 'path to dataset')

# flags.DEFINE_string('weights', './checkpoints/yolov3.tf','path to weights file')

flags.DEFINE_string('model_name', 'ckpt_polyp', 'name of the model to save checkpoints')
flags.DEFINE_string('device', 'cpu', 'device: cuda or cpu')

flags.DEFINE_boolean('transform_train', True, 'train set transfrom ')
flags.DEFINE_boolean('transform_test', False, 'test set transfrom ')
flags.DEFINE_integer('image_size', 352, 'image size')
flags.DEFINE_integer('epochs', 50, 'number of epochs')\
flags.DEFINE_integer('batch_size', 10, 'batch size')
flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
flags.DEFINE_float('weight_decay', 1e-4, 'weight_decay')
flags.DEFINE_integer('num_classes', 1, 'number of classes')

def main(_argv):

    # Training
    def train(epoch):
        net.train()
        loss_total = AverageMeter()
        dice_total = AverageMeter()
        iou_total = AverageMeter()

        dice = Dice()
        iou = IoU()

        for batch_idx, (inputs, masks) in enumerate(train_loader()):
            inputs, masks = inputs.to(FLAGS.device), masks.to(FLAGS.device)
            optimizer.zero_grad()

            out = net(inputs)

            loss = structure_loss(out, masks)
            loss.backward()
            # clip_gradient(optimizer, 0.5)
            optimizer.step()

            loss_total.update(loss)

            ## calculte Dice metric
            out = F.sigmoid(out)
            train_dice = dice(out, masks.type(torch.int64))
            dice_total.update(train_dice)

            train_iou = iou(out, masks.type(torch.int64))
            iou_total.update(train_iou)

        # writer.add_scalar('Loss/train', loss_total.avg.item(), epoch)
        # writer.add_scalar('Dice/train', dice_total.avg.item(), epoch)

        print(f'Train: Epoch[{epoch}]:',
                    f'Loss:{loss_total.avg:0.4}',
                    f'Dice:{dice_total.avg:0.4}',
                    f'IoU:{iou_total.avg:0.4}',
                    )


    def validation(epoch, checkpoint):
        net.eval()
        test_datasets = ['test', 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
        data=[]
        for dataset_name in test_datasets:
            ## for every dataset the Averagemeter Reset and all the items become clear
            val_loss_total = AverageMeter()
            val_dice_total = AverageMeter()
            val_iou_total = AverageMeter()

            dice = Dice()
            iou = IoU()

            val_loader = DataLoader(dataset_name=dataset_name,
                                    path = FLAGS.dataset,
                                    mode='test',
                                    transform = FLAGS.transform_test,
                                    batch_size=FLAGS.batch_size,
                                    image_size=(FLAGS.image_size,FLAGS.image_size)
                                    )
            
            with torch.no_grad():
                for batch_idx, (inputs, masks) in enumerate(val_loader()):
                    inputs, masks = inputs.to(FLAGS.device), masks.to(FLAGS.device)
                    out = net(inputs)

                    if dataset_name =='test':

                        val_loss = structure_loss(out, masks)
                        val_loss_total.update(val_loss)

                    out = F.sigmoid(out)

                    val_dice = dice(out, masks.type(torch.int64))
                    val_dice_total.update(val_dice)

                    val_iou = iou(out, masks.type(torch.int64))
                    val_iou_total.update(val_iou)

                if dataset_name=='test':
                    # writer.add_scalar('val_Loss/test', val_loss_total.avg.item(), epoch)
                    # writer.add_scalar('val_Dice/test', val_dice_total.avg.item(), epoch)
                    lr = scheduler.optimizer.param_groups[0]['lr']

                    print(f'Valid: Epoch[{epoch}]:',
                                f'loss:{val_loss_total.avg:.4}',
                                f'Dice:{val_dice_total.avg:0.4}',
                                f'IoU:{val_iou_total.avg:0.4}',
                                f'lr:{lr:0.1}',
                        )

                    checkpoint.save(val_dice_total.avg, epoch=epoch)
                    print(Fore.BLACK + '>>---------------------------------------------------<<')


                else:
                    data.append([dataset_name, torch.round(val_dice_total.avg, decimals=4), torch.round(val_iou_total.avg, decimals=4)])

        print(tabulate(data, headers=["Dataset", "Dice", "IoU"]))

        print()
        scheduler.step(val_loss_total.avg)

    ##=======================================================================================================================
    print('Initialzing and Training Process Started...')

    # writer = SummaryWriter()
    net = Polyp_Net(num_classes=FLAGS.num_classes)
    net = net.to(FLAGS.device)

    checkpoint = Checkpoint(FLAGS.model_name, net)
    
    optimizer = optim.AdamW(net.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', 
                                                        factor=0.2, 
                                                        patience=3,
                                                        min_lr=0, 
                                                        verbose=False)
    
    train_loader = DataLoader(dataset_name='',
                              path = FLAGS.dataset,
                              mode='train',
                              transform = FLAGS.transform_train,
                              batch_size=FLAGS.batch_size,
                              image_size=(FLAGS.image_size,FLAGS.image_size)
                            )


    start, end = 1, FLAGS.epochs
    for epoch in range(start, end+1):
        train(epoch)
        validation(epoch, checkpoint)

    # writer.close()

if __name__ == "__main__":
    app.run(main)
