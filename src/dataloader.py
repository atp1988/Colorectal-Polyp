import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms
from torchvision import transforms

import cv2
from glob import glob
import numpy as np
import albumentations as A

class Polyp_Dataset(torch.utils.data.Dataset):

    def __init__(self, image_list, mask_list, transform, image_size):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform
        self.image_size = image_size
    def __len__(self):
        return len(self.image_list)


    def _load_image(self, id):
        image_id = self.image_list[id]
        img = cv2.imread(image_id)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, image_id

    def _load_mask(self, id):
        mask_id = self.mask_list[id]
        img = cv2.imread(mask_id)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
        return np.expand_dims(mask, axis=-1), mask_id

    def _totensor(self, tensor):
        totensor = transforms.Compose([transforms.ToTensor()]) ## convert NHWC to NCHW and resacaled to [0,1]
        return totensor(tensor)

    def __getitem__(self, index: int):
        image, image_id = self._load_image(index)
        mask, mask_id = self._load_mask(index)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = self._totensor(image)
        mask = self._totensor(mask)

        return image, mask
    

class DataLoader():
    def __init__(self, dataset_name, path, mode, transform, batch_size, image_size):
        self.dataset_name = dataset_name
        self.path = path
        self.mode = mode
        self.transform = transform
        self.batch_size = batch_size
        self.image_size = image_size

    def _transforms(self, status=None):
        if status == 'train':
            transform = A.Compose([
                A.Affine(scale=(.8, 1.2), always_apply=False),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(shear=(-20, 20), always_apply=False, p=0.5),
                A.Affine(rotate=(-45, 45), always_apply=False, p=0.5),
                A.CLAHE(2, always_apply=False, p=0.5),
                # A.RandomResizedCrop(height=img_size[0], width=img_size[1], p=.3, always_apply=False)

            ])

        elif status == 'test':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.Affine(scale=(1, 1.3), always_apply=False),
            ])
        else:
            transform = A.Compose([
            ])

        return transform

    def __call__(self):
        if self.mode == 'train':
            image_path = sorted(glob(self.path+'TrainDataset/images/*'))
            target_path = sorted(glob(self.path+'TrainDataset/masks/*'))
        else:
            image_path = sorted(glob(self.path+'TestDataset/'+self.dataset_name+'/images/*'))
            target_path = sorted(glob(self.path+'TestDataset/'+self.dataset_name+'/masks/*'))

        if self.transform:
            transforms = self._transforms(status=self.mode)
            # if self.mode == 'train':
            #     print(f'transform on train set...')
            # elif self.mode == 'test':
            #     print(f'transform on val set...')
        else:
            transforms=None
            # print(f'transform skipped for {self.mode} set...')

        dataset = Polyp_Dataset(image_path, target_path, transforms, self.image_size)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=1,
                                                    collate_fn=None,
                                                  )

        return data_loader