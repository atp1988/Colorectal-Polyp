
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

from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from starlette.responses import Response


from net import Polyp_Net
from utils import _load_image, _load_mask, _totensor, Dice
##-------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## load model
PATH = 'ckpt_pvt2_Decoder_2.pth'
model = Polyp_Net(num_classes=1)
model.load_state_dict(torch.load(PATH, map_location=device)['net'])
model.eval()


##---------------------------------------------------------------
app = FastAPI()

## mask
@app.post('/GroundTruth')
async def predict(file: UploadFile=File(...)):
    
    contents = await file.read()
    mask = Image.open(BytesIO(contents)).convert("RGB")
    mask = mask.resize((352,352))
    # segmented_image = Image.fromarray(pred)
    bytes_io = BytesIO()
    mask.save(bytes_io, format="JPEG")

    return Response(bytes_io.getvalue(), media_type="image/jpeg")


@app.post('/Prediction')
async def predict(file: UploadFile=File(...)):
    
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    torch_tensor = _load_image(image)   ## NCHW
    out = model(torch_tensor)
    out = torch.sigmoid(out)

    # dice = Dice()
    # dice_score = dice(out, target.type(torch.int64))

    # print(f'Dice: {dice_score}')
    # print('\n')

    ## convert torch tensor to numpy array
    pred = torch.permute(out, (0,2,3,1)) ## NCHW
    pred = pred.detach().numpy()
    pred = np.squeeze(pred, axis=0)      ## HWC
    pred = pred*255
    _,pred = cv2.threshold(pred,200,255,cv2.THRESH_BINARY)
    pred = pred.astype(np.uint8)

    pred = np.array(pred)

    segmented_image = Image.fromarray(pred)
    bytes_io = BytesIO()
    segmented_image.save(bytes_io, format="JPEG")

    return Response(bytes_io.getvalue(), media_type="image/jpeg")

###----------------------------------------------------------------------
@app.post('/double-plot')
async def predict(files: list[UploadFile]):
    ## image
    for j, file in enumerate(files):
        if j==0:
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")

            torch_tensor = _load_image(image)   ## NCHW
            out = model(torch_tensor)
            out = torch.sigmoid(out)

            pred = torch.permute(out, (0,2,3,1)) ## NCHW
            pred = pred.detach().numpy()
            pred = np.squeeze(pred, axis=0)      ## HWC
            pred = pred*255
            _,pred = cv2.threshold(pred,200,255,cv2.THRESH_BINARY)
            pred = pred.astype(np.uint8)

            segmented_image = Image.fromarray(pred)
            bytes_io_image = BytesIO()
            segmented_image.save(bytes_io_image, format="JPEG")
            response1 = Response(bytes_io_image.getvalue(), media_type="image/jpeg")

        else:
            ## mask
            contents = await file.read()
            mask = Image.open(BytesIO(contents)).convert("RGB")

            segmented_mask = Image.fromarray(np.array(mask))
            bytes_io_mask = BytesIO()
            segmented_mask.save(bytes_io_mask, format="JPEG")
            response2 = Response(bytes_io_mask.getvalue(), media_type="image/jpeg")

    return response1
