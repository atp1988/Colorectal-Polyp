from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from io import BytesIO

import uvicorn


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import _totensor
from net import Polyp_Net


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model
PATH = 'checkpoints/ckpt_pvt2_Decoder_2.pth'
model = Polyp_Net()
model.load_state_dict(torch.load(PATH, map_location=device)['net'])
model.eval()

app = FastAPI()


def double_plot_images(image1: Image.Image) -> Image.Image:
    image1 = image1.resize((352, 352))
    total_width = image1.width + image1.width

    new_image = Image.new("RGB", (total_width, image1.height))
    new_image.paste(image1, (0, 0))
    
    image = np.array(image1)
    # image = cv2.resize(image, (352,352))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    torch_tensor = _totensor(image)
    torch_tensor = torch_tensor.unsqueeze(dim=0)  ## NCHW
    out = model(torch_tensor)
    out = torch.sigmoid(out)

    ## convert torch tensor to numpy array
    pred = torch.permute(out, (0,2,3,1)) ## NCHW
    pred = pred.detach().numpy()
    pred = np.squeeze(pred, axis=0)      ## HWC
    pred = pred*255
    _,pred = cv2.threshold(pred,200,255,cv2.THRESH_BINARY)
    pred = pred.astype(np.uint8)

    pred = np.array(pred)

    segmented_image = Image.fromarray(pred)
    new_image.paste(segmented_image, (image1.width, 0))

    return new_image




@app.post("/double_plot_images/")
async def upload_images(image1: UploadFile = File(...)):
    image1_path = f"temp_{image1.filename}"

    with open(image1_path, "wb") as f1:
        f1.write(image1.file.read())

    pil_image1 = Image.open(image1_path)
    # pil_image1 = pil_image1.resize((352,352))
    
    plotted_image = double_plot_images(pil_image1)

    # Save the plotted image temporarily
    output_image_path = "output_plot.jpg"
    plotted_image.save(output_image_path, "JPEG")

    return FileResponse(output_image_path, media_type="image/jpeg")


if __name__ == "__main__":
    '''run fastApi with uvicorn'''
    uvicorn.run("server:app", port=8000, log_level="info")
