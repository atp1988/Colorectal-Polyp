# Colorectal Polyp Segmentation in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/atp1988/gastrointestinal-polyp/blob/main/Polyp-Net.ipynb)

This repo provides a clean implementation of Polyp Segmentation in Pytorch on standard and real polyp datasets.

![demo](https://github.com/atp1988/gastrointestinal-polyp/blob/main/images/net.png)

## Key Features

- [x] Executable, Clean and simple implementation using PyTorch 
- [x] `Pyramid Vision Transformer` (PVT) as a backbone for medical segmentation task
	- [x] Multi-Scale Contextual Information, capture contextual information at different scales
	- [x] Handling of Fine Details, capture fine-grained details at various resolutions
	- [x] PVT's self-attention mechanism enables it to consider global context understanding the relationship between different regions for accurate segmentation
	- [x] Transfer Learning, using pretrained PVT models and fine tune on a polyp datasets help to leverage the knowledge acquired from diverse data sources, improve the model's segmentation performance and reduce computational costs due to fast convergence
- [x] `Atrous Spatial Pyramid Pooling` or ASPP module
	- [x] capture information at multiple spatial scales
	- [x] adjust the receptive field size to extract features in various object sizes
	- [x] prohibited from loss of fine-grained details with reduction of number of downsampler blocks
	- [x] improved boundary prediction, sharper and more accurate object masks
	- [x] computational overhead reduction 
- [x] `Convolutional Block Attention Module` or CBAM module
	- [x] highlight important spatial locations and channels within feature maps
	- [x] better object recognition with focusing on discriminative regions and ignore irrelevant information.
	- [x] better generalization, By learning to attend to relevant features being robust to trandformation, leading to better performance to unseen data
	- [x] be easily integrated into CNN models without major architectural changes
- [x] Inference example with GUI on FastAPI
- [x] Fully integrated with `absl-py` from [abseil.io](https://abseil.io)

![demo](https://github.com/atp1988/gastrointestinal-polyp/blob/main/images/pred6.png)


## How to run
#### Dependency Installation

```bash
pip install -r requirements.txt
```

### Detection

Before to test the model, please download the pretrained weight from here: [Google Drive](https://drive.google.com/uc?export=download&id=1-OBlpRqGbt3-OIgdH5JbuzChwmKWCxa8) and move it to the 'checkpoints/' directory.

```bash
python test.py --image_path data/images/xxx.jpg --mask_path data/masks/xxx.jpg 
```

![demo](https://github.com/atp1988/gastrointestinal-polyp/blob/main/images/pred4.png)

### FastApi 

you are able to watch the segmentation of a colon polyp using FastApi as a Demo. 
But you just need the pretrained weights downloading from here: [Google Drive](https://drive.google.com/uc?export=download&id=1-OBlpRqGbt3-OIgdH5JbuzChwmKWCxa8) and move it to the 'checkpoints/' directory and then run:

```bash
python server.py 
```
or you should create an docker image through below commands:

```bash
docker build -t polyp-fastapi .
docker run -p 8000:8000 -t polyp-fastapi
```

after execution, open a browser and browse `http://127.0.0.1:8000/docs`. and then push `Try it out` botton to appear `Choose File` botton, then enjoy it.

![demo](https://github.com/atp1988/gastrointestinal-polyp/blob/main/images/fastapi1.png)


## How to train

Before to start training, please download the backbone weights and polyp dataset from here:

- [x] Download polyp dataset includes train and test directories: [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing) and move them to the './polyp_dataset/' directory.

- [x] Download the pretrained PVT-Version2 model: [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing), and then move it to the './weights/' directory for initialization. 

for training with default set up please only run:
```bash
python train.py
```

#### Command Line Args Reference to change training parameters

```bash
train.py:
  --dataset: path to dataset
    (default: 'polyp_dataset')
  --weights: path to backbone weights file
    (default: './weights/pvt_v2_b2.pth')
    (an integer)
  --batch_size: batch size
    (default: '10')
  --device: cuda or cpu to process
    (default: 'cuda')
  --model_name: model name to save checkpoints
    (default: 'ckpt_polyp')
  --epochs: number of epochs
    (default: '50')
    (an integer)
  --learning_rate: learning rate
    (default: '0.0005')
    (a float number)
  --weight_decay: weight decay
    (default: '0.0001')
    (a float number)
  --transfrom_train: train dataset transfrom
    (default: 'True')
    (a boolean)
  --transfrom_test: test dataset transfrom
    (default: 'True')
    (a boolean)
  --size: image_size
    (default: '352')
    (an integer)
```
