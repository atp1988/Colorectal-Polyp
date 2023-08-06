# Gastrointestinal Polyp Segmentation in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/atp1988/gastrointestinal-polyp/blob/main/Polyp-Net.ipynb)

This repo provides a clean implementation of Polyp Segmentation in Pytorch on standard and real polyp datasets.

## Key Features

- [x] Executable using PyTorch 
- [x] `Polyp-Net` with pre-trained Weights
- [x] Inference example with GUI on FastAPI
- [x] GPU Accelerated 
- [x] Fully integrated with `absl-py` from [abseil.io](https://abseil.io)
- [x] Clean implementation
- [x] MIT License

![demo](https://github.com/atp1988/gastrointestinal-polyp/blob/main/predictions/pred5.png)

#### Dependency Installation

```bash
pip install -r requirements.txt
```

### Detection

```bash
python test.py --image_path data/images/xxx.jpg --mask_path data/masks/xxx.jpg 
```

![demo](https://github.com/atp1988/gastrointestinal-polyp/blob/main/predictions/pred4.png)

### Training Procedure

Before to start training, please download the backbone weights and polyp dataset from here:

- [x] Download polyp dataset includes train and test directories: [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing) and move them to the './polyp_dataset/'.

- [x] Download the pretrained PVT-Version2 model: [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing), and then move it to the './weights/' folder for initialization. 

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
  --transfrom_train: train set data transfrom
    (default: 'True')
    (a boolean)
  --transfrom_test: train set data transfrom
  --transfrom_train: data transfrom for train set
    (default: 'True')
    (a boolean)
  --transfrom_test: test set data transfrom
    (default: 'True')
    (a boolean)
  --size: image_size
    (default: '352')
    (an integer)
```
