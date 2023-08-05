
import torch
import torch.nn as nn
import torch.nn.functional as F

from absl import app, flags, logging
from absl.flags import FLAGS

from pvt import *
from modules import *


class Polyp_Net(nn.Module):
    def __init__(self,
                backbone_weights=None,
                ch=[64,128,320,512], ## channel
                ):
        super(Polyp_Net ,self).__init__()

        self.backbone = pvt_v2_b2()
        if backbone_weights is not None:
            save_model = torch.load(backbone_weights)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)


        self.fire3 = fire_module(ch[3], ch[3]//4, ch[3]//2)
        self.cam3 = CAM(in_planes=ch[3])
        self.resfire3_1 = residual_fire_module(ch[3], ch[3]//4, ch[2]//2)
        self.resfire3_2 = residual_fire_module(ch[2], ch[2]//4, ch[1]//2)
        self.resfire3_3 = residual_fire_module(ch[1], ch[1]//4, ch[0]//2)

        self.fire2 = fire_module(ch[2], ch[2]//4, ch[2]//2)
        self.cam2 = CAM(in_planes=ch[2])
        self.resfire2_1 = residual_fire_module(ch[2], ch[2]//4, ch[1]//2)
        self.resfire2_2 = residual_fire_module(ch[1], ch[1]//4, ch[0]//2)

        self.fire1 = fire_module(ch[1], ch[1]//4, ch[1]//2)
        self.cam1 = CAM(in_planes=ch[1])
        self.resfire1_1 = residual_fire_module(ch[1], ch[1]//4, ch[0]//2)


        self.conv3x3 = conv_block(ch[0], ch[0], kernel_size=3, padding=1)

        self.AG3 = Attention_Gate(F_g=ch[2], F_l=ch[2], F_int=ch[2])
        self.AG2 = Attention_Gate(F_g=ch[1], F_l=ch[1], F_int=ch[1])
        self.AG1 = Attention_Gate(F_g=ch[0], F_l=ch[0], F_int=ch[0])

        self.resfire_ag3 = residual_fire_module(ch[2], ch[2]//4, ch[1]//2)
        self.resfire_ag2 = residual_fire_module(2*ch[1], 2*ch[1]//4, ch[0]//2)
        self.resfire_ag1 = residual_fire_module(ch[1], ch[0]//4, ch[0]//2)

        # self.conv1 = conv_block(ch[1], ch[0], kernel_size=3, padding=0)
        self.head = nn.Conv2d(ch[1], 1, kernel_size=1, padding=0)

        self.up4x = nn.Upsample(scale_factor=4)
        self.up = nn.Upsample(scale_factor=2)


    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x0 = pvt[0]
        x1 = pvt[1]
        x2 = pvt[2] ## 20**20*320
        x3 = pvt[3] ## 10*10*512

#######

        x = self.fire3(x3)
        x = self.cam3(x)
        x3_1 = self.resfire3_1(x)
        x = self.resfire3_2(x3_1)
        y3 = self.resfire3_3(x)

        x = self.fire2(x2)
        x = self.cam2(x)
        x2_1 = self.resfire2_1(x)
        y2 = self.resfire2_2(x2_1)

        x = self.fire1(x1)
        x = self.cam1(x)
        y1 = x1_1 = self.resfire1_1(x)

        y_x = self.conv3x3(self.conv3x3(self.conv3x3(x0)*y1)*y2)*y3

        ag3 = self.AG3(x3_1, x2)
        x = self.resfire_ag3(ag3)

        ag2 = self.AG2(x2_1, x1)
        x = torch.cat([x, ag2], 1)
        x = self.resfire_ag2(x)
        ag1 = self.AG1(x1_1, x0)
        x = torch.cat([x, ag1], 1)

        x = self.resfire_ag1(x)
        x = torch.cat([x, self.up(y_x)], 1)

        head = self.up(self.head(x))

        return head
