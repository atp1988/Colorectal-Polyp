import torch
import torch.nn as nn
import torch.nn.functional as F

class CAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CAM, self).__init__()
        ## CA module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        ## SA module
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        x = self.sigmoid(out) * x

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)

        return self.sigmoid(y) * x

class Attention_Gate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_Gate,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class conv_block(nn.Module):
    def __init__(self, filter_in, filter_out, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm2d(filter_out, affine=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)

class fire_module(nn.Module):
    def __init__(self, filter_in, squeeze=16, expand=64):
        super().__init__()
        self.conv = conv_block(filter_in, squeeze, kernel_size=1, padding=0)
        self.left = conv_block(squeeze, expand, kernel_size=1, padding=0)
        self.right = conv_block(squeeze, expand, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        x = F.relu(torch.cat([left, right], 1))
        return x

class residual_fire_module(nn.Module):
    def __init__(self, filter_in, squeeze=16, expand=64, dilation=1):
        super().__init__()
        self.conv = conv_block(filter_in, squeeze, kernel_size=1, padding=0)
        self.left = conv_block(squeeze, expand, kernel_size=1, padding=0, dilation=dilation)
        self.right = conv_block(squeeze, expand, kernel_size=3, padding=1, dilation=dilation)
        self.bypass = conv_block(filter_in, 2*expand, kernel_size=1, padding=0)
        self.up = nn.Upsample(scale_factor=2)
    def forward(self, xi):
        x = self.conv(xi)
        left = self.left(x)
        right = self.right(x)
        x1 = F.relu(torch.cat([left, right], 1))
        x2 = self.bypass(xi)
        y = torch.add(x1, x2)
        return self.up(y)
    

if __name__ == "__main__":
    print('file successfully executed.')