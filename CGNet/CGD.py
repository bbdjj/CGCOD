# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
# -*- coding: utf-8 -*-
import numpy as np
from .lib import *


ConvBR
from einops import rearrange


class MultiScaleAttentionModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiScaleAttentionModule, self).__init__()
        self.conv1 = ConvBR(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBR(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBR(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBR(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBR(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBR(4 * out_dim, out_dim, 3, 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(in_dim // 4, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        groups = torch.chunk(x, 4, dim=1)
        b, c, w, h = groups[0].shape
        i = 0
        attention_weights = [self.fc(group.view(b, w * h, -1)).view(b, -1, w, h) for group in groups]
        conv1 = self.conv1(x) * attention_weights[0]
        conv2 = self.conv2(x) * attention_weights[1]
        conv3 = self.conv3(x) * attention_weights[2]
        conv4 = self.conv4(x) * attention_weights[3]

        return self.fuse(torch.cat((conv1, conv2, conv3, conv4), 1))


class CAM(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(in_dim, in_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(in_dim * 2, in_dim, 3, padding=1)
        self.conv = nn.ModuleList([nn.Conv2d(in_dim, in_dim, 3, padding=1) for i in range(8)])
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 16, in_dim, bias=False),
            nn.Sigmoid()
        )
        self.MAM = MultiScaleAttentionModule(in_dim, in_dim)

    def forward(self, attn, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x_1_2 = x1 + x2
        x_1_2 = self.conv3(x_1_2) * attn
        attn = self.softmax(x_1_2)
        x1 = self.conv[0](x1 * attn)
        x2 = self.conv[1](x2 * attn)
        attn = torch.cat((x1, x2), 1)
        b, c, w, h = attn.shape
        attn = self.fc(attn.view(b, -1, c)).view(b, -1, h, w)
        x1 = self.conv[2](x1 + attn)
        x2 = self.conv[3](x2 + attn)
        xx = torch.cat([x1, x2], 1)
        xx = self.conv4(xx)

        x = self.MAM(xx)

        return x, attn


def rescale_2x(x: torch.Tensor, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64, fl=None):
        super(Network, self).__init__()
        self.rfb2_1 = RFB_modified(fl[1], channel)
        self.rfb3_1 = RFB_modified(fl[2], channel)
        self.rfb4_1 = RFB_modified(fl[3], channel)
        self.rfb1_1 = RFB_modified(fl[0], channel)
        self.CAM1 = CAM(channel)
        self.CAM2 = CAM(channel)
        self.CAM3 = CAM(channel)
        self.conv = BasicConv2d(fl[3], channel, 3, padding=1)
        self.hid_dim = channel
        self.heads = nn.ModuleList([LNConvAct(self.hid_dim, 1, 3, 1, 1, act_name="idy") for _ in range(4)])
        self.stems = nn.ModuleList([nn.Sequential(LNConvAct(self.hid_dim, self.hid_dim, 3, 1, 1, act_name="relu"),
                                                  nn.Conv2d(self.hid_dim, self.hid_dim, 3, 1, 1)) for _ in range(5)])

    def forward(self, x1, x2, x3, x4, xx1):
        x1 = self.rfb1_1(x1)
        x2 = self.rfb2_1(x2)
        x3 = self.rfb3_1(x3)
        x4 = self.rfb4_1(x4)
        xx1 = self.conv(xx1)
        xx = xx1
        x3, xx = self.CAM1(rescale_2x(xx), x3, rescale_2x(x4))
        x2, xx = self.CAM2(rescale_2x(xx), x2, rescale_2x(x3))
        x1, xx = self.CAM3(rescale_2x(xx), x1, rescale_2x(x2))

        fx = [x3, x2, x1]

        for k in range(2):
            fer = self.stems[0](xx1 + x4)
            x4 = fer
            i = 1
            for x in fx:
                fer = rescale_2x(fer) + x
                fer = self.stems[i](fer)
                fx[i - 1] = fer
                i += 1

        x1 = self.heads[0](fx[2])
        x2 = self.heads[1](fx[1])
        x3 = self.heads[2](fx[0])
        x4 = self.heads[3](x4)
        return x1, x2, x3, x4