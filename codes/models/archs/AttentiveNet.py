import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

class AttNet(nn.Module):
    def __init__(self,in_nc,out_nc):
        super(AttNet, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, 64, 5, 1, 2, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.dia_conv1 = nn.Conv2d(256, 256, 3, 1, padding=2, dilation=2, bias=False)
        self.dia_conv2 = nn.Conv2d(256, 256, 3, 1, padding=4, dilation=4, bias=False)
        self.dia_conv3 = nn.Conv2d(256, 256, 3, 1, padding=8, dilation=8, bias=False)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, output_padding=0, bias=False)
        self.avg_pool = nn.AvgPool2d(3,1,1)
        self.conv9 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, output_padding=0, bias=False)
        self.conv10 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.skip_3 = nn.Conv2d(32, out_nc, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
    def forward(self, x):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.lrelu(self.conv3(conv2))
        conv4 = self.lrelu(self.conv4(conv3))
        conv5 = self.lrelu(self.conv5(conv4))
        conv6 = self.lrelu(self.conv6(conv5))
        dia_conv1 = self.lrelu(self.dia_conv1(conv6))
        dia_conv2 = self.lrelu(self.dia_conv2(dia_conv1))
        dia_conv3 = self.lrelu(self.dia_conv3(dia_conv2))
        conv7 = self.lrelu(self.conv7(dia_conv3))
        conv8 =self.lrelu(self.conv8(conv7))
        deconv1 = self.lrelu(self.deconv1(conv8))
        # conv9 = self.lrelu(self.conv9(self.avg_pool(deconv1)))
        conv9 = self.lrelu(self.conv9(deconv1))
        # deconv_2 = self.lrelu(self.avg_pool(self.deconv2(conv9)))
        deconv_2 = self.lrelu(self.deconv2(conv9))
        conv10 = self.lrelu(self.conv10(deconv_2))
        skip_3 = torch.tanh(self.skip_3(conv10))
        # skip_3 = torch.tanh(skip_3)
        return skip_3
