import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from models.archs.feature_fusion import SELayer, SELayerPlus, SELayerPlusAdd, GAU, CAB, CABPlus

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

# 修复
# class RRDBNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(RRDBNet, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
#         # RRDB_block_f_SE = functools.partial(RRDB_SE, nf=nf, gc=gc)
#         # RRDB_block_f_CABPlus = functools.partial(RRDB_CABPlus, nf=nf, gc=gc)
#
#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = arch_util.make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         #### upsampling
#         # self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         # self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         # self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         # self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         #### 特征融合后卷积
#         # self.catconv1 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
#         ## SElayer
#         # self.se = SELayer(nf)
#         # self.sePlus = SELayerPlus(nf)
#         # self.sePlusAdd = SELayerPlusAdd(nf)
#         # ##GAU
#         # self.GAU = GAU(nf, nf, upsample=False)
#         ###CAB
#         # self.CAB = CAB(nf*2, nf)
#         # self.CABPlus = CABPlus(nf*2, nf)
#         ### 亚像素卷积层
#         # self.psconv1 = nn.Conv2d(nf, 4, 3, 1, 1, bias=True)
#         # self.pixel = nn.PixelShuffle(2)
#         ### 反卷积
#         # self.deconv1 = nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
#         ### EDSR
#         # self.edsr = arch_util.make_layer(ResBlock, nb)
#         ###不同卷积核
#         self.conv3x3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv5x5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
#         self.conv7x7 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
#         self.conv357 = nn.Conv2d(nf*3, nf, 3, 1, 1, bias=True)
#         self.conv_inpainting = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last_inpainting = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
#         # self.conv1x1 = nn.Conv2d(nf*3, nf, 1, 1, 0, bias=True)
#         ### 加入SElayer的RRDB
#         # self.RRDB_trunk_SE = arch_util.make_layer(RRDB_block_f_SE, nb)
#         ### 加入CABPlus的RRDB
#         # self.RRDB_trunk_CABPlus = arch_util.make_layer(RRDB_block_f_CABPlus, nb)
#     def forward(self, x):
#         fea = self.conv_first(x)
#         # EDSR
#         # trunk = self.trunk_conv(self.edsr(fea))
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         ##SE实验
#         # trunk = self.trunk_conv(self.RRDB_trunk_SE(fea))
#         ##CABPlus
#         # trunk = self.trunk_conv(self.RRDB_trunk_CABPlus(fea))
#
#
#         #### 修复网络实验内容
#         ##
#         # fea = fea + trunk
#         # fea = trunk
#         # fea = self.lrelu(self.upconv1(fea))
#         ##特征重建方式
#         # fea = fea + trunk
#         # fea3 = self.lrelu(self.conv3x3(fea))
#         # fea5 = self.lrelu(self.conv5x5(fea))
#         # fea7 = self.lrelu(self.conv7x7(fea))
#         # fea_concat = torch.cat((fea3, fea5, fea7), 1)
#         # fea = self.lrelu(self.conv357(fea_concat))
#         ### 特征融合
#         ### 修复网络
#         fea = fea + trunk
#         fea3 = self.lrelu(self.conv3x3(fea))
#         fea5 = self.lrelu(self.conv5x5(fea))
#         fea7 = self.lrelu(self.conv7x7(fea))
#         fea_concat = torch.cat((fea3, fea5, fea7), 1)
#         fea_inpaiting = self.lrelu(self.conv357(fea_concat))
#         out = self.conv_last_inpainting(self.lrelu(self.conv_inpainting(fea_inpaiting)))
#
#         # fea = self.lrelu(self.conv1x1(fea_concat))
#         # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         ##concat
#         # fea = torch.cat((fea, trunk), 1)
#         # fea = self.lrelu(self.catconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         ##SElayer
#         # trunk = self.se(trunk)
#         ##SElayerPLus
#         # trunk = self.sePlus(trunk)
#         # trunk = self.sePlusAdd(trunk)
#         # fea = fea + trunk
#
#         ###GAU
#         # fea = self.GAU(trunk, fea)
#         ###CAB
#         # fea= self.CAB(trunk, fea)
#         # fea = self.CABPlus(trunk, fea)
#         # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         ### 上采样方式
#         # fea = fea + trunk
#         # fea = self.lrelu(self.HRconv(fea))
#         # out = self.lrelu(self.pixel(self.lrelu(self.psconv1(fea))))
#
#         # fea = fea + trunk
#         # fea = self.lrelu(self.deconv1(fea))
#         # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         # out = self.conv_last(self.lrelu(self.HRconv(fea)))
#
#         # out = torch.tanh(out)
#         return out

# 超分辨
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        # RRDB_block_f_SE = functools.partial(RRDB_SE, nf=nf, gc=gc)
        # RRDB_block_f_CABPlus = functools.partial(RRDB_CABPlus, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = arch_util.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #### 特征融合后卷积
        # self.catconv1 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        ## SElayer
        # self.se = SELayer(nf)
        # self.sePlus = SELayerPlus(nf)
        # self.sePlusAdd = SELayerPlusAdd(nf)
        # ##GAU
        # self.GAU = GAU(nf, nf, upsample=False)
        ###CAB
        # self.CAB = CAB(nf*2, nf)
        # self.CABPlus = CABPlus(nf*2, nf)
        ### 亚像素卷积层
        # self.psconv1 = nn.Conv2d(nf, 4, 3, 1, 1, bias=True)
        # self.pixel = nn.PixelShuffle(2)
        ### 反卷积
        # self.deconv1 = nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        ### EDSR
        # self.edsr = arch_util.make_layer(ResBlock, nb)
        ###不同卷积核
        # self.conv3x3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.conv5x5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
        # self.conv7x7 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
        # self.conv357 = nn.Conv2d(nf*3, nf, 3, 1, 1, bias=True)
        # self.conv_inpainting = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.conv_last_inpainting = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        # self.conv1x1 = nn.Conv2d(nf*3, nf, 1, 1, 0, bias=True)
        ### 加入SElayer的RRDB
        # self.RRDB_trunk_SE = arch_util.make_layer(RRDB_block_f_SE, nb)
        ### 加入CABPlus的RRDB
        # self.RRDB_trunk_CABPlus = arch_util.make_layer(RRDB_block_f_CABPlus, nb)
    def forward(self, x):
        fea = self.conv_first(x)
        # EDSR
        # trunk = self.trunk_conv(self.edsr(fea))
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        ##SE实验
        # trunk = self.trunk_conv(self.RRDB_trunk_SE(fea))
        ##CABPlus
        # trunk = self.trunk_conv(self.RRDB_trunk_CABPlus(fea))


        #### 修复网络实验内容
        ##
        # fea = fea + trunk
        # fea = trunk
        # fea = self.lrelu(self.upconv1(fea))
        ##特征重建方式
        # fea = fea + trunk
        # fea3 = self.lrelu(self.conv3x3(fea))
        # fea5 = self.lrelu(self.conv5x5(fea))
        # fea7 = self.lrelu(self.conv7x7(fea))
        # fea_concat = torch.cat((fea3, fea5, fea7), 1)
        # fea = self.lrelu(self.conv357(fea_concat))
        ### 特征融合
        ### 修复网络
        fea = fea + trunk
        # fea3 = self.lrelu(self.conv3x3(fea))
        # fea5 = self.lrelu(self.conv5x5(fea))
        # fea7 = self.lrelu(self.conv7x7(fea))
        # fea_concat = torch.cat((fea3, fea5, fea7), 1)
        # fea_inpaiting = self.lrelu(self.conv357(fea_concat))
        # out = self.conv_last_inpainting(self.lrelu(self.conv_inpainting(fea_inpaiting)))

        # fea = self.lrelu(self.conv1x1(fea_concat))
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        ##concat
        # fea = torch.cat((fea, trunk), 1)
        # fea = self.lrelu(self.catconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        ##SElayer
        # trunk = self.se(trunk)
        ##SElayerPLus
        # trunk = self.sePlus(trunk)
        # trunk = self.sePlusAdd(trunk)
        # fea = fea + trunk

        ###GAU
        # fea = self.GAU(trunk, fea)
        ###CAB
        # fea= self.CAB(trunk, fea)
        # fea = self.CABPlus(trunk, fea)
        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        ### 上采样方式
        # fea = fea + trunk
        # fea = self.lrelu(self.HRconv(fea))
        # out = self.lrelu(self.pixel(self.lrelu(self.psconv1(fea))))

        # fea = fea + trunk
        # fea = self.lrelu(self.deconv1(fea))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        # out = torch.tanh(out)
        return out

#修复
# class RRDBNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(RRDBNet, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
#
#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = arch_util.make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         # #### upsampling
#         # self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         # self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         # self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         # self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
#         ##修复部分
#         self.conv3x3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv5x5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
#         self.conv7x7 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
#         self.conv357 = nn.Conv2d(nf*3, nf, 3, 1, 1, bias=True)
#         self.conv_inpainting = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last_inpainting = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#         #### 特征融合后卷积
#         # self.catconv1 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
#         # self.se = SELayer(nf, )
#         ### 亚像素卷积层
#         # self.psconv1 = nn.Conv2d(nf, 4, 3, 1, 1, bias=True)
#         # self.pixel = nn.PixelShuffle(2)
#         ### 反卷积
#         # self.deconv1 = nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
#         ### EDSR
#         # self.edsr = arch_util.make_layer(ResBlock, nb)
#     def forward(self, x):
#         fea = self.conv_first(x)
#         # EDSR
#         # trunk = self.trunk_conv(self.edsr(fea))
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         #### 修复网络实验内容
#         fea = fea + trunk
#         fea3 = self.lrelu(self.conv3x3(fea))
#         fea5 = self.lrelu(self.conv5x5(fea))
#         fea7 = self.lrelu(self.conv7x7(fea))
#         fea_concat = torch.cat((fea3, fea5, fea7), 1)
#         fea_inpaiting = self.lrelu(self.conv357(fea_concat))
#         out = self.conv_last_inpainting(self.lrelu(self.conv_inpainting(fea_inpaiting)))
#         # fea = trunk
#         # fea = self.lrelu(self.upconv1(fea))
#
#         #### 超分辨网络实验内容
#         ### 特征融合
#         # fea = fea + trunk
#         # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         #
#         # fea = torch.cat((fea, trunk), 1)
#         # fea = self.lrelu(self.catconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         #
#         # trunk = self.se(trunk)
#         # fea = fea + trunk
#         # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         ### 上采样方式
#         # fea = fea + trunk
#         # fea = self.lrelu(self.HRconv(fea))
#         # out = self.lrelu(self.pixel(self.lrelu(self.psconv1(fea))))
#
#         # fea = fea + trunk
#         # fea = self.lrelu(self.deconv1(fea))
#         # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         # out = self.conv_last(self.lrelu(self.HRconv(fea)))
#         return out
#         # return out, fea_inpaiting, fea3, fea5, fea7

class ResidualDenseBlock_5C_SE(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C_SE, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.se = SELayerPlus(nf)
        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.se(x5)
        return x5 + x

class RRDB_SE(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB_SE, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_SE(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C_SE(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C_SE(nf, gc)
        self.se = SELayerPlus(nf)
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = self.se(out)
        return out + x

class ResidualDenseBlock_5C_CABPlus(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C_CABPlus, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.CABPlus_RDB = CABPlus(nf*2, nf)
        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.CABPlus_RDB(x5, x)
        return x5

class RRDB_CABPlus(nn.Module):

    def __init__(self, nf, gc=32):
        super(RRDB_CABPlus, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_CABPlus(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C_CABPlus(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C_CABPlus(nf, gc)
        self.CABPlus_RRDB = CABPlus(nf*2, nf)
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = self.CABPlus_RRDB(out, x)
        return out