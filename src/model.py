import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            Conv3x3GNReLU(in_channels, channels),
            Conv3x3GNReLU(channels, out_channels),
        )

        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.block(x)
        x = self.SCSE(x)

        return x


class UResNet34(nn.Module):
    def __init__(self, classes=4, pretrained=True):
        super(UResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encoder2 = nn.Sequential(self.resnet.layer1, SCSEBlock(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2, SCSEBlock(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3, SCSEBlock(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4, SCSEBlock(512))

        self.decoder5 = DecoderBlock(512 + 256, 256, 64)
        self.decoder4 = DecoderBlock(64 + 128, 128, 64)
        self.decoder3 = DecoderBlock(64 + 64, 64, 64)
        self.decoder2 = DecoderBlock(64 + 64, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        self.output = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(32, classes, kernel_size=1, padding=0))

    def forward(self, x):
        encode1 = self.encoder1(x)  # 3x256x1600 ==> 64x128x800 (1/4)
        encode2 = self.encoder2(self.resnet.maxpool(encode1))  # 64x128x800 ==> 64x64x400 (1/8)
        encode3 = self.encoder3(encode2)  # 64x64x400 ==> 128x32x200 (1/16)
        encode4 = self.encoder4(encode3)  # 128x32x200 ==> 256x16x100 (1/32)
        encode5 = self.encoder5(encode4)  # 256x16x100 ==> 512x8x50 (1/64)

        decode5 = self.decoder5(encode5, encode4)  # 512x8x50 + 256x16x100 ==> 64x16x100
        decode4 = self.decoder4(decode5, encode3)  # 64x16x100 + 128x32x200 ==> 64x32x200
        decode3 = self.decoder3(decode4, encode2)  # 64x32x200 + 64x64x400 ==> 64x64x400
        decode2 = self.decoder2(decode3, encode1)  # 64x64x400 + 64x128x800 ==> 64x128x800
        x = self.decoder1(decode2, None)

        x = self.output(x)

        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.scse = SCSEBlock(pyramid_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        x = self.scse(x)

        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)
        self.scse = SCSEBlock(out_channels)

    def forward(self, x):
        x = self.block(x)
        x = self.scse(x)

        return x


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class FPResNet34(nn.Module):
    def __init__(self, classes=4, pretrained=True):
        super(FPResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encoder2 = nn.Sequential(self.resnet.layer1, SCSEBlock(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2, SCSEBlock(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3, SCSEBlock(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4, SCSEBlock(512))

        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1),
                                     Downsample(channels=64, filt_size=3, stride=2))

        self.conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1))

        self.p4 = FPNBlock(256, 256)
        self.p3 = FPNBlock(256, 128)
        self.p2 = FPNBlock(256, 64)

        self.s5 = SegmentationBlock(256, 64, n_upsamples=3)
        self.s4 = SegmentationBlock(256, 64, n_upsamples=2)
        self.s3 = SegmentationBlock(256, 64, n_upsamples=1)
        self.s2 = SegmentationBlock(256, 64, n_upsamples=0)

        self.output = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(64),
                                    nn.Conv2d(64, classes, kernel_size=1, padding=0))

    def forward(self, x):
        encode1 = self.encoder1(x)  # 3x256x1600 ==> 64x128x800 (1/4)
        # encode2 = self.encoder2(self.resnet.maxpool(encode1))  # 64x128x800 ==> 64x64x400 (1/8)
        encode2 = self.encoder2(self.maxpool(encode1))
        encode3 = self.encoder3(encode2)  # 64x64x400 ==> 128x32x200 (1/16)
        encode4 = self.encoder4(encode3)  # 128x32x200 ==> 256x16x100 (1/32)
        encode5 = self.encoder5(encode4)  # 256x16x100 ==> 512x8x50 (1/64)

        decode5 = self.conv1(encode5)  # 256x8x50
        decode4 = self.p4(decode5, encode4)  # 256x8x50 + 256x16x100 ==> 256x16x100
        decode3 = self.p3(decode4, encode3)  # 256x16x100 + 128x32x200 ==> 256x32x200
        decode2 = self.p2(decode3, encode2)  # 256x32x200 + 64x64x400 ==> 256x64x400

        s5 = self.s5(decode5)
        s4 = self.s4(decode4)
        s3 = self.s3(decode3)
        s2 = self.s2(decode2)

        x = torch.cat([s2, s3, s4, s5], 1)
        x = self.output(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x


class ResNet34(nn.Module):
    def __init__(self, classes=4, pretrained=True):
        super(ResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1 = nn.Sequential(self.resnet.layer1, SCSEBlock(64))
        self.layer2 = nn.Sequential(self.resnet.layer2, SCSEBlock(128))
        self.layer3 = nn.Sequential(self.resnet.layer3, SCSEBlock(256))
        self.layer4 = nn.Sequential(self.resnet.layer4, SCSEBlock(512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.layer0(x)  # 3x256x1600 ==> 64x128x800 (1/4)
        x = self.layer1(x)  # 64x128x800 ==> 64x64x400 (1/8)
        x = self.layer2(x)  # 64x64x400 ==> 128x32x200 (1/16)
        x = self.layer3(x)  # 128x32x200 ==> 256x16x100 (1/32)
        x = self.layer4(x)  # 256x16x100 ==> 512x8x50 (1/64)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ResNext50(nn.Module):
    def __init__(self, classes=4, pretrained=True):
        super(ResNext50, self).__init__()
        self.resnet = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1 = nn.Sequential(self.resnet.layer1, SCSEBlock(256))
        self.layer2 = nn.Sequential(self.resnet.layer2, SCSEBlock(512))
        self.layer3 = nn.Sequential(self.resnet.layer3, SCSEBlock(1024))
        self.layer4 = nn.Sequential(self.resnet.layer4, SCSEBlock(2048))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.layer0(x)  # 3x256x1600 ==> 64x128x800 (1/4)
        x = self.layer1(x)  # 64x128x800 ==> 64x64x400 (1/8)
        x = self.layer2(x)  # 64x64x400 ==> 128x32x200 (1/16)
        x = self.layer3(x)  # 128x32x200 ==> 256x16x100 (1/32)
        x = self.layer4(x)  # 256x16x100 ==> 512x8x50 (1/64)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ResNet34V2(nn.Module):
    def __init__(self, classes=4, pretrained=True):
        super(ResNet34V2, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        self.layer1 = nn.Sequential(self.resnet.layer1, SCSEBlock(64))
        self.layer2 = nn.Sequential(self.resnet.layer2, SCSEBlock(128))
        self.layer3 = nn.Sequential(self.resnet.layer3, SCSEBlock(256))
        self.layer4 = nn.Sequential(self.resnet.layer4, SCSEBlock(512))

        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1),
                                     Downsample(channels=64, filt_size=3, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 64x128x800 ==> 64x64x400 (1/8)
        x = self.layer2(x)  # 64x64x400 ==> 128x32x200 (1/16)
        x = self.layer3(x)  # 128x32x200 ==> 256x16x100 (1/32)
        x = self.layer4(x)  # 256x16x100 ==> 512x8x50 (1/64)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ResNext50V2(nn.Module):
    def __init__(self, classes=4, pretrained=True):
        super(ResNext50V2, self).__init__()
        self.resnet = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        self.layer1 = nn.Sequential(self.resnet.layer1, SCSEBlock(256))
        self.layer2 = nn.Sequential(self.resnet.layer2, SCSEBlock(512))
        self.layer3 = nn.Sequential(self.resnet.layer3, SCSEBlock(1024))
        self.layer4 = nn.Sequential(self.resnet.layer4, SCSEBlock(2048))

        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1),
                                     Downsample(channels=64, filt_size=3, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 64x128x800 ==> 64x64x400 (1/8)
        x = self.layer2(x)  # 64x64x400 ==> 128x32x200 (1/16)
        x = self.layer3(x)  # 128x32x200 ==> 256x16x100 (1/32)
        x = self.layer4(x)  # 256x16x100 ==> 512x8x50 (1/64)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

# if __name__ == '__main__':
#     from data_loader import *
#
#     dataloader = get_dataloader_seg(phase="valid", fold=0, train_batch_size=4,
#                                     valid_batch_size=4, num_workers=2)
#     model = FPEfficientNet()
#     model.cuda()
#     model.eval()
#
#     imgs, masks = next(iter(dataloader))
#     imgs, masks = imgs.cuda(), masks.cuda()
#
#     pred = model(imgs)
#     print(masks)
#     print(torch.sigmoid(pred))
