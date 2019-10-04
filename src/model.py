import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import math
from torch.utils import model_zoo
from configure import *


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.block(x)
        x = self.SCSE(x)

        return x


class UResNet34(nn.Module):
    def __init__(self, classes=1, pretrained=True):
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
        self.decoder1 = DecoderBlock(64, 32, 64)

        self.dropout = nn.Dropout2d(p=0.5)
        self.output = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, classes, kernel_size=1, padding=0))

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

        x = torch.cat((x,
                       F.interpolate(decode2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.interpolate(decode3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.interpolate(decode4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.interpolate(decode5, scale_factor=16, mode='bilinear', align_corners=True)),
                      1)  # 320, 256, 1600
        x = self.dropout(x)
        x = self.output(x)

        return x


class ResNet34(nn.Module):
    def __init__(self, classes=4, pretrained=True):
        super(ResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

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


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class USeResNext50(nn.Module):
    def __init__(self, classes=4, pretrain=True):
        super(USeResNext50, self).__init__()
        self.senet = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                           inplanes=64, input_3x3=False, downsample_kernel_size=1,
                           dropout_p=None, downsample_padding=0)

        if pretrain:
            self.senet.load_state_dict(model_zoo.load_url(SEResNext50URL))

        self.encoder1 = self.senet.layer0
        self.encoder2 = self.senet.layer1
        self.encoder3 = self.senet.layer2
        self.encoder4 = self.senet.layer3
        self.encoder5 = self.senet.layer4

        self.decoder5 = DecoderBlock(2048 + 1024, 256, 64)
        self.decoder4 = DecoderBlock(64 + 512, 128, 64)
        self.decoder3 = DecoderBlock(64 + 256, 64, 64)
        self.decoder2 = DecoderBlock(64 + 64, 64, 64)
        self.decoder1 = DecoderBlock(64, 32, 64)

        self.dropout = nn.Dropout2d(p=0.5)
        self.output = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, classes, kernel_size=1, padding=0))

    def forward(self, x):
        for module in self.encoder1[:-1]:
            x = module(x)

        encode1 = x  # 3x256x1600 ==> 64x128x800
        encode2 = self.encoder2(self.encoder1[-1](x))  # 64x128x800 ==> 256x64x400
        encode3 = self.encoder3(encode2)  # 256x64x400 ==> 512x32x200
        encode4 = self.encoder4(encode3)  # 512x32x200 ==> 1024x16x100
        encode5 = self.encoder5(encode4)  # 1024x16x100 ==> 2048x8x50

        decode5 = self.decoder5(encode5, encode4)  # 2048x8x50 + 1024x16x100 ==> 64x16x100
        decode4 = self.decoder4(decode5, encode3)  # 64x16x100 + 512x32x200 ==> 64x32x200
        decode3 = self.decoder3(decode4, encode2)  # 64x32x200 + 256x64x400 ==> 64x64x400
        decode2 = self.decoder2(decode3, encode1)  # 64x64x400 + 64x128x800 ==> 64x128x800
        x = self.decoder1(decode2, None)

        x = torch.cat((x,
                       F.interpolate(decode2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.interpolate(decode3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.interpolate(decode4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.interpolate(decode5, scale_factor=16, mode='bilinear', align_corners=True)),
                      1)  # 320, 256, 1600
        x = self.dropout(x)
        x = self.output(x)
        return x

# if __name__ == '__main__':
#     import os
#     from data_loader import get_dataloader
#     from loss import DiceBCEVAELoss
#
#     dataloader = get_dataloader(phase="valid", fold=0, batch_size=4, num_workers=2)
#     model = UResNet34VAE()
#     model.cuda()
#     model.eval()
#
#     imgs, masks = next(iter(dataloader))
#     imgs, masks = imgs.cuda(), masks.cuda()
#
#     masks_pred, images_pred, mu, logvar = model(imgs)
#
#     criterion = DiceBCEVAELoss()
#
#     print(masks_pred.shape)
#     print(images_pred.shape)
#     print(mu.shape)
#     print(logvar.shape)
#
#     loss, bce_loss, dice_loss, vae_loss = criterion(masks_pred=masks_pred, masks=masks,
#                                                     images_pred=images_pred, images=imgs,
#                                                     mu=mu, logvar=logvar)
#     print(loss)
#     print(bce_loss)
#     print(dice_loss)
#     print(vae_loss)
