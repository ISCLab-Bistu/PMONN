
import torchvision
import torch
import torch.nn as nn
from ding.phi_conv2d import phi_conv

from collections import OrderedDict
from torchinfo import summary


class VGG(nn.Module):
    # The construction of the network
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, num_classes, bias=False),


    def forward(self, x):
        for name, modu in self.features.items():
            if 'sigmoid_' in name:
                x = (x - x.min()) / (x.max() - x.min())
                x = modu(x)
            else:
                x = modu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



cfg = [64, 64, 'M', 128, 128, 'M']


def make_layers(cfg, batch_norm=True, bias=False, init=True, isEnlarge=True, ac='relu'):
    layers = nn.ModuleDict()
    in_channels = 1
    for i, v in enumerate(cfg):
        i = str(i)
        # down sampling
        if v == 'M':
            # PB convolution
            conv2d = phi_conv(init, in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
            # DPW convolution
            conv1d = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=1, bias=bias)
        # no down sampling, only convolution
        else:
            conv2d = phi_conv(init, in_channels, v, kernel_size=3, padding=1, bias=False)
            conv1d = nn.Conv2d(v, v, groups=v, kernel_size=1, bias=bias)
            in_channels = v
        """Whether to add the DPW layer to the network"""
        if isEnlarge:
            layers["phi_conv_" + i] = conv2d
            layers["normal_conv_" + i] = conv1d
        else:
            layers["phi_conv_" + i] = conv2d
            # layers += [("phi_conv_" + i, conv2d)]
      
        """Whether to add the DPW layer to the network"""
        if batch_norm:
            layers["bn_" + i] = nn.BatchNorm2d(in_channels)
        """select the nonlinear activation function"""
        if ac == 'relu':
            layers[ac + '_' + i] = nn.ReLU(inplace=True)
        else:
            layers[ac + '_' + i] = nn.Sigmoid()
    return layers



"""The inilization of the network"""
def vgg6(num_classes=10, batch_norm=True, bias=False, init=True, isEnlarge=True, ac='sigmoid'):
   
    model = VGG(make_layers(cfg, batch_norm=batch_norm, bias=bias, init=init, isEnlarge=isEnlarge, ac=ac), num_classes)

    return model



