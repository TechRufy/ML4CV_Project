########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import torch
import torch.nn as nn
import torch.nn.functional as F


class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=None,
        dilation=1,
        norm_layer=None,
        activation=nn.ReLU(inplace=True),
        residual_only=False,
    ):
        super().__init__()
        # print(
        #     "Notice: parameters groups, base_width and norm_layer are "
        #     "ignored in NonBottleneck1D"
        # )
        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(
            inplanes, planes, (3, 1), stride=(stride, 1), padding=(1, 0), bias=True
        )
        self.conv1x3_1 = nn.Conv2d(
            planes, planes, (1, 3), stride=(1, stride), padding=(0, 1), bias=True
        )
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(
            planes,
            planes,
            (3, 1),
            padding=(1 * dilation, 0),
            bias=True,
            dilation=(dilation, 1),
        )
        self.conv1x3_2 = nn.Conv2d(
            planes,
            planes,
            (1, 3),
            padding=(0, 1 * dilation),
            bias=True,
            dilation=(1, dilation),
        )
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        # +input = identity (residual connection)
        return self.act(output + identity)

class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernel_size,
        activation=nn.ReLU(inplace=True),
        dilation=1,
        stride=1,
    ):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module(
            "conv",
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                dilation=dilation,
                stride=stride,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(channels_out))
        self.add_module("act", activation)


class ConvBN(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(ConvBN, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(channels_out))


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExcitationTensorRT(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitationTensorRT, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # TensorRT restricts the maximum kernel size for pooling operations
        # by "MAX_KERNEL_DIMS_PRODUCT" which leads to problems if the input
        # feature maps are of large spatial size
        # -> workaround: use cascaded two-staged pooling
        # see: https://github.com/onnx/onnx-tensorrt/issues/333
        if x.shape[2] > 120 and x.shape[3] > 160:
            weighting = F.adaptive_avg_pool2d(x, 4)
        else:
            weighting = x
        weighting = F.adaptive_avg_pool2d(weighting, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def swish(x):
    return x * torch.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0
