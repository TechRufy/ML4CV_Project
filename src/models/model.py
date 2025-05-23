import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights

from src.models.model_utils import ConvBNAct, NonBottleneck1D

resnet = {'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50,
          'resnet101': models.resnet101, 'resnet152': models.resnet152}


class OSNet(nn.Module):
    def __init__(self, num_classes, encoder="resnet18", decoder="PSPNet", sizes=(1, 2, 3, 6), psp_size=2048,
                 deep_features_size=1024):
        super(OSNet, self).__init__()
        if encoder in resnet.keys():
            self.encoder = resnet.get(encoder)(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            assert "encoder must be one of 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'"
        if decoder in ["PSPNet"]:
            self.decoder = PSPModule(psp_size, 1024, sizes)
            self.drop_1 = nn.Dropout2d(p=0.3)

            self.up_1 = PSPUpsample(1024, 256)
            self.up_2 = PSPUpsample(256, 64)
            self.up_3 = PSPUpsample(64, 64)

            self.drop_2 = nn.Dropout2d(p=0.15)
            self.final = nn.Sequential(
                nn.Conv2d(64, num_classes, kernel_size=1),
                nn.LogSoftmax()
            )

            self.classifier = nn.Sequential(
                nn.Linear(deep_features_size, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        else:
            assert "decoder must be 'PSPNet'"
        self.num_classes = num_classes

    def forward(self, x):
        f = self.encoder(x)
        print(f)
        p = self.decoder(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=f, output_size=(1, 1)).view(-1, f.size(1))

        return self.final(p), self.classifier(auxiliary)


def _make_stage(features, size):
    prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
    conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
    return nn.Sequential(prior, conv)


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([_make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)
