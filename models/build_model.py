########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################

import torch
from torch import nn

from .model_one_modality import OWSNetwork
from .resnet import ResNet

from torchsummary import summary


def build_model(height,
                width,
                pretrained_on_imagenet, 
                encoder,
                encoder_block,
                activation,
                encoder_decoder_fusion,
                n_classes,
                pretrained_dir,
                he_init = True):

    input_channels = 3
    model = OWSNetwork(
        height=height,
        width=width,
        pretrained_on_imagenet=pretrained_on_imagenet,
        encoder=encoder,
        encoder_block=encoder_block,
        activation=activation,
        input_channels=input_channels,
        encoder_decoder_fusion=encoder_decoder_fusion,
        num_classes=n_classes,
        pretrained_dir=pretrained_dir,
        nr_decoder_blocks=None,
        channels_decoder=None,
        upsampling="bilinear",
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("\n\n")

    print("\n\n")
    if he_init:
        module_list = []

        # first filter out the already pretrained encoder(s)
        for c in model.children():
            if pretrained_on_imagenet and isinstance(c, ResNet):
                # already initialized
                continue
            for m in c.modules():
                module_list.append(m)

        # iterate over all the other modules
        # output layers, layers followed by sigmoid (in SE block) and
        # depthwise convolutions (currently only used in learned upsampling)
        # are not initialized with He method
        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if (
                    m.out_channels == n_classes
                    or m.out_channels == 19
                    or isinstance(module_list[i + 1], nn.Sigmoid)
                    or m.groups == m.in_channels
                ):
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("Applied He init.")

    return model, device
