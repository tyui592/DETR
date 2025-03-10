# -*- coding: utf-8 -*-
"""Backbone Code."""
import torch
import torch.nn as nn
from torchvision.models import (convnext_tiny, ConvNeXt_Tiny_Weights,
                                resnet18, ResNet18_Weights,
                                resnet34, ResNet34_Weights,
                                resnet50, ResNet50_Weights)


def get_backbone(args):
    """Get the backbone model."""
    backbone = Backbone(backbone=args.backbone,
                        layer_index=args.layer_index)
    return backbone


class Backbone(nn.Module):
    """Backbone Class."""

    def __init__(self, backbone, layer_index):
        """Initialize."""
        super().__init__()
        if backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet50_dc5']:
            self.feature_extractor = ResNet(backbone=backbone,
                                            layer_index=layer_index)
            self.out_ch = self.feature_extractor.out_ch
        else:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.feature_extractor = convnext_tiny(weights=weights).features
            self.out_ch = 768

    def forward(self, x):
        """Forward fucntion."""
        return self.feature_extractor(x)


class ResNet(nn.Module):
    """Resnet Backbone."""

    def __init__(self,
                 backbone='resnet18',
                 layer_index=7):
        """Initilize.

        Layer index, stride, channels per resnet model.
            * resnet18, resnet34
            - (4, 1/4, 64), (5, 1/8, 128), (6, 1/16, 256), (7, 1/32, 512)
            * resnet50, resnet101
            - (4, 1/4, 256), (5, 1/8, 512), (6, 1/16, 1024), (7, 1/32, 2048)
        """
        super(ResNet, self).__init__()
        if backbone in ['resnet18', 'resnet34']:
            channels = {4: 64, 5: 128, 6: 256, 7: 512}
        else:
            channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

        # num channels of the last feature map.
        self.out_ch = channels[layer_index]

        if backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1
            layers = list(resnet18(weights=weights,
                                   norm_layer=FrozenBatchNorm2d).children())
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1
            layers = list(resnet34(weights=weights,
                                   norm_layer=FrozenBatchNorm2d).children())
        else:
            dilation = [False, False, False]
            if backbone == 'resnet50_dc5':
                dilation[-1] = True
            weights = ResNet50_Weights.IMAGENET1K_V1
            layers = list(resnet50(weights=weights,
                                   norm_layer=FrozenBatchNorm2d,
                                   replace_stride_with_dilation=dilation).children())

        temp = []
        self.layers = nn.ModuleList()
        for i in range(layer_index + 1):
            temp.append(layers[i])
            if i == layer_index:
                self.layers.append(nn.Sequential(*temp))
                temp = []

    def forward(self, x):
        """Forward."""
        for layer in self.layers:
            x = layer(x)
        return x


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        """Initialize."""
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        """Forward function."""
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
