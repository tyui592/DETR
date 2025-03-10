# -*- coding: utf-8 -*-
"""DETR Model Code."""

import math
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_backbone
from .transformer import get_transformer
from .position_encoding import get_pos_embedding
from .layers import MLP


def get_detr(args, device):
    """Get a detr model."""
    backbone = get_backbone(args)
    transformer = get_transformer(args)
    pos_embedding = get_pos_embedding(args)

    n_cls = args.n_cls
    if args.cls_loss == 'ce':
        n_cls += 1

    model = DETR(backbone=backbone,
                 transformer=transformer,
                 pos_embedding=pos_embedding,
                 n_query=args.n_query,
                 n_cls=n_cls,
                 d_model=args.d_model,
                 activation=args.activation).to(device)

    return model


class DETR(nn.Module):
    """DETR Model."""

    def __init__(self,
                 backbone: nn.Module,
                 transformer: nn.Module,
                 pos_embedding: nn.Module,
                 n_query: int = 300,
                 n_cls: int = 2,
                 d_model: int = 256,
                 activation: str = 'relu'):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.pos_embedding = pos_embedding

        self.input_proj = nn.Conv2d(self.backbone.out_ch,
                                    d_model,
                                    kernel_size=1)

        self.query = nn.Embedding(n_query, d_model)

        self.box_embed = MLP(input_dim=d_model,
                             hidden_dim=d_model,
                             activation=activation,
                             output_dim=4,
                             num_layers=3)

        self.cls_embed = nn.Linear(d_model, n_cls)

        # initialize parameters
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_embed.bias.data, bias_value)

        nn.init.constant_(self.box_embed.layers[-1].weight, 0.0)
        nn.init.constant_(self.box_embed.layers[-1].bias, 0.0)

        nn.init.normal_(self.query.weight.data, mean=0.0, std=0.02)


    def forward(self, img, mask, target=None):
        """Forward."""
        # feature extraction
        feature = self.backbone(img)
        mask = F.interpolate(mask.unsqueeze(1), feature.shape[2:], mode='nearest')

        # positional embedding
        pos_embed = self.pos_embedding(mask)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        mask = mask.flatten(2).unsqueeze(1).bool()
        f = self.input_proj(feature).flatten(2).permute(0, 2, 1)
        
        hs = self.transformer(f, pos_embed, self.query.weight, mask)
        
        if self.training:
            outputs = {'model': []}
            for h in hs:
                outputs['model'].append({
                    'pred_logits': self.cls_embed(h),
                    'pred_boxes': self.box_embed(h).sigmoid()
                })
        else:
            outputs = {
                'pred_logits': self.cls_embed(hs[-1]),
                'pred_boxes': self.box_embed(hs[-1]).sigmoid()
            }

        return outputs