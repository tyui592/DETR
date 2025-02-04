# -*- coding: utf-8 -*-
"""Conditional DETR Model Code."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import inverse_sigmoid

from .backbone import get_backbone
from .position_encoding import get_pos_embedding
from .layers import MHA, FeedForward, MLP
from .transformer import get_transformer


def get_conditional_detr(args, device):
    """Get a detr model."""
    backbone = get_backbone(args)
    transformer = get_transformer(args)
    pos_embedding = get_pos_embedding(args)

    n_cls = args.n_cls
    if args.cls_loss == 'ce':
        n_cls += 1

    model = ConditionalDETR(backbone=backbone,
                            transformer=transformer,
                            pos_embedding=pos_embedding,
                            n_query=args.n_query,
                            n_cls=n_cls,
                            d_model=args.d_model,
                            activation=args.activation).to(device)

    return model


class ConditionalDETR(nn.Module):
    """Conditional DETR Model."""

    def __init__(self,
                 backbone: nn.Module,
                 transformer: nn.Module,
                 pos_embedding: nn.Module,
                 n_query: int = 300,
                 n_cls: int = 2,
                 d_model: int = 256,
                 activation: str = 'relu'):
        """Init."""
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
        nn.init.constant_(self.cls_embed.bias, bias_value)

        nn.init.constant_(self.box_embed.layers[-1].weight, 0.0)
        nn.init.constant_(self.box_embed.layers[-1].bias, 0.0)

        nn.init.normal_(self.query.weight, mean=0.0, std=0.02)


    def forward(self, img, mask):
        """Forward.

        img: B, C, H, W
        mask: B, H, W
        """
        bs = img.shape[0]

        # feature extraction
        feature = self.backbone(img)
        mask = F.interpolate(mask.unsqueeze(1), feature.shape[2:], mode='nearest')

        # positional embedding
        pos_embed = self.pos_embedding(mask)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        mask = mask.flatten(2).unsqueeze(1).bool()
        f = self.input_proj(feature).flatten(2).permute(0, 2, 1)

        query = self.query.weight.unsqueeze(0).repeat(bs, 1, 1)
        hs, ref_points, enc_sa, dec_sa, dec_ca = self.transformer(f, pos_embed, query, mask)
        ref_points_before_sigmoid = inverse_sigmoid(ref_points)

        outputs = []
        for h in hs:
            tmp = self.box_embed(h)
            tmp[..., :2] += ref_points_before_sigmoid
            outputs.append(
                {
                    'pred_logits': self.cls_embed(h),
                    'pred_boxes': tmp.sigmoid()
                }
            )

        # for visualize attention weights
        attns = {
            'enc_sa': enc_sa,
            'dec_sa': dec_sa,
            'dec_ca': dec_ca
        }

        return outputs, attns
