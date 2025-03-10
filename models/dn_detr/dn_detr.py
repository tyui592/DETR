# -*- coding: utf-8 -*-
"""Conditional DETR Model Code."""

import math
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import inverse_sigmoid

from .backbone import get_backbone
from .position_encoding import get_pos_embedding
from .transformer import get_transformer
from .layers import MLP
from .dn_func import make_cdn_query, split_outputs


def get_dn_detr(args, device):
    """Get a detr model."""
    backbone = get_backbone(args)
    transformer = get_transformer(args)
    pos_embedding = get_pos_embedding(args)

    n_cls = args.n_cls
    if args.cls_loss == 'ce':
        n_cls += 1

    model = DN_DETR(backbone=backbone,
                    transformer=transformer,
                    pos_embedding=pos_embedding,
                    n_query=args.n_query,
                    n_cls=n_cls,
                    d_model=args.d_model,
                    activation=args.activation,
                    iter_update=args.iter_update,
                    num_group=args.num_group,
                    box_noise_scale=args.box_noise_scale,
                    label_noise_scale=args.label_noise_scale,
                    fix_init_xy=args.fix_init_xy).to(device)

    return model


class DN_DETR(nn.Module):
    """DN DETR Model."""

    def __init__(self,
                 backbone: nn.Module,
                 transformer: nn.Module,
                 pos_embedding: nn.Module,
                 n_query: int = 300,
                 n_cls: int = 2,
                 d_model: int = 256,
                 activation: str = 'relu',
                 iter_update: bool = True,
                 num_group: int = 3,
                 box_noise_scale: float = 0.4,
                 label_noise_scale: float = 0.2,
                 fix_init_xy: bool = False):
        """Initialize."""
        super().__init__()
        self.num_group = num_group
        self.box_noise_scale = box_noise_scale
        self.label_noise_scale = label_noise_scale
        self.d_model = d_model
        self.num_class = n_cls
        self.backbone = backbone
        self.transformer = transformer
        self.pos_embedding = pos_embedding

        self.input_proj = nn.Conv2d(self.backbone.out_ch,
                                    d_model,
                                    kernel_size=1)

        # anchor query
        self.anchor_query = nn.Embedding(n_query, 4)
        self.label_enc = nn.Embedding(self.num_class + 1, d_model - 1)

        self.box_embed = MLP(input_dim=d_model,
                             hidden_dim=d_model,
                             activation=activation,
                             output_dim=4,
                             num_layers=3)

        self.cls_embed = nn.Linear(d_model, n_cls)

        if iter_update:
            self.transformer.decoder.box_embed = self.box_embed

        # initialize parameters
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_embed.bias, bias_value)

        nn.init.constant_(self.box_embed.layers[-1].weight, 0.0)
        nn.init.constant_(self.box_embed.layers[-1].bias, 0.0)

        nn.init.normal_(self.label_enc.weight.data, mean=0.0, std=0.02)

        nn.init.uniform_(self.anchor_query.weight.data[:, :2], 0.0, 1.0)
        nn.init.uniform_(self.anchor_query.weight.data[:, 2:], 0.0, 0.5)
        self.anchor_query.weight.data = inverse_sigmoid(self.anchor_query.weight.data)
        if fix_init_xy:
            # fix randomly initialized xy points
            self.anchor_query.weight.data[:, :2].requires_grad = False

    def forward(self, img, mask, targets=None):
        """Forward."""
        # feature extraction
        feature = self.backbone(img)
        mask = F.interpolate(mask.unsqueeze(1), feature.shape[2:], mode='nearest')

        # positional embedding
        pos_embed = self.pos_embedding(mask)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        mask = mask.flatten(2).unsqueeze(1).bool()
        img_feature = self.input_proj(feature).flatten(2).permute(0, 2, 1)

        noised_query = None
        if self.training:
            noised_query = make_cdn_query(targets=targets,
                                          num_group=self.num_group, 
                                          label_enc=self.label_enc,
                                          num_class=self.num_class,
                                          label_noise_scale=self.label_noise_scale,
                                          box_noise_scale=self.box_noise_scale,
                                          device=img.device)

        hs, anchors = self.transformer(img_feature=img_feature,
                                       img_pos_embed=pos_embed,
                                       img_mask=mask,
                                       anchor_query=self.anchor_query,
                                       noised_query=noised_query,
                                       label_enc=self.label_enc)

        if self.training:
            outputs = []
            for h, anchor in zip(hs, anchors):
                offset = self.box_embed(h)
                pred_box = (inverse_sigmoid(anchor) + offset).sigmoid()
                outputs.append({
                    'pred_logits': self.cls_embed(h),
                    'pred_boxes': pred_box
                })

            outputs = split_outputs(outputs, targets, self.num_group)
            outputs['noised_targets'] = noised_query['targets']
            outputs['noised_indices'] = noised_query['indices']
            
        else:
            outputs = {
                'pred_logits': self.cls_embed(hs[-1]),
                'pred_boxes': (inverse_sigmoid(anchors[-1]) + self.box_embed(hs[-1])).sigmoid()
            }

        return outputs