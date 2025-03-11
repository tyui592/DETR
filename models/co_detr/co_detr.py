# -*- coding: utf-8 -*-
"""Conditional DETR Model Code."""

import math
import copy
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import inverse_sigmoid

from .backbone import get_backbone
from .position_encoding import get_pos_embedding
from .transformer import get_transformer
from .layers import MLP
from .dino_func import make_cdn_query, split_outputs


def get_co_detr(args, device):
    """Get a detr model."""
    backbone = get_backbone(args)
    transformer = get_transformer(args)
    pos_embedding = get_pos_embedding(args)

    n_cls = args.n_cls
    if args.cls_loss == 'ce':
        n_cls += 1 # add background class
        
    model = Co_DETR(backbone=backbone,
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
                    num_dn_query=args.num_dn_query,
                    add_neg_query=args.add_neg_query,
                    fix_init_xy=args.fix_init_xy,
                    two_stage_mode=args.two_stage_mode,
                    two_stage_share_head=args.two_stage_share_head).to(device)

    return model


class Co_DETR(nn.Module):
    """Co-DETR Model."""

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
                 num_dn_query: int = 100,
                 add_neg_query: bool = True,
                 fix_init_xy: bool = False,
                 two_stage_mode: str = 'none',
                 two_stage_share_head: bool = True):
        """Initialize.
        """
        super().__init__()
        self.num_group = num_group
        self.box_noise_scale = box_noise_scale
        self.label_noise_scale = label_noise_scale
        self.d_model = d_model
        self.num_class = n_cls
        self.num_dn_query = num_dn_query
        self.add_neg_query = add_neg_query # contrastive denoised query
        self.backbone = backbone
        self.transformer = transformer
        self.pos_embedding = pos_embedding

        self.input_proj = nn.Sequential(
            nn.Conv2d(self.backbone.out_ch, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        self.two_stage_mode = two_stage_mode
        self.anchor_query = None
        if self.two_stage_mode in ['none', 'static']:
            # anchor query
            self.anchor_query = nn.Embedding(n_query, 4)
            
        # label embedding
        self.label_enc = nn.Embedding(self.num_class + 1, d_model - 1)

        # head
        _box_embed = MLP(input_dim=d_model,
                             hidden_dim=d_model,
                             activation=activation,
                             output_dim=4,
                             num_layers=3)
        nn.init.constant_(_box_embed.layers[-1].weight, 0.0)
        nn.init.constant_(_box_embed.layers[-1].bias, 0.0)

        _cls_embed = nn.Linear(d_model, self.num_class)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(_cls_embed.bias, bias_value)

        self.cls_embed = _cls_embed
        self.box_embed = _box_embed

        if iter_update:
            self.transformer.decoder.box_embed = _box_embed

        # two stage
        if two_stage_share_head:
            self.transformer.enc_box_embed = _box_embed
            self.transformer.enc_cls_embed = _cls_embed
        else:
            self.transformer.enc_box_embed = copy.deepcopy(_box_embed)
            self.transformer.enc_cls_embed = copy.deepcopy(_cls_embed)

        # init 
        nn.init.normal_(self.label_enc.weight.data, mean=0.0, std=0.02)

        if self.anchor_query is not None:
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
        image_feature = self.input_proj(feature).flatten(2).permute(0, 2, 1)
        
        # Make input query and target for denosing queries
        noised_query = None
        if self.training:
            noised_query = make_cdn_query(targets=targets,
                                          bs=img.shape[0],
                                          num_group=self.num_group, 
                                          label_enc=self.label_enc,
                                          num_class=self.num_class,
                                          label_noise_scale=self.label_noise_scale,
                                          box_noise_scale=self.box_noise_scale,
                                          num_cdn_query=self.num_dn_query,
                                          add_neg_query=self.add_neg_query,
                                          device=img.device)
        
        hs, anchors, enc_ref, memory, aux_indices = self.transformer(img_feature=image_feature,
                                                        img_pos_embed=pos_embed,
                                                        img_mask=mask,
                                                        anchor_query=self.anchor_query,
                                                        feature_shape=feature.shape,
                                                        noised_query=noised_query,
                                                        label_enc=self.label_enc,
                                                        targets=targets)
        
        if self.training:
            outputs = []
            for h, anchor in zip(hs, anchors):
                offset = self.box_embed(h)
                outputs.append({
                    'pred_logits': self.cls_embed(h),
                    'pred_boxes': (inverse_sigmoid(anchor) + offset).sigmoid()
                })

            if self.two_stage_mode != 'none':
                first_stage = []
                first_stage.append({
                'pred_logits': self.transformer.enc_cls_embed(memory),
                'pred_boxes': (enc_ref + self.transformer.enc_box_embed(memory)).sigmoid()
                })

            # Split matching part and denosing part
            num_noised_query = self.num_dn_query * self.num_group * (2 if self.add_neg_query else 1)
            if self.anchor_query is not None:
                num_model_query = self.anchor_query.weight.shape[0]
            else:
                num_model_query = min(self.transformer.num_encoder_query, memory.shape[1])
            num_aux_query = memory.shape[1]
            split_sizes = [num_noised_query, num_model_query, num_aux_query]
            split_labels = ['cdn', 'model', 'aux']
            outputs = split_outputs(outputs, split_sizes, split_labels) # split matching part and denoising part
            
            # add taragets for training
            outputs['cdn_targets'] = noised_query['targets']
            outputs['cdn_indices'] = noised_query['indices']
            outputs['first_stage'] = first_stage
            outputs['aux_indices'] = aux_indices
            
        else:
            outputs = {
                'pred_logits': self.cls_embed(hs[-1]),
                'pred_boxes': (inverse_sigmoid(anchors[-1]) + self.box_embed(hs[-1])).sigmoid()
                }
        
        return outputs