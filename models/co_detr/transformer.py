# -*- coding: utf-8 -*-
"""Transformer Code."""

import math
import torch
import torch.nn as nn
from typing import Optional
from collections import defaultdict
from utils.misc import inverse_sigmoid
from .layers import MHA, FeedForward, MLP


def get_transformer(args):
    """Get a transformer model with the arguments."""
    transformer = Transformer(num_encoder_layers=args.n_encoder_layers,
                              num_decoder_layers=args.n_decoder_layers,
                              d_model=args.d_model,
                              n_heads=args.n_heads,
                              d_ff=args.d_ff,
                              return_intermediate=args.return_intermediate,
                              activation=args.transformer_activation,
                              p_drop=args.p_drop,
                              encoder_position_mode=args.encoder_position_mode,
                              decoder_sa_position_mode=args.decoder_sa_position_mode,
                              decoder_ca_position_mode=args.decoder_ca_position_mode,
                              query_scale_mode=args.query_scale_mode,
                              num_pattern=args.num_pattern,
                              modulate_wh_attn=args.modulate_wh_attn,
                              decoder_temperature=args.temperature,
                              two_stage_mode=args.two_stage_mode,
                              num_encoder_query=args.num_encoder_query)
    return transformer


class Transformer(nn.Module):
    """Transformer Class."""

    def __init__(self,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 d_model: int = 256,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 return_intermediate: bool = True,
                 activation: str = 'relu',
                 p_drop: float = 0.0,
                 encoder_position_mode: str = 'add',
                 decoder_sa_position_mode: str = 'add',
                 decoder_ca_position_mode: str = 'cat',
                 query_scale_mode: str = 'diag',
                 num_pattern: int = 0,
                 modulate_wh_attn: bool = True,
                 decoder_temperature: int = 10000,
                 two_stage_mode: str = 'static',
                 num_encoder_query: int = 100,
                #  aux_heads: dict = {}
                 ):
        """Initiailze."""
        super().__init__()

        self.encoder = Encoder(n_layers=num_encoder_layers,
                               d_model=d_model,
                               n_heads=n_heads,
                               d_ff=d_ff,
                               activation=activation,
                               p_drop=p_drop,
                               add_pos_enc=encoder_position_mode)

        self.decoder = Decoder(n_layers=num_decoder_layers,
                               d_model=d_model,
                               n_heads=n_heads,
                               d_ff=d_ff,
                               activation=activation,
                               return_intermediate=return_intermediate,
                               p_drop=p_drop,
                               sa_position_mode=decoder_sa_position_mode,
                               ca_position_mode=decoder_ca_position_mode,
                               query_scale_mode=query_scale_mode,
                               temperature=decoder_temperature,
                               modulate_wh_attn=modulate_wh_attn)
        
        # self.aux_heads = aux_heads

        self.two_stage_mode = two_stage_mode
        if self.two_stage_mode in ['pure', 'mix']:
            self.enc_box_embed = None
            self.enc_cls_embed = None
            self.num_encoder_query = num_encoder_query
            self.enc_out_layer = nn.Linear(d_model, d_model)
            self.enc_out_norm = nn.LayerNorm(d_model)
            
        self._reset_parameters()
        # Pattern embedding for multiple object detections on a location.
        if num_pattern > 0:
            self.patterns = nn.Embedding(num_pattern, d_model)
            nn.init.normal_(self.patterns.weight, 0.0, 0.02)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                img_feature: torch.Tensor,
                img_pos_embed: torch.Tensor,
                img_mask: torch.Tensor,
                anchor_query: Optional[torch.nn.Embedding],
                noised_query: Optional[dict],
                feature_shape: list,
                label_enc: torch.nn.Embedding,
                targets: list):
        """Transformer Forward.

        image feautre: image feuatre (batch size x HW x d_model)
        pos_embed: positional encoding for image feature (batch_size x HW x d_model)
        anchor_query: object positional query (num_query x 4)
        label_enc: nn.Embedding for label embedding
        img_mask: key padding mask (batch size x 1 x HW)
        cdn
            - 'anchor_query': batch size x (num_dn_query x num_group x 2(if add_neg_query)) x 4
            - 'label_query': batch size x ~ x 1
            - 'targets': for cdn query training
            - 'indices': for cdn query training
            - 'attn_mask': self-attention mask to avoid cheating the noised query
        """
        memory = self.encoder(img_feature, img_pos_embed, img_mask)

        # two stage
        enc_topk_output = None
        reference = None
        if self.two_stage_mode in ['pure', 'mix'] or hasattr(self, 'atss'):
            memory = self.enc_out_norm(self.enc_out_layer(memory))
            
            # get referecne with mask and feature map shape
            reference = self.get_encoder_reference(img_mask, feature_shape)

            # get logits and box coordinates
            logits = self.enc_cls_embed(memory)
            boxes = (reference + self.enc_box_embed(memory)).sigmoid()

            # get topk(by logit) samples(boxs, labels) from the encoder output
            enc_topk_output = self.get_topk_query(boxes, logits, img_mask, self.num_encoder_query)

            # aux_query = None
            # aux_indices = None
            # if hasattr(self, 'atss') and self.training:
            #     aux_query, aux_indices = self.make_aux_query(boxes, logits, targets, label_enc, memory.device)
                
        # make query for decoder (merge multiple queries for efficiency)
        label_query, anchor_query, dec_sa_mask = self.make_object_query(bs=memory.shape[0],
                                                                        mode=self.two_stage_mode, 
                                                                        encoder_output=enc_topk_output,
                                                                        label_enc=label_enc,
                                                                        anchor_query=anchor_query,
                                                                        noised_query=noised_query,
                                                                        aux_query=None,
                                                                        device=memory.device)
                
        # decoder
        hs, anchors = self.decoder(x=label_query,
                                   memory=memory,
                                   pos_embed=img_pos_embed,
                                   anchor_query=anchor_query,
                                   mask=img_mask,
                                   attn_mask=dec_sa_mask)
                
        return hs, anchors, reference, memory
    
    # def make_aux_query(self, boxes, logits, targets, label_enc, device):
    #     # make index for loss calculation
    #     atss_indices = self.atss({'pred_boxes': boxes, 'pred_logits': logits}, targets)

    #     labels = logits.argmax(dim=-1)
    #     bs, nq = labels.shape
    #     label_embeddings = label_enc(labels)
    #     indicator1 = torch.ones([1, 1, 1], dtype=label_embeddings.dtype, device=device)
    #     label_query = torch.cat([label_embeddings, indicator1.repeat(bs, nq, 1)], dim=-1)
    #     anchor_query = inverse_sigmoid(boxes)

    #     aux_query = {'label_query': label_query, 'anchor_query': anchor_query}
    #     aux_indices = {'atss': atss_indices}
    #     return aux_query, aux_indices
    
    def get_encoder_reference(self, mask, shapes):
        N, _, H, W = shapes
        device = mask.device
        mask = mask.squeeze().view(N, H, W)
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H-1, H, dtype=torch.float32), 
                                        torch.linspace(0, W-1, W, dtype=torch.float32),
                                        indexing='ij')

        xy = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1).to(device)

        scale_h = torch.sum(mask[:,  :, 0], dim=1)
        scale_w = torch.sum(mask[:, 0, :], dim=1)
        scale = torch.cat([scale_w.unsqueeze(-1), scale_h.unsqueeze(-1)], dim=1)
        scale = scale.unsqueeze(1).unsqueeze(2)

        xy = (xy.unsqueeze(0) + 0.5) / scale

        wh = torch.ones_like(xy) * 0.1

        enc_ref = inverse_sigmoid(torch.cat([xy, wh], dim=-1))
        enc_ref = enc_ref.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        enc_ref = enc_ref.flatten(1, 2)

        return enc_ref

    def get_topk_query(self, boxes, logits, mask, topk=100):
        """
        boxes, logits: output of the encoder
        mask: image padding mask
        topk: for encoder query
        """
        logits = logits.masked_fill(mask.squeeze().unsqueeze(-1) == 0, float('-inf'))
        max_logits, max_labels = torch.max(logits, dim=2)
        topk_indices = torch.topk(max_logits, k=topk, dim=1)[1]
        
        boxes = torch.gather(boxes, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        labels = torch.gather(max_labels, 1, topk_indices)
        output = {'boxes': boxes, 'labels': labels}
        return output

    def make_object_query(self,
                          bs: int,
                          mode: str,
                          encoder_output: Optional[dict],
                          label_enc: torch.nn.Embedding,
                          anchor_query: Optional[torch.nn.Embedding],
                          noised_query: dict,
                          device: torch.device):
        """Make decoder input query and attn. mask
        mode: 'add', 'pure', 'mix'
            - 'add': add encoder's topk boxes as additional input query to train encoder
                * query order: cdn query, model query, encoder output
            - 'pure': use encoder's topk boxes and labels as decoder query
            - 'mix': mixed query selection, only use box query
        encoder_output
        label_enc: torch.nn.Embedding [num_class+1, d_model-1]
        anchor_query: None or torch.Tensor [num_query, 4]
        cdn: dict
        """
        # make the query
        if mode in ['none', 'static']:
            num_class = label_enc.weight.shape[0] - 1
            num_query = anchor_query.weight.shape[0]
            label_embedding = label_enc(torch.tensor(num_class, device=device))
            indicator = torch.tensor([0], device=device)
            
            model_label_query = torch.cat([label_embedding, indicator]) # d_model
            model_label_query = model_label_query[None, None, :].repeat(bs, num_query, 1)
            model_anchor_query = anchor_query.weight.unsqueeze(0).repeat(bs, 1, 1) # bs, num_query, 4
            
        # make query
        if mode in ['pure', 'mix']:
            enc_boxes = encoder_output['boxes']
            enc_anchor_query = inverse_sigmoid(enc_boxes)
            
            if mode == 'pure':
                enc_labels = encoder_output['labels']
                N, C = enc_labels.shape # batch size, num query
                enc_label_embedding = label_enc(enc_labels) # batch size, num query, d_model
                indicator = torch.tensor([0], device=device)[None, None, :].repeat(N, C, 1)
                enc_label_query = torch.cat([enc_label_embedding, indicator], dim=-1)
                
            elif mode == 'mix':
                # use learnable label query
                num_class = label_enc.weight.shape[0] - 1
                num_query = enc_boxes.shape[1]
                label_embedding = label_enc(torch.tensor(num_class, device=device))
                indicator = torch.tensor([0], device=device)
                enc_label_query = torch.cat([label_embedding, indicator]) # d_model
                enc_label_query = enc_label_query[None, None, :].repeat(bs, num_query, 1)
            
        # merge cdn query and model query
        if self.training:
            _label_query = []
            _anchor_query = []
            _attn_mask = []

            noised_label_query = noised_query['label_query']
            noised_anchor_query = noised_query['anchor_query']
            noised_attn_mask = noised_query['attn_mask']

            _label_query.append(noised_label_query)
            _anchor_query.append(noised_anchor_query)
            _attn_mask.append(noised_attn_mask)
       
            if mode in ['none', 'static']:
                k = model_label_query.shape[1]
                _label_query.append(model_label_query)
                _anchor_query.append(model_anchor_query)
                _attn_mask.append(torch.ones((k, k), device=device))

            elif mode in ['pure', 'mix']:
                k = enc_label_query.shape[1]
                _label_query.append(enc_label_query)
                _anchor_query.append(enc_anchor_query)
                _attn_mask.append(torch.ones((k, k), device=device))
            
            label_query = torch.cat(_label_query, dim=1)
            anchor_query = torch.cat(_anchor_query, dim=1)
            attn_mask = torch.block_diag(*_attn_mask)
        else:
            attn_mask = None
            if mode in ['none', 'static']:
                label_query = model_label_query
                anchor_query = model_anchor_query

            elif mode in ['pure', 'mix']:
                label_query = enc_label_query
                anchor_query = enc_anchor_query
            
        return label_query, anchor_query, attn_mask
    
class Encoder(nn.Module):
    """Encoder."""

    def __init__(self,
                 n_layers: int = 3,
                 d_model: int = 768,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 activation: str = 'relu',
                 p_drop: float = 0.1,
                 add_pos_enc: str = 'add'):
        """Init."""
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         n_heads=n_heads,
                         d_ff=d_ff,
                         activation=activation,
                         add_pos_enc=add_pos_enc,
                         p_drop=p_drop) for _ in range(n_layers)
        ])

    def forward(self, x, pos_embed, mask=None):
        """Forward function."""
        for layer in self.layers:
            x = layer(x, pos_embed, mask)
        return x


class EncoderLayer(nn.Module):
    """Encoder Layer."""

    def __init__(self,
                 d_model: int = 768,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 activation: str = 'relu',
                 add_pos_enc: str = 'add',
                 p_drop: float = 0.1):
        """Init."""
        super().__init__()
        self.add_pos_enc = add_pos_enc
        if self.add_pos_enc == 'add':
            mha_d_model = d_model
        else:
            mha_d_model = d_model * 2

        self.mha = MHA(d_model=mha_d_model,
                       n_heads=n_heads,
                       p_drop=p_drop,
                       d_value=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_drop)

        self.ff = FeedForward(d_model=d_model,
                              d_ff=d_ff,
                              p_drop=p_drop,
                              activation=activation)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p_drop)

    def forward(self, x, pos_embed, mask=None):
        """Forward."""
        if self.add_pos_enc == 'add':
            q = k = x + pos_embed
        elif self.add_pos_enc == 'cat':
            q = k = torch.cat([x, pos_embed], dim=-1)
        else:
            q = k = torch.cat([x + pos_embed, pos_embed], dim=-1)
        
        # TODO: return only the embedding
        y, _ = self.mha(q=q, k=k, v=x, mask=mask)
        x = self.norm1(x + self.dropout1(y))

        y = self.ff(x)
        x = self.norm2(x + self.dropout2(y))
        return x


class Decoder(nn.Module):
    """Decoder."""

    def __init__(self,
                 n_layers: int = 3,
                 d_model: int = 768,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 activation: str = 'relu',
                 return_intermediate: bool = False,
                 p_drop: float = 0.1,
                 sa_position_mode: str = 'add',
                 ca_position_mode: str = 'cat',
                 query_scale_mode: str = 'diag',
                 temperature: int = 10000, 
                 modulate_wh_attn: bool = False):
        """Init.

        modulate_wh_attn: edit positional attention weight scale info(w, h)
        """
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         n_heads=n_heads,
                         d_ff=d_ff,
                         activation=activation,
                         p_drop=p_drop,
                         sa_position_mode=sa_position_mode,
                         ca_position_mode=ca_position_mode) for _ in range(n_layers)
        ])
        self.return_intermediate = return_intermediate

        # Transformer of positional embedding in cross-attnetion.
        # Ref: Conditional DETR
        # Currently, three mode are available(diag, identity, scalar)
        self.t = temperature
        if query_scale_mode == 'diag':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_mode == 'identity':
            self.query_scale = 1
        else:
            self.query_scale = MLP(d_model, d_model, 1, 2)

        # Use positional embedding of self-attention in first cross-attention layer
        # Ref: Conditional DETR
        for index in range(n_layers - 1):
            self.layers[index + 1].ca_qpos_proj = None

        # for positional embedding(P_q) in self-attnetion
        # sa_pos_head로 이름 변경하기.
        self.ref_point_head = MLP(d_model * 2, d_model, d_model, 2)

        # use shared parameter of transformer
        self.box_embed = None

        # to modulate cross-attention with scale info
        if modulate_wh_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        # self.look_forward_twice = look_forward_twice

    def forward(self, x, memory, pos_embed, anchor_query, mask=None, attn_mask=None):
        """Forward function.

        x: label query (batch_size x n_query x d_model)
        memory: encoder feature
        pos_embed: pos embeding for self-attention
        anchor_query: anchor query [batch, num_query, 4]
        mask: key padding mask
        """
        d_model = x.shape[-1]
        xs = []

        # Normalize to queries from 0 to 1.
        anchor = anchor_query.sigmoid()
        anchors = [anchor]
        for index, layer in enumerate(self.layers):
            # generate a pos embed with anchor
            query_sine_embed = gen_sineembed_for_position(anchor, self.t, d_model)

            # transform to get pos embed in self-attn
            sa_pos_embed = self.ref_point_head(query_sine_embed)

            # pos embed for cross-attn
            if index == 0 or self.query_scale == 1:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(x)
            ca_pos_embed = query_sine_embed[..., :d_model] * pos_transformation  # use xy embedding(half)
            if hasattr(self, 'ref_anchor_head'):
                wh_ref = self.ref_anchor_head(x).sigmoid()
                ca_pos_embed[..., d_model // 2:] *= (wh_ref[..., 0] / anchor[..., 2]).unsqueeze(-1)
                ca_pos_embed[..., :d_model // 2] *= (wh_ref[..., 1] / anchor[..., 3]).unsqueeze(-1)
                
            x = layer(x, memory, pos_embed, sa_pos_embed, ca_pos_embed, mask, attn_mask, is_first=index == 0)

            # iter update
            if self.box_embed is not None:
                # get offset from shared box_embed
                offset = self.box_embed(x)
                new_anchor = (inverse_sigmoid(anchor) + offset).sigmoid()

                # append anchors
                if index < len(self.layers) - 1:
                    anchors.append(new_anchor)
                anchor = new_anchor.detach()

            if self.return_intermediate:
                xs.append(x)

        # just use the final feature
        if not self.return_intermediate:
            xs.append(x)

        return xs, anchors


class DecoderLayer(nn.Module):
    """Decoder Layer."""

    def __init__(self,
                 d_model: int = 768,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 activation: str = 'relu',
                 p_drop: float = 0.1,
                 sa_position_mode: str = 'add',
                 ca_position_mode: str = 'add'):
        """Init."""
        super().__init__()
        self.sa_position_mode = sa_position_mode
        self.ca_position_mode = ca_position_mode

        if self.sa_position_mode == 'add':
            sa_mha_d_model = d_model
        else:
            sa_mha_d_model = d_model * 2

        if self.ca_position_mode == 'add':
            ca_mha_d_model = d_model
        else:
            ca_mha_d_model = d_model * 2

        # self-attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.mha1 = MHA(d_model=sa_mha_d_model,
                        n_heads=n_heads,
                        p_drop=p_drop,
                        d_value=d_model,
                        in_proj_flag=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_drop)

        # cross-attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.mha2 = MHA(d_model=ca_mha_d_model,
                        n_heads=n_heads,
                        p_drop=p_drop,
                        d_value=d_model,
                        in_proj_flag=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p_drop)

        # feed-forward network
        self.ff = FeedForward(d_model=d_model,
                              d_ff=d_ff,
                              p_drop=p_drop,
                              activation=activation)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p_drop)

    def forward(self, x, memory, pos_embed, sa_pos_embed, ca_pos_embed, mask=None, attn_mask=None, is_first=False):
        """Forward."""
        ##################
        # self-attention #
        ##################
        q_content = self.sa_qcontent_proj(x)
        q_pos = self.sa_qpos_proj(sa_pos_embed)
        k_content = self.sa_kcontent_proj(x)
        k_pos = self.sa_kpos_proj(sa_pos_embed)
        v = self.sa_v_proj(x)

        if self.sa_position_mode == 'add':
            q = q_content + q_pos
            k = k_content + k_pos
        elif self.sa_position_mode == 'cat':
            q = torch.cat([q_content, q_pos], dim=-1)
            k = torch.cat([k_content, k_pos], dim=-1)
        else:
            q = torch.cat([q_content + q_pos, q_pos], dim=-1)
            k = torch.cat([k_content + k_pos, k_pos], dim=-1)

        y, self_attention = self.mha1(q=q, k=k, v=v, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(y))

        ###################
        # cross-attention #
        ###################
        q_content = self.ca_qcontent_proj(x)
        k_content = self.ca_kcontent_proj(memory)
        k_pos = self.ca_kpos_proj(pos_embed)
        v = self.ca_v_proj(memory)

        # Ref: Conditional DETR
        # For the first decoder layer, we add the positional embedding predicted from
        # the object query (the positional embedding) into the original query in DETR.
        if is_first:
            q_pos = self.ca_qpos_proj(sa_pos_embed)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        ca_pos_embed = self.ca_qpos_sine_proj(ca_pos_embed)
        if self.ca_position_mode == 'add':
            q = q + ca_pos_embed
            k = k + k_pos
        elif self.ca_position_mode == 'cat':
            q = torch.cat([q, ca_pos_embed], dim=-1)
            k = torch.cat([k, k_pos], dim=-1)
        else:
            q = torch.cat([q + ca_pos_embed, ca_pos_embed], dim=-1)
            k = torch.cat([k + k_pos, k_pos], dim=-1)

        y, cross_attention = self.mha2(q=q, k=k, v=v, mask=mask)
        x = self.norm2(x + self.dropout2(y))

        ########################
        # feed-forward network #
        ########################
        y = self.ff(x)
        x = self.norm3(x + self.dropout3(y))
        return x

def gen_sineembed_for_position(pos_tensor, temperature=10000, d_model=256):
    """Generate sine embedding.

    - pos tensor has 3-dim(bs, n_query, 4)
    - return: (bs, n_query, d_model * 2)
    """
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / (d_model // 2))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    w_embed = pos_tensor[:, :, 2] * scale
    pos_w = w_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    h_embed = pos_tensor[:, :, 3] * scale
    pos_h = h_embed[:, :, None] / dim_t
    pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_y, pos_w, pos_h), dim=2)

    return pos