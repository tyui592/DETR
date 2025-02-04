# -*- coding: utf-8 -*-
"""Transformer Code."""

import math
import torch
import torch.nn as nn
from .layers import MHA, FeedForward, MLP


def get_transformer(args):
    """Get the transformer model."""
    transformer = Transformer(num_encoder_layers=args.n_encoder_layers,
                              num_decoder_layers=args.n_decoder_layers,
                              d_model=args.d_model,
                              n_heads=args.n_heads,
                              d_ff=args.d_ff,
                              return_intermediate=args.return_intermediate,
                              activation=args.activation,
                              p_drop=args.p_drop,
                              encoder_position_mode=args.encoder_position_mode,
                              decoder_sa_position_mode=args.decoder_sa_position_mode,
                              decoder_ca_position_mode=args.decoder_ca_position_mode,
                              query_scale_mode=args.query_scale_mode)
    return transformer


class Transformer(nn.Module):
    """Transformer."""

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
                 query_scale_mode: str = 'diag'):
        """Initialize."""
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
                               query_scale_mode=query_scale_mode)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, f, pos_embed, query, mask):
        """Forward function."""
        memory, enc_sa = self.encoder(f, pos_embed, mask)

        x = torch.zeros_like(query)
        h_s, ref_points, dec_sa, dec_ca = self.decoder(x, memory, pos_embed, query, mask)

        return h_s, ref_points, enc_sa, dec_sa, dec_ca


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
        """Init.

        * add_pos_enc: Flag for attention
            - 'add': add
            - 'cat': concatenate feature and positional encoding
        """
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
            x, sa = layer(x, pos_embed, mask)
        return x, sa


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

        y, attention = self.mha(q=q, k=k, v=x, mask=mask)
        x = self.norm1(x + self.dropout1(y))

        y = self.ff(x)
        x = self.norm2(x + self.dropout2(y))
        return x, attention


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
                 ):
        """Init.

        module_flag: Flag to select sub-module
            - 0: original decoder
            - 1: concatenate feature and pos embedding
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
        if query_scale_mode == 'diag':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_mode == 'identity':
            self.query_scale = 1
        elif query_scale_mode == 'scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)

        # Use positional embedding of self-attention in first cross-attention layer
        # Ref: Conditional DETR
        for index in range(n_layers - 1):
            self.layers[index + 1].ca_qpos_proj = None

        # Head to generate 2d ref-point('s')
        self.ref_point_head = MLP(d_model, d_model, 2, 2)

    def forward(self, x, memory, pos_embed, query, mask=None):
        """Forward."""
        xs = []
        ref_points = self.ref_point_head(query).sigmoid()  # batch, num_query, d_model

        for index, layer in enumerate(self.layers):
            if index == 0 or self.query_scale == 1:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(x)
            query_sine_embed = gen_sineembed_for_position(ref_points)
            query_sine_embed *= pos_transformation

            x, sa, ca = layer(x, memory, pos_embed, query, query_sine_embed, mask, is_first=index == 0)

            if self.return_intermediate:
                xs.append(x)

        if not self.return_intermediate:
            xs.append(x)

        return xs, ref_points, sa, ca


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

        self.ff = FeedForward(d_model=d_model,
                              d_ff=d_ff,
                              p_drop=p_drop,
                              activation=activation)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p_drop)

    def forward(self, x, memory, pos_embed, sa_pos_embed, ca_pos_embed, mask=None, is_first=False):
        """Forward."""
        # self-attention
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

        y, self_attention = self.mha1(q=q, k=k, v=v)
        x = self.norm1(x + self.dropout1(y))

        # cross-attention
        q_content = self.ca_qcontent_proj(x)
        k_content = self.ca_kcontent_proj(memory)
        k_pos = self.ca_kpos_proj(pos_embed)
        v = self.ca_v_proj(memory)

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

        # feed-forward
        y = self.ff(x)
        x = self.norm3(x + self.dropout3(y))
        return x, self_attention, cross_attention


def gen_sineembed_for_position(pos_tensor):
    """Generate sine embedding."""
    # n_query, bs, 2 = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_x, pos_y), dim=2)
    return pos
