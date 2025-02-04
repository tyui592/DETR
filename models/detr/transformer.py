# -*- coding: utf-8 -*-
"""Transformer Code."""

import torch
import torch.nn as nn
from .layers import MHA, FeedForward


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
                              decoder_ca_position_mode=args.decoder_ca_position_mode)
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
                 decoder_ca_position_mode: str = 'add'):
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
                               ca_position_mode=decoder_ca_position_mode)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, f, pos_embed, query, mask):
        """Forward function."""
        memory, enc_sa = self.encoder(f, pos_embed, mask)

        x = torch.zeros_like(query)
        hs, dec_sa, dec_ca = self.decoder(x, memory, pos_embed, query, mask)

        return hs, enc_sa, dec_sa, dec_ca


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
        """Forward."""
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
                 ca_position_mode: str = 'cat'):
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

    def forward(self, x, memory, pos_embed, query, mask=None):
        """Forward."""
        xs = []
        for layer in self.layers:
            x, sa, ca = layer(x, memory, pos_embed, query, mask)
            if self.return_intermediate:
                xs.append(x)

        if not self.return_intermediate:
            xs.append(x)

        return xs, sa, ca


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
        self.mha1 = MHA(d_model=sa_mha_d_model,
                        n_heads=n_heads,
                        p_drop=p_drop,
                        d_value=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_drop)

        # cross-attention
        self.mha2 = MHA(d_model=ca_mha_d_model,
                        n_heads=n_heads,
                        p_drop=p_drop,
                        d_value=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.ff = FeedForward(d_model=d_model,
                              d_ff=d_ff,
                              p_drop=p_drop,
                              activation=activation)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p_drop)

    def forward(self, x, memory, pos_embed, query, mask=None):
        """Forward."""
        # self-attention
        if self.sa_position_mode == 'add':
            q = k = x + query
        elif self.sa_position_mode == 'cat':
            q = k = torch.cat([x, query], dim=-1)
        else:
            q = k = torch.cat([x + query, query], dim=-1)

        y, self_attention = self.mha1(q=q, k=k, v=x)
        x = self.norm1(x + self.dropout1(y))

        # cross-attention
        if self.ca_position_mode == 'add':
            q = x + query
            k = memory + pos_embed
        elif self.ca_position_mode == 'cat':
            q = torch.cat([x, query], dim=-1)
            k = torch.cat([memory, pos_embed], dim=-1)
        else:
            q = torch.cat([x + query, query], dim=-1)
            k = torch.cat([memory + pos_embed, pos_embed], dim=-1)

        y, cross_attention = self.mha2(q=q, k=k, v=memory, mask=mask)
        x = self.norm2(x + self.dropout2(y))

        # feed-forward
        y = self.ff(x)
        x = self.norm3(x + self.dropout3(y))
        return x, self_attention, cross_attention
