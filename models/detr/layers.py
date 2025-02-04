# -*- coding: utf-8 -*-
"""Model Code."""

import torch
import torch.nn as nn
from typing import Optional


class Activation(nn.Module):
    """Activation Module."""

    def __init__(self, activation: str):
        """Init."""
        super(Activation, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        """Forward."""
        return self.activation(x)


class MHA(nn.Module):
    """Multi-Head Attention Module."""

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 p_drop: float = 0.1,
                 d_value: Optional[int] = None):
        """Init."""
        super().__init__()
        self.n_heads = n_heads
        if d_value is None:
            d_value = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_value, d_value)
        self.w_o = nn.Linear(d_value, d_value)
        self.dropout = nn.Dropout(p_drop)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.constant_(self.w_q.bias, 0.0)
        nn.init.constant_(self.w_k.bias, 0.0)
        nn.init.constant_(self.w_v.bias, 0.0)
        nn.init.constant_(self.w_o.bias, 0.0)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """Forward.

        mask: boolean mask to ignore future words in decoder during training.
        """
        #  q: (batch) x (# of tokens) x (d_model)
        batch_size, _, d_model = q.shape
        d_head = d_model // self.n_heads
        scale = d_model / self.n_heads
        d_value = v.shape[-1]
        d_head_v = d_value // self.n_heads

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Current Tensor Dim: (batch) x (# of tokens) x (d_model)
        # (1) split (d_model) to (n_heads) x (head_dim)
        #      -> (batch) x (# of tokens) x (n_heads) x (head_dim)
        # (2) transpose dimensions for multiplication
        #      -> (batch) x (n_heads) x (# of tokens) x (head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, d_head).transpose(1, 2).contiguous()
        K = K.view(batch_size, -1, self.n_heads, d_head).transpose(1, 2).contiguous()
        V = V.view(batch_size, -1, self.n_heads, d_head_v).transpose(1, 2).contiguous()

        # calculate attention weights,
        #   Dim: (batch) x (n_heads) x (# of Q tokens) x (# of K tokens)
        attention_score = torch.matmul(Q, K.transpose(2, 3)) / scale ** 0.5

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e10)
        attention_weight = torch.softmax(attention_score, dim=-1)

        # y: (batch) x (n_heads) x (# of tokens) x (head_dim)
        y = torch.matmul(self.dropout(attention_weight), V)

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, -1, d_value)
        y = self.w_o(y)
        return y, attention_weight


class FeedForward(nn.Module):
    """Feed forward module."""

    def __init__(self,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 p_drop: float = 0.1,
                 activation: str = 'relu'):
        """init."""
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = Activation(activation)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        """Forward."""
        x = self.dropout(self.activation(self.w1(x)))
        x = self.w2(x)
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 activation='relu'):
        """Init."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = Activation(activation)

    def forward(self, x):
        """Forward."""
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
