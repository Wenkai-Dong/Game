# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from torch import nn as nn
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.attention import sdpa_kernel, SDPBackend


class GatedMHA(nn.Module):
    """Multi-Head Attention Network (MHA).

    The MHA network is a sequence of Multi-Head Attention, Add layers, Norm layers and optional activation functions.
    The final output can be flattened.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: int = 0.0,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        gated_position: str | None = None, # "key", "value", "sdpa", "dense"
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.gated = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            nn.Sigmoid()
        )
        self._reset_parameters()

        self.gated_position = gated_position

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.gated[0].weight)

        constant_(self.q_proj.bias, 0.0)
        constant_(self.k_proj.bias, 0.0)
        constant_(self.v_proj.bias, 0.0)
        constant_(self.o_proj.bias, 0.0)
        constant_(self.gated[0].bias, 1.0)

    def forward(
        self,
        query: Tensor,  # (N, L, E)
        key: Tensor,    # (N, S, E)
        value: Tensor,  # (N, S, E)
        need_weights: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        batch_size, target_seq_len, _ = query.shape
        _, source_seq_len, _ = key.shape

        q = self.q_proj(query).view(batch_size, target_seq_len, self.num_heads, self.head_dim).transpose(1, 2)   # (N, H, L, E/H)
        k_full = self.k_proj(key)   # (N, S, E)
        v_full = self.v_proj(value)   # (N, S, E)

        if self.gated_position == "key":
            k_full = self.gated(key) * k_full
        elif self.gated_position == "value":
            v_full = self.gated(value) * v_full
        k = k_full.view(batch_size, source_seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (N, H, S, E/H)
        v = v_full.view(batch_size, source_seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (N, H, S, E/H)

        if need_weights:
            attn_output_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1).mean(dim=1)
        else:
            attn_output_weights = None

        output = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2) # (N, L, E)
        if self.gated_position == "sdpa":
            output = self.gated(query) * output
        attn_output = self.o_proj(output)
        if self.gated_position == "dense":
            attn_output = self.gated(query) * attn_output

        return attn_output, attn_output_weights