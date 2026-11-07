"""LAMP propagation adapted from MIT-licensed upstream 69f6eb5.

Upstream: Copyright (c) 2025 Fabrizio De Castelli, MIT license; see
external/LAMP/LICENSE. The implementation keeps upstream matrix conventions,
using PyTorch index_add instead of torch_scatter and explicit node counts.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrize import register_parametrization
from torch_geometric.utils import remove_self_loops, to_undirected


class AntiSymmetric(nn.Module):
    def forward(self, weight: Tensor) -> Tensor:
        upper = weight.triu(diagonal=1)
        return upper - upper.T

    def right_inverse(self, weight: Tensor) -> Tensor:
        return weight.triu(diagonal=1)


class Symmetric(nn.Module):
    def forward(self, weight: Tensor) -> Tensor:
        upper = weight.triu()
        return upper + upper.T  # Upstream doubles the diagonal as well.

    def right_inverse(self, weight: Tensor) -> Tensor:
        return weight.triu()


def normalized_adjacency(edge_index: Tensor, edge_weight: Optional[Tensor],
                         normalization: str, num_nodes: int) -> tuple[Tensor, Tensor]:
    if normalization not in ('sym', 'antisym'):
        raise ValueError('Unknown normalization')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    row, col = edge_index
    degrees = edge_weight.new_zeros(num_nodes).index_add(0, row, edge_weight).float()
    if normalization == 'sym':
        inv = degrees.pow(-0.5)
        inv.masked_fill_(torch.isinf(inv), 0)
        weights = -inv[row] * edge_weight * inv[col]
    else:
        inv = degrees.reciprocal()
        inv.masked_fill_(torch.isinf(inv), 0)
        weights = -inv[row] * edge_weight + edge_weight * inv[col]
    return edge_index, weights


class ConstrainedMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, *, antisym: bool = False, sym: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if antisym:
            register_parametrization(self.linear, 'weight', AntiSymmetric())
        if sym:
            register_parametrization(self.linear, 'weight', Symmetric())

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        source, target = edge_index
        messages = self.linear(x)[source] * edge_weight.unsqueeze(-1)
        return x.new_zeros(x.shape).index_add(0, target, messages)


class AdaptiveDissipativity(nn.Module):
    def __init__(self, hidden_dim: int, gamma: float) -> None:
        super().__init__()
        if not 0 < gamma < 1:
            raise ValueError('gamma must be strictly between zero and one')
        self.diagonal = nn.Parameter(torch.full((hidden_dim,), math.log(gamma / (1-gamma))))

    def forward(self) -> Tensor:
        return torch.diag(torch.sigmoid(self.diagonal))


class DissipativeAntiSymmetric(nn.Module):
    def __init__(self, hidden_dim: int, gamma: float) -> None:
        super().__init__()
        self.dissipativity = AdaptiveDissipativity(hidden_dim, gamma)

    def forward(self, weight: Tensor) -> Tensor:
        return weight - weight.T - self.dissipativity()


class LAMPAdapter(nn.Module):
    """Recurrent graph cell; memory is supplied and owned by its caller."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, L: int = 3,
                 epsilon: float = 0.1, beta: float = 0.1, mu: float = 1.0,
                 xi: float = 1.0, initial_damping: float = 0.1) -> None:
        super().__init__()
        if L < 0 or xi <= 0 or epsilon < 0:
            raise ValueError('L, xi or epsilon outside supported range')
        self.hidden_dim, self.L = hidden_dim, L
        self.epsilon, self.beta, self.mu, self.xi = epsilon, beta, mu, xi
        self.U = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.empty(hidden_dim))
        self.S = ConstrainedMessagePassing(hidden_dim, antisym=True)
        self.K = ConstrainedMessagePassing(hidden_dim, sym=True)
        self.mixing = DissipativeAntiSymmetric(hidden_dim, initial_damping)
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        bound = 1 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

    def damping(self, context: Optional[Tensor], num_nodes: int) -> Tensor:
        return torch.sigmoid(self.mixing.dissipativity.diagonal).expand(num_nodes, -1)

    def forward_snapshot(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor],
                         previous_hidden: Optional[Tensor], node_ids: Tensor,
                         context: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        n = x.shape[0]
        if node_ids.shape != (n,) or torch.unique(node_ids).numel() != n:
            raise ValueError('node_ids must uniquely identify every node')
        if previous_hidden is None:
            previous_hidden = x.new_zeros((n, self.hidden_dim))
        if previous_hidden.shape != (n, self.hidden_dim):
            raise ValueError('previous_hidden must be aligned to node_ids')
        if previous_hidden.dtype != x.dtype or previous_hidden.device != x.device:
            raise ValueError('previous_hidden dtype/device must match x')
        h = (self.mu * x @ self.U + previous_hidden) / self.xi
        sym_i, sym_w = normalized_adjacency(edge_index, edge_weight, 'sym', n)
        anti_i, anti_w = normalized_adjacency(edge_index, edge_weight, 'antisym', n)
        damping = self.damping(context, n)
        for _ in range(self.L):
            flow = (h @ (self.W-self.W.T).T - h*damping
                    + self.S(h, sym_i, sym_w) + self.beta*self.K(h, anti_i, anti_w) + self.bias)
            h = h + self.epsilon*torch.tanh(flow)
        return h, h
