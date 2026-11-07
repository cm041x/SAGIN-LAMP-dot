import math
from typing import Optional

import torch
import torch_geometric as tg
import torch_scatter as ts
from torch import Tensor

from src.layers.convolutions.simple_conv import ConstrainedMessagePassing
from src.layers.discretizations.euler import ForwardEuler
from src.layers.parameterizations.dissipative_antisymmetric import (
    DissipativeAntiSymmetric,
)


def normalized_adjacency(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    normalization: str,
) -> tuple[Tensor, Tensor]:
    """Symmetric- or antisymmetric-normalized (and negated) adjacency.

    - ``"sym"``:     ``-D^{-1/2} A D^{-1/2}``
    - ``"antisym"``: ``-D^{-1} A + (D^{-1} A)^T``

    :param edge_index: Edge indices of shape ``(2, num_edges)``.
    :param edge_weight: Edge weights, or ``None`` for an unweighted graph.
    :param normalization: Either ``"sym"`` or ``"antisym"``.
    :return: ``(edge_index, normalized_edge_weight)``.
    """
    edge_index, edge_weight = tg.utils.remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = tg.utils.to_undirected(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    num_nodes = tg.utils.num_nodes.maybe_num_nodes(edge_index)
    row, col = edge_index
    node_degrees = ts.scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).float()

    if normalization == "sym":
        deg_inv_sqrt = node_degrees.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    else:
        deg_inv = 1.0 / node_degrees
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        edge_weight = -deg_inv[row] * edge_weight + edge_weight * deg_inv[col]
    return edge_index, edge_weight


class LAMPConv(torch.nn.Module):
    r"""One LAMP convolution: impulse injection followed by L forward-Euler steps.

    Given a hidden state ``h`` and the current input ``x``, the cell first injects
    the input into the memory (:meth:`impulse_map`, the ``xi`` rule) and then runs
    :attr:`L` forward-Euler steps of the antisymmetric LAMP dynamics: a
    skew-symmetric mixing ``W - W.T`` damped by the adaptive term ``Gamma``, a
    symmetric coupling ``S`` and a ``beta``-weighted antisymmetric coupling ``K``
    (:meth:`continuous_flow`).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        L: int,
        xi: float,
        epsilon: float = 0.1,
        beta: float = 0.1,
        mu: float = 1.0,
        gamma: float = 0.1,
        activation: str = "tanh",
        dissipation: str = "sigmoid",
        antisymmetric_weights: bool = True,
    ) -> None:
        """
        :param in_channels: Input feature size (mapped to ``hidden_channels`` by
            the impulse-map matrix ``U``).
        :param hidden_channels: Hidden-state size operated on by the dynamics.
        :param L: Number of forward-Euler steps per call.
        :param xi: Impulse-map divisor (``xi = 1`` sum, ``xi = 2`` mean).
        :param epsilon: Forward-Euler step size.
        :param beta: Weight of the antisymmetric (``K``) coupling.
        :param mu: Scaling of the impulse ``mu * x @ U``.
        :param gamma: Initial value of the damping ``Gamma``.
        :param activation: Pointwise activation in the Euler update.
        :param dissipation: ``"sigmoid"`` (learned ``Gamma``) or ``"fixed"``.
        :param antisymmetric_weights: If ``False``, ``W``, ``S`` and ``K`` are
            unconstrained and ``Gamma`` is not applied.
        """
        super().__init__()
        self.L = L
        self.antisymmetric_weights = antisymmetric_weights
        self.xi = xi
        self.mu = mu
        self.beta = beta
        self.epsilon = epsilon

        # Impulse-map injection matrix U: maps inputs into the hidden space.
        self.U = torch.nn.Parameter(torch.empty(in_channels, hidden_channels))
        # Skew-symmetric mixing weight W and additive bias.
        self.W = torch.nn.Parameter(torch.empty(hidden_channels, hidden_channels))
        self.bias = torch.nn.Parameter(torch.empty(hidden_channels))
        # Symmetric (S) and antisymmetric (K) edge couplings.
        self.S = ConstrainedMessagePassing(hidden_channels, antisym=True)
        self.K = ConstrainedMessagePassing(hidden_channels, sym=True)
        if not antisymmetric_weights:
            self.K = ConstrainedMessagePassing(hidden_channels)
            self.S = ConstrainedMessagePassing(hidden_channels)
        self.reset_mixing_parameters()
        # Skew-symmetric mixing with adaptive damping: W - W.T - Gamma (one forward call).
        self.mixing = DissipativeAntiSymmetric(
            hidden_channels, gamma=gamma, dissipation=dissipation
        )
        # Forward-Euler discretization with step size epsilon.
        self.discretization = ForwardEuler(epsilon=epsilon, activation=activation)

        self.reset_parameters()

    def reset_mixing_parameters(self) -> None:
        # Kaiming-uniform for W; bias uniform, scaled by the W fan-in.
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self) -> None:
        self.reset_mixing_parameters()
        torch.nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inject the input into ``h`` (impulse map), then run the continuous flow.

        :param x: Current input features.
        :param h: Previous hidden state.
        :param edge_index: Graph connectivity.
        :param edge_weight: Optional edge weights.
        :return: The updated hidden state after L Euler steps.
        """
        h = self.impulse_map(x, h)
        h = self.continuous_flow(h, edge_index, edge_weight)
        return h

    def impulse_map(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (mu * x @ U + h) / xi   (xi = 1 -> sum, xi = 2 -> mean).
        return (self.mu * x @ self.U + h) / self.xi

    def continuous_flow(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run L forward-Euler steps of the antisymmetric LAMP dynamics.

        Each step combines the skew-symmetric mixing ``h @ (W - W.T - Gamma).T``,
        the symmetric coupling ``S`` and the ``beta``-weighted antisymmetric
        coupling ``K``, then advances ``h`` by one Euler step.

        :param h: Hidden state after the impulse injection.
        :param edge_index: Graph connectivity.
        :param edge_weight: Optional edge weights.
        :return: The hidden state after L steps.
        """
        # Symmetric / antisymmetric normalized adjacencies (computed once).
        antisym_ei, antisym_ew = normalized_adjacency(edge_index, edge_weight, "antisym")
        sym_ei, sym_ew = normalized_adjacency(edge_index, edge_weight, "sym")
        # Skew-symmetric mixing damped by Gamma (constant across the L steps).
        skew = self.mixing(self.W) if self.antisymmetric_weights else self.W
        for _ in range(self.L):
            linear = h @ skew.T
            S = self.S(h, edge_index=sym_ei, edge_weight=sym_ew)
            K = self.beta * self.K(h, antisym_ei, antisym_ew)
            flow = linear + S + K + self.bias
            h = self.discretization(h, flow)
        return h
