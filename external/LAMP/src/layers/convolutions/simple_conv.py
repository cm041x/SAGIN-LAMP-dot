from typing import Optional
import torch
import torch_geometric as tg
from torch.nn.utils.parametrize import register_parametrization

from src.layers.parameterizations.antisymmetric import AntiSymmetric
from src.layers.parameterizations.symmetric import Symmetric


class ConstrainedMessagePassing(tg.nn.MessagePassing):
    """Edge-weighted graph convolution with a parametrized linear map.

    Applies a linear transform whose weight is optionally constrained to be
    antisymmetric (``antisym``) or symmetric (``sym``) via
    ``torch.nn.utils.parametrize``, then sums neighbour messages. Used for the
    ``S`` (symmetric) and ``K`` (antisymmetric) couplings of the LAMP dynamics.
    """

    def __init__(
        self,
        in_channels: int,
        antisym: bool = False,
        sym: bool = False,
    ) -> None:
        """
        :param in_channels: Node feature size.
        :param antisym: Constrain the weight to be skew-symmetric.
        :param sym: Constrain the weight to be symmetric.
        """
        # the aggregation function is set to 'add' by default
        super().__init__(aggr="add")
        self.linear = torch.nn.Linear(in_channels, in_channels, bias=False)
        # constrains the weights to be antisymmetric or symmetric
        if antisym:
            register_parametrization(self.linear, "weight", AntiSymmetric())
        if sym:
            register_parametrization(self.linear, "weight", Symmetric())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs message passing using the linear transformation. Uses attention weights if provided.
        The propagate method is used with the adjacency matrix and the edge weights, starting from the input x.

        :param x: Input node features of shape (num_nodes, in_channels).
        :param edge_index: Edge index tensor of shape (2, num_edges).
        :param edge_weight: Optional edge weight tensor of shape (num_edges,).
        """
        # \sum_{u in N(v)} (\hat{A}_vu + \hat{A}_uv) * (V - V^T) * h_u(t)
        out = self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=self.linear(x))
        return out

    def message(
        self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Weight each neighbour message ``x_j`` by its (scalar or vector) edge weight."""
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j
