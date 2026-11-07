from collections import deque

import torch
from torch_geometric_temporal.nn.recurrent import TGCN
from src.models.baseline.spatiotemporal import SpatioTemporalModel


class A3TGCNConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(A3TGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.attention_1 = torch.nn.Linear(self.out_channels, self.out_channels)
        self.attention_2 = torch.nn.Linear(self.out_channels, 1)
        self.past_states = deque(maxlen=self.periods - 1)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.Tensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        current_state = self._base_tgcn(X, edge_index, edge_weight=edge_weight, H=H)
        H_concat = torch.stack([current_state] + list(self.past_states))
        attention_scores = self.attention_2(self.attention_1(H_concat.mean(dim=1)))
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=0).unsqueeze(dim=-1)
        self.past_states.append(current_state.detach())
        return torch.sum(H_concat * attention_scores, dim=0)

    def reset_state(self):
        """
        Reset the hidden state of the model.
        """
        self.past_states.clear()


class A3TGCN(SpatioTemporalModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        periods: int = 2,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, output_dim=output_dim, **kwargs)

        self.conv = A3TGCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            periods=periods,
        )

    def reset_state(self):
        """
        Reset the hidden state of the model.
        """
        self.conv.reset_state()
