import torch
from typing import Optional, Union
from torch_geometric.data import Data, Batch
from src.layers.readout import get_predictor


class SpatioTemporalModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation_function: str = "tanh",
        level: str = "node",
        **kwargs,
    ):
        super().__init__()
        self.activation_function = getattr(torch, activation_function)
        self.readout = get_predictor(
            level=level,
            input_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(
        self,
        data: Union[Data, Batch],
        previous_hidden_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.conv(data.x, data.edge_index, edge_weight=data.edge_attr, H=previous_hidden_state)
        h = self.activation_function(h)
        return self.readout(h), h

    def reset_state(self):
        pass
