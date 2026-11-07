import torch
from src.layers.readout import get_predictor
from torch_geometric_temporal.nn.recurrent import EvolveGCNO, EvolveGCNH


class EvolveGCN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_function: str = "relu",
        level: str = "node",
        **kwargs,
    ):
        super().__init__()
        self.activation_function = getattr(torch, activation_function)
        self.readout = get_predictor(
            level=level,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def forward(self, snapshot, previous_hidden_state=None):
        h = self.conv(snapshot.x, snapshot.edge_index, edge_weight=snapshot.edge_attr)
        h = self.activation_function(h)
        return self.readout(h), h

    def reset_state(self):
        self.conv.weight = None


class EvolveGCN_H(EvolveGCN):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        normalize: bool = False,
        activation_function: str = "tanh",
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            activation_function=activation_function,
            **kwargs,
        )
        self.conv = EvolveGCNH(num_of_nodes=num_nodes, in_channels=input_dim, normalize=normalize)


class EvolveGCN_O(EvolveGCN):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        normalize: bool = False,
        activation_function: str = "relu",
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            activation_function=activation_function,
            **kwargs,
        )
        self.conv = EvolveGCNO(in_channels=input_dim, normalize=normalize)
