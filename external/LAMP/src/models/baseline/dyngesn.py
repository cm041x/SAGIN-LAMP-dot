import torch
from graphesn import DynamicGraphReservoir, initializer
from graphesn.util import compute_dynamic_graph_alpha

from src.layers.readout import get_predictor


class DynGESN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layer: int = 1,
        sigma: float = 0.99,
        leakage: float = 1.0,
        input_scaling: float = 1.0,
        bias_scaling: float = 1.0,
        activation_function: str = "tanh",
        level: str = "node",
        **kwargs,
    ):
        super().__init__()
        self.sigma = sigma
        self.leakage = leakage
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.reservoir = DynamicGraphReservoir(
            num_layers=n_layer,
            in_features=input_dim,
            hidden_features=hidden_dim,
            bias=True,
            activation=activation_function,
            return_sequences=False,
        )
        self.readout = get_predictor(
            level=level,
            input_dim=hidden_dim,
            output_dim=output_dim,
        )

    def initialize(self, dataset):
        alpha = compute_dynamic_graph_alpha([dataset.train_data[0][0].edge_index])
        self.reservoir.initialize_parameters(
            recurrent=initializer("uniform", sigma=self.sigma / alpha),
            input=initializer("uniform", scale=self.input_scaling),
            bias=initializer("uniform", scale=self.bias_scaling),
            leakage=self.leakage,
        )

    def forward(self, edge_index_seq, input_seq):
        h = self.reservoir(edge_index=edge_index_seq, input=input_seq).detach()
        return self.readout(h)

    def reset_state(self):
        pass
