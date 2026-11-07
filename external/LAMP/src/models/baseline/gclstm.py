import torch
from torch_geometric.utils import get_laplacian
import torch_geometric_temporal as tgt
from src.layers.readout import get_predictor


class GCLSTM(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        K: int = 2,
        normalization: str = None,
        activation_function: str = "tanh",
        level: str = "node",
        **kwargs,
    ):
        super().__init__()
        self.normalization = normalization
        self.activation_function = getattr(torch, activation_function)
        self.conv = tgt.GCLSTM(
            in_channels=input_dim,
            out_channels=hidden_dim,
            K=K,
            normalization=normalization,
        )
        self.readout = get_predictor(
            level=level,
            input_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, snapshot, previous_hidden_state=None):
        h, c = previous_hidden_state if previous_hidden_state is not None else (None, None)

        _, _edge_weight = get_laplacian(snapshot.edge_index, normalization=self.normalization)
        h, c = self.conv(
            snapshot.x,
            snapshot.edge_index,
            edge_weight=snapshot.edge_attr,
            H=h,
            C=c,
            lambda_max=_edge_weight.max(),
        )
        h = self.activation_function(h)
        return self.readout(h), torch.stack((h, c), dim=0)

    def reset_state(self):
        pass
