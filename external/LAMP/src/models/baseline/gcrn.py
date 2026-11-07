from torch_geometric.utils import get_laplacian
from torch_geometric_temporal import GConvLSTM, GConvGRU
from torch_geometric.data import Data

from src.models.baseline.spatiotemporal import SpatioTemporalModel
from typing import Optional
import torch


class GCRN_LSTM(SpatioTemporalModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        K: int = 2,
        normalization: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, output_dim=output_dim, **kwargs)
        self.normalization = normalization
        self.conv = GConvLSTM(
            in_channels=input_dim,
            out_channels=hidden_dim,
            K=K,
            normalization=normalization,
        )

    def forward(
        self, data: Data, previous_hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if previous_hidden_state is None:
            h, c = None, None
        else:
            h, c = previous_hidden_state

        _, _edge_weight = get_laplacian(
            data.edge_index, edge_weight=data.edge_attr, normalization=self.normalization
        )
        h, c = self.conv(
            data.x,
            data.edge_index,
            edge_weight=data.edge_attr,
            H=h,
            C=c,
            lambda_max=_edge_weight.max(),
        )
        h = self.activation_function(h)
        return self.readout(h), torch.stack((h, c))


class GCRN_GRU(SpatioTemporalModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        K: int = 2,
        normalization: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, output_dim=output_dim, **kwargs)
        self.normalization = normalization
        self.conv = GConvGRU(
            in_channels=input_dim,
            out_channels=hidden_dim,
            K=K,
            normalization=normalization,
        )

    def forward(
        self, data: Data, previous_hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _edge_weight = get_laplacian(
            data.edge_index, edge_weight=data.edge_attr, normalization=self.normalization
        )
        h = self.conv(
            data.x,
            data.edge_index,
            edge_weight=data.edge_attr,
            H=previous_hidden_state,
            lambda_max=_edge_weight.max(),
        )
        h = self.activation_function(h)
        return self.readout(h), h
