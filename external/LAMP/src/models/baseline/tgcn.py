from src.models.baseline.spatiotemporal import SpatioTemporalModel
import torch_geometric_temporal as tgt


class TGCN(SpatioTemporalModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, **kwargs):
        super().__init__(hidden_dim=hidden_dim, output_dim=output_dim, **kwargs)
        self.conv = tgt.nn.recurrent.TGCN(in_channels=input_dim, out_channels=hidden_dim)
