"""Node-wise lifecycle-conditioned damping over the upstream LAMP flow."""
from typing import Optional
import torch
from torch import Tensor, nn
from .upstream_adapter import LAMPAdapter


class LifecycleLAMP(LAMPAdapter):
    def __init__(self, input_dim: int, context_dim: int, **kwargs) -> None:
        super().__init__(input_dim, **kwargs)
        self.context_dim = context_dim
        # Do not consume the common encoder/head RNG stream: plain and lifecycle
        # variants can start with identical shared parameters under one seed.
        with torch.random.fork_rng(devices=[]):
            self.conditioner = nn.Sequential(nn.Linear(context_dim, 32), nn.Tanh(), nn.Linear(32, self.hidden_dim))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def damping(self, context: Optional[Tensor], num_nodes: int) -> Tensor:
        if context is None or context.shape != (num_nodes, self.context_dim):
            raise ValueError('lifecycle context is required per node')
        logits = self.mixing.dissipativity.diagonal.unsqueeze(0)
        # Context index 0 is the UAV type indicator. Infrastructure uses shared damping.
        adjustment = self.conditioner(context) * context[:, :1]
        return torch.sigmoid(logits + adjustment)
