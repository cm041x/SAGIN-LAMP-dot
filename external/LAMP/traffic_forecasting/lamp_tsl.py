from typing import Optional

import torch
from torch import nn
from torch.nn import Dropout

from tsl.nn.utils import get_layer_activation, maybe_cat_exog
from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.models import BaseModel

from src.layers.convolutions.lamp_conv import LAMPConv


class LAMP(BaseModel):
    """Temporal LAMP model — impulsive-ODE configuration.

    Each call to ``LAMPConv(x_t, h_t)`` applies an impulse
    ``h ← h + μ·x_t·U`` followed by ``num_layers`` LAMP forward-Euler
    steps. Between blocks the hidden state is forwarded as-is (optional
    residual + dropout) — no normalization or MLP that would break the
    ODE-flow interpretation. During the horizon phase the impulse driver
    is set to **zero**, so each horizon step is one autonomous LAMP step.
    """

    def __init__(
            self,
            input_size: int,
            input_window_size: int,
            horizon: int,
            n_nodes: int,
            hidden_size: int,
            step_size: float,
            output_size: int = None,
            exog_size: int = 0,
            num_iters: int = None,
            full_graph: bool = False,
            activation: str = "tanh",
            zero_init: bool = False,
            layer_dropout: float = 0.0,
            # LAMPConv-specific
            num_layers: int = 1,
            aggregation: str = "sum",
            dissipation: str = "sigmoid",
            gamma: float = 0.1,
            beta: float = 0.1,
            mu: float = 1.0,
            conv: str = "simple_antisym",
            discretization_method: str = "forward_euler",
            antisymmetric_weights: bool = True,
            use_bias: bool = True,
    ):
        super().__init__()

        # Data parameters
        self.input_size = input_size
        self.input_window_size = input_window_size
        self.horizon = horizon
        self.num_iters = (
            input_window_size + horizon if num_iters is None else num_iters
        )
        assert (
                self.num_iters >= self.horizon
        ), "num_iters should be at least equal to horizon"
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_size
        self.output_size = output_size
        self.exog_size = exog_size
        self.zero_init = zero_init

        # Model parameters
        self.step_size = step_size
        self.full_graph = full_graph
        self.activation = activation
        self.layer_dropout = layer_dropout

        # LAMP convolution parameters (some are retained only for config compatibility)
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.dissipation = dissipation
        self.gamma = gamma
        self.beta = beta
        self.mu = mu
        self.conv_kind = conv
        self.discretization_method = discretization_method
        self.antisymmetric_weights = antisymmetric_weights
        self.use_bias = use_bias

        # Raw feature dimensionality fed to LAMPConv's U projection.
        total_input_size = input_size + exog_size
        self.total_input_size = total_input_size

        # Node embeddings
        self.emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)

        # Init encoder builds h0 from the first window step.
        self.init_encoder = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            get_layer_activation(activation)(),
            nn.Linear(hidden_size, hidden_size),
        )

        def _make_conv(in_ch):
            # xi=1 reproduces the original impulse h <- h + mu * x_t @ U.
            return LAMPConv(
                in_channels=in_ch,
                hidden_channels=self.hidden_dim,
                L=self.num_layers,
                xi=1.0,
                epsilon=self.step_size,
                beta=self.beta,
                mu=self.mu,
                gamma=self.gamma,
                activation=self.activation,
            )

        # A single LAMP block driven by the raw external input.
        self.conv = _make_conv(total_input_size)
        # Dropout is allowed (regularization), but no BN / MLP — those would
        # break the ODE-flow interpretation.
        self.dropout = Dropout(p=self.layer_dropout)

        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            get_layer_activation(activation)(),
            nn.Linear(
                self.hidden_dim,
                self.output_size if self.output_size is not None else self.input_size,
            ),
        )

        print(f"Total parameters: {self.total_parameters()}")

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _run_block(
            self,
            h0: torch.Tensor,
            x_input: Optional[torch.Tensor],
            num_iters: int,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor],
    ):
        """Unroll the LAMPConv over ``num_iters`` time steps.

        While ``x_input`` has data the impulse term ``μ·x_t·U`` mixes new
        observations into ``H``. Once ``x_input`` is exhausted the impulse
        is set to **zero** (pure autonomous LAMP evolution).
        """
        conv = self.conv
        # External-input feature dim for this conv: U has shape
        # (in_channels, hidden_channels), so conv.U.shape[0] == in_channels.
        in_ch = conv.U.shape[0]
        h = h0
        states = [h]
        zero_impulse = None  # built lazily, matched to conv input dim
        for t in range(num_iters):
            if x_input is not None and x_input.size(1) > t:
                x_t = x_input[:, t]
            else:
                if zero_impulse is None:
                    zero_impulse = torch.zeros(
                        h.size(0), h.size(1), in_ch,
                        device=h.device, dtype=h.dtype,
                    )
                x_t = zero_impulse
            h = conv(x=x_t, h=h, edge_index=edge_index, edge_weight=edge_weight)
            states.append(h)
        return torch.stack(states, dim=1)

    def forward(
            self, x: torch.Tensor, edge_index=None, edge_weight=None, u=None
    ) -> torch.Tensor:

        # x: [batch, time_steps, nodes, features] - contains full history
        x = maybe_cat_exog(x, u)

        if self.full_graph or edge_index is None:
            num_nodes = x.size(-2)
            nodes = torch.arange(num_nodes, device=x.device)
            edge_index = torch.cartesian_prod(nodes, nodes).T

        # Raw window slice that drives the block via LAMPConv's U.
        x_window = x[:, : self.input_window_size]

        # Initial hidden state: zeros, or the first window step + node embedding.
        if self.zero_init:
            h0 = torch.zeros(
                x_window.size(0),
                x_window.size(2),
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            h0 = self.init_encoder(x[:, 0]) + self.emb()

        # Unroll the single block over the full horizon (impulse during the
        # window, autonomous afterwards), then read out the horizon tail.
        h = self._run_block(
            h0=h0,
            x_input=x_window,
            num_iters=self.num_iters,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        h = self.dropout(h)
        return self.readout(h[:, -self.horizon:])
