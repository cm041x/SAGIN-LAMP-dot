"""The LAMP recurrent cell (Long-range Adaptive Memory Propagation).

LAMP injects the external input into a hidden memory state and then evolves that
state with the antisymmetric LAMP dynamics — skew-symmetric weights, an adaptive
damping term Gamma, and symmetric/antisymmetric graph couplings — discretized
with forward Euler. The state evolution itself lives in :class:`LAMPConv`; this
module wraps it with the input-injection step and a task-specific readout head.
"""

from typing import Optional, Union

import torch
from torch_geometric.data import Batch, Data

from src.layers.convolutions.lamp_conv import LAMPConv
from src.layers.readout import get_predictor


class LAMP(torch.nn.Module):
    """Long-range Adaptive Memory Propagation Impulsive Graph Neural ODE with a readout head.

    The hidden memory state is injected with the current input, propagated over
    the graph by the antisymmetric LAMP dynamics, and decoded by a node-, link-
    or graph-level readout.

    :param input_dim: Number of input features per node.
    :param output_dim: Number of output features (task targets).
    :param hidden_dim: Size of the hidden memory state.
    :param L: Number of forward-Euler propagation steps.
    :param xi: Impulse-map divisor (``xi = 1`` sum, ``xi = 2`` mean).
    :param beta: Weight of the antisymmetric (K) edge coupling.
    :param gamma: Initial value of the damping Gamma.
    :param epsilon: Forward-Euler step size.
    :param mu: Scaling of the impulse ``mu * X @ U``.
    :param activation: Pointwise activation inside the LAMP iteration.
    :param level: Readout level (``"node"`` or ``"link"``).
    :param dissipation: ``"sigmoid"`` (learned ``Gamma``) or ``"fixed"``.
    :param antisymmetric_weights: Constrain ``W``, ``S`` and ``K`` (``False`` for
        the unconstrained ablation).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        L: int,
        xi: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1,
        epsilon: float = 0.1,
        mu: float = 1.0,
        activation: Optional[str] = "tanh",
        level: str = "node",
        dissipation: str = "sigmoid",
        antisymmetric_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # Propagation core.
        self.conv = LAMPConv(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            L=L,
            xi=xi,
            beta=beta,
            epsilon=epsilon,
            mu=mu,
            gamma=gamma,
            activation=activation,
            dissipation=dissipation,
            antisymmetric_weights=antisymmetric_weights,
        )

        # Task head mapping the hidden state to predictions.
        self.readout = get_predictor(
            level=level,
            input_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(
        self, data: Union[Data, Batch], prev_h: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject the input, propagate the memory, and decode predictions.

        :param data: Graph snapshot with ``x`` and ``edge_index`` (and optional
            ``edge_attr`` / ``link_pred_ids``).
        :param prev_h: Previous hidden state, or ``None`` to start from zeros.
        :return: ``(y, h)`` — the predictions and the updated hidden state.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr
        link_pred_ids = getattr(data, "link_pred_ids", None)

        # Start from a zero memory when no previous state is provided.
        if prev_h is None:
            prev_h = torch.zeros((*x.shape[:-1], self.hidden_dim)).to(x.device)

        # One propagation step (L inner Euler steps) followed by the readout.
        h = self.conv(x, prev_h, edge_index, edge_weight=edge_weight)
        y = self.readout(h, link_pred_ids=link_pred_ids)

        return y, h

    def reset_state(self) -> None:
        pass
