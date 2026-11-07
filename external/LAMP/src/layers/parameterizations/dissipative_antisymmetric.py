import torch

from src.layers.attention.dissipative import AdaptiveDissipativity


class DissipativeAntiSymmetric(torch.nn.Module):
    r"""Parametrization ``W -> W - W^T - Gamma``: anti-symmetric mixing with adaptive damping.

    This is the LAMP mixing operator. ``W - W^T`` is skew-symmetric, hence purely
    oscillatory (imaginary eigenvalues, a non-expansive rotation); subtracting the learned
    diagonal dissipativity ``Gamma = diag(sigmoid(d))`` pushes every eigenvalue's real part
    ``<= 0``, giving the stable, non-expansive Jacobian the LAMP dynamics rely on for
    long-range propagation. The damping term is owned here, so the whole
    ``W - W^T - Gamma`` map is a single ``forward`` call.
    """

    def __init__(
        self, hidden_channels: int, gamma: float = 0.1, dissipation: str = "sigmoid"
    ) -> None:
        """
        :param hidden_channels: Size of the (square) mixing matrix / damping diagonal.
        :param gamma: Initial value of the adaptive damping ``Gamma``.
        :param dissipation: ``"sigmoid"`` (learned) or ``"fixed"`` damping.
        """
        super().__init__()
        self.dissipativity = AdaptiveDissipativity(
            hidden_channels, gamma=gamma, dissipation=dissipation
        )

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # Anti-symmetric mixing W - W^T, damped by the adaptive dissipativity Gamma.
        return W - W.T - self.dissipativity()
