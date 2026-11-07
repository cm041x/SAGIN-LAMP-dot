import torch


class AdaptiveDissipativity(torch.nn.Module):
    """Adaptive damping term ``Gamma = diag(sigmoid(d))``.

    Produces the diagonal damping matrix used by the LAMP dynamics. The diagonal
    ``d`` is learned and squashed by a sigmoid, initialized so that
    ``sigmoid(d) == gamma``.
    """

    def __init__(self, in_channels: int, gamma: float, dissipation: str = "sigmoid") -> None:
        """
        :param in_channels: Hidden-state size (diagonal length).
        :param gamma: Target / initial damping value.
        :param dissipation: ``"sigmoid"`` (learned) or ``"fixed"`` (``Gamma = gamma``).
        """
        super().__init__()
        self.dissipation = dissipation

        if dissipation == "fixed":
            self.diagonal = torch.nn.Parameter(
                torch.tensor([gamma] * in_channels), requires_grad=False
            )
        else:
            self.diagonal = torch.nn.Parameter(torch.empty(in_channels), requires_grad=True)
            self.diagonal.data = torch.log(
                torch.full((in_channels,), gamma / (1 - gamma), dtype=torch.float32)
            )

    def forward(self) -> torch.Tensor:
        # Gamma = diag(activation(d)).
        if self.dissipation == "fixed":
            return torch.diag(self.diagonal)
        return torch.diag(torch.sigmoid(self.diagonal))
