import torch


class ForwardEuler(torch.nn.Module):
    """Explicit forward-Euler step: ``h <- h + epsilon * activation(conv)``."""

    def __init__(self, epsilon: float, activation: str = "tanh") -> None:
        super().__init__()
        self.epsilon = epsilon
        self.activation = getattr(torch, activation)

    def forward(self, h: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
        """Advance the state by one forward-Euler step.

        :param h: Current state.
        :param conv: Vector field evaluated at ``h``.
        :return: ``h + epsilon * activation(conv)``.
        """
        # Euler update with step size epsilon.
        return h + self.epsilon * self.activation(conv)
