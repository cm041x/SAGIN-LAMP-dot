import torch


class Symmetric(torch.nn.Module):
    """
    Symmetric Parametrization: enforces the weight matrix to be symmetric
    """

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu() + W.triu().T

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu()
