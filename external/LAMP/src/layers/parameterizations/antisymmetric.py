import torch


class AntiSymmetric(torch.nn.Module):
    """
    Anti-Symmetric Parametrization: enforces the weight matrix to be anti-symmetric
    """

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu(diagonal=1) - W.triu(diagonal=1).T

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu(diagonal=1)
