"""Finite numerical checks for conditional-value algebra and temperature."""
import math
import torch
from torch import Tensor


def decompose(prob: Tensor, values: Tensor, policy: Tensor, action: int) -> tuple[Tensor, Tensor, Tensor]:
    """Exact decomposition; probabilities have shape [actions, events]."""
    pbar = (policy[:, None] * prob).sum(0)
    weighted = (policy[:, None] * prob * values).sum(0)
    baseline = torch.zeros_like(pbar)
    positive = pbar > 0
    baseline[positive] = weighted[positive] / pbar[positive]
    within = (prob[action] * (values[action]-baseline)).sum()
    shift = ((prob[action]-pbar)*baseline).sum()
    q = (prob*values).sum(-1)
    return within, shift, q[action] - (policy*q).sum()


def alpha_gradient(entropy: float, entropy_ratio: float = 0.5, actions: int = 5) -> float:
    beta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    loss = beta.exp()*(torch.tensor(entropy, dtype=torch.float64)-entropy_ratio*math.log(actions))
    loss.backward()
    return beta.grad.item()


def run_checks() -> dict:
    generator = torch.Generator().manual_seed(2401)
    worst = 0.0
    for _ in range(100):
        p = torch.rand(5, 8, generator=generator, dtype=torch.float64)
        p[:, 7] = 0
        p /= p.sum(-1, keepdim=True)
        q = torch.randn(5, 8, generator=generator, dtype=torch.float64)
        pi = torch.rand(5, generator=generator, dtype=torch.float64); pi /= pi.sum()
        first, second, advantage = decompose(p, q, pi, 2)
        worst = max(worst, abs((first+second-advantage).item()))
    p = torch.full((5, 8), 1/8, dtype=torch.float64)
    pi = torch.full((5,), 1/5, dtype=torch.float64)
    _, shift, _ = decompose(p, torch.randn(5, 8, generator=generator, dtype=torch.float64), pi, 0)
    threshold = 0.5*math.log(5)
    gradients = {'above':alpha_gradient(threshold+0.2), 'below':alpha_gradient(threshold-0.2), 'equal':alpha_gradient(threshold)}
    return {'max_advantage_identity_error':worst, 'action_independent_shift':shift.item(),
            'alpha_gradients':gradients,
            'passed':worst < 1e-12 and abs(shift.item()) < 1e-12 and gradients['above'] > 0 and gradients['below'] < 0 and abs(gradients['equal']) < 1e-12}
