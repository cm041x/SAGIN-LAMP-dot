"""Joint-action team critics; event marginalization precedes twin minimum."""
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def marginal_value(logits: Tensor, conditional_values: Tensor) -> Tensor:
    return (torch.softmax(logits, dim=-1) * conditional_values).sum(dim=-1)


class JointCriticHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, event_conditioned: bool = False) -> None:
        super().__init__()
        self.event_conditioned = event_conditioned
        self.action_embedding = nn.Embedding(6, 16)  # index 5 means no action
        self.node_fusion = nn.Sequential(nn.Linear(hidden_dim+16, hidden_dim), nn.Tanh())
        self.readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, 16 if event_conditioned else 1))

    def forward(self, h: Tensor, actions: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if h.shape[-2] != 10 or actions.shape[-1] != 6 or action_mask.shape != actions.shape:
            raise ValueError('Expected 10 nodes and 6 aligned action slots')
        if ((actions[action_mask] < 0) | (actions[action_mask] >= 5)).any():
            raise ValueError('Invalid eligible action')
        slot_actions = torch.where(action_mask, actions, torch.full_like(actions, 5))
        infra = torch.full((*actions.shape[:-1], 4), 5, device=actions.device, dtype=torch.long)
        embedded = self.action_embedding(torch.cat((slot_actions, infra), dim=-1))
        fused = self.node_fusion(torch.cat((h, embedded), dim=-1)).mean(dim=-2)
        out = self.readout(fused)
        if not self.event_conditioned:
            return out.squeeze(-1), None, None
        logits, values = out[..., :8], out[..., 8:]
        return marginal_value(logits, values), logits, values


def event_critic_loss(logits: Tensor, values: Tensor, observed: Tensor, target: Tensor,
                      lambda_cond: float = 1.0, lambda_event: float = 0.1) -> Tensor:
    q = marginal_value(logits, values)
    selected = values.gather(-1, observed.unsqueeze(-1)).squeeze(-1)
    target = target.detach()
    return ((q-target).square() + lambda_cond*(selected-target).square()).mean() + lambda_event*F.cross_entropy(logits, observed)


def event_category(events: list[dict]) -> int:
    names = {event['event_type'] for event in events}
    return int('active_to_returning' in names) + 2*int('standby_to_active' in names) + 4*int('charging_to_standby' in names)
