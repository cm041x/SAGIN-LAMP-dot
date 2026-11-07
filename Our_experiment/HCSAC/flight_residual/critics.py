"""Eight-action team critics; inactive UAVs use a separate sentinel."""
import torch
from torch import Tensor, nn
from Our_experiment.HCSAC.lamp_method.critics import marginal_value, event_critic_loss, event_category
from .policy import ACTION_DIM, INACTIVE_ACTION, UAV_COUNT, NODE_COUNT


class FlightCriticHead(nn.Module):
    def __init__(self, hidden_dim: int=128, event_conditioned: bool=False):
        super().__init__(); self.event_conditioned=event_conditioned
        self.action_embedding=nn.Embedding(ACTION_DIM+1,16)
        self.node_fusion=nn.Sequential(nn.Linear(hidden_dim+16,hidden_dim),nn.Tanh())
        self.readout=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,16 if event_conditioned else 1))

    def forward(self,h: Tensor,actions: Tensor,action_mask: Tensor):
        if h.shape[-2]!=NODE_COUNT or actions.shape[-1]!=UAV_COUNT or action_mask.shape!=actions.shape:
            raise ValueError('Expected ten nodes and six aligned flight slots')
        if ((actions[action_mask]<0)|(actions[action_mask]>=ACTION_DIM)).any():
            raise ValueError('Invalid proposed flight action')
        chosen=torch.where(action_mask,actions,torch.full_like(actions,INACTIVE_ACTION))
        infra=torch.full((*actions.shape[:-1],NODE_COUNT-UAV_COUNT),INACTIVE_ACTION,dtype=torch.long,device=actions.device)
        embedded=self.action_embedding(torch.cat((chosen,infra),dim=-1))
        fused=self.node_fusion(torch.cat((h,embedded),dim=-1)).mean(dim=-2)
        out=self.readout(fused)
        if not self.event_conditioned:return out.squeeze(-1),None,None
        logits,values=out[...,:8],out[...,8:]
        return marginal_value(logits,values),logits,values
