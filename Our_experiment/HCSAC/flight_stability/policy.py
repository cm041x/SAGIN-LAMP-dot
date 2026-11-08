"""Scale-controlled graph branch and bounded residual; CNN remains frozen."""
from __future__ import annotations
import torch
from torch import nn
from Our_experiment.HCSAC.flight_residual.policy import (
    ACTION_DIM, INACTIVE_ACTION, UAV_COUNT, NODE_COUNT, LOCAL_DIM,
    FlightMemory, flight_observe, encode, FlightSnapshot,
)
from Our_experiment.HCSAC.lamp_method.graph_observation import CONTEXT_SCHEMA
from Our_experiment.HCSAC.lamp_method.policy import GCNEncoder
from Our_experiment.HCSAC.lamp_method.upstream_adapter import LAMPAdapter

VARIANTS=('stabilized_gcn_residual','stabilized_plain_lamp_residual')

class StableEncoder(nn.Module):
    """Normalize local input and readout independently; recurrent state remains raw."""
    def __init__(self, variant: str, hidden_dim: int=128, xi: float=2.):
        super().__init__()
        dim=6+len(CONTEXT_SCHEMA)+LOCAL_DIM
        self.local_norm=nn.LayerNorm(LOCAL_DIM)
        self.readout_norm=nn.LayerNorm(hidden_dim)
        if variant==VARIANTS[0]:self.core=GCNEncoder(dim,hidden_dim)
        elif variant==VARIANTS[1]:self.core=LAMPAdapter(dim,hidden_dim,xi=xi)
        else:raise ValueError(f'Unknown variant {variant}')

    def normalized_local(self, local):
        # Preserve zero placeholders for inactive UAVs and infrastructure.
        present=local.abs().sum(-1,keepdim=True)>0
        return self.local_norm(local)*present

    def forward_snapshot(self,x,edge_index,edge_weight,previous_hidden,node_ids,context=None):
        split=6+len(CONTEXT_SCHEMA)
        x=torch.cat((x[:,:split],self.normalized_local(x[:,split:])),dim=-1)
        raw,hidden=self.core.forward_snapshot(x,edge_index,edge_weight,previous_hidden,node_ids,context)
        return self.readout_norm(raw),hidden

class StableActor(nn.Module):
    def __init__(self, variant: str, hidden_dim: int=128, residual_bound: float=1., bounded: bool=True, xi: float=2.):
        super().__init__()
        if residual_bound<=0:raise ValueError('residual_bound must be positive')
        self.encoder=StableEncoder(variant,hidden_dim,xi)
        self.residual_head=nn.Sequential(nn.Linear(LOCAL_DIM+hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,ACTION_DIM))
        nn.init.zeros_(self.residual_head[-1].weight);nn.init.zeros_(self.residual_head[-1].bias)
        self.residual_bound=float(residual_bound);self.bounded=bounded

    def residual(self, embedding, snapshot):
        local=self.encoder.normalized_local(snapshot.local_features)
        raw=self.residual_head(torch.cat((local,embedding[:UAV_COUNT]),-1))
        return self.residual_bound*torch.tanh(raw) if self.bounded else raw

    def probabilities(self,embedding,snapshot):
        return torch.softmax(snapshot.base_probs.clamp_min(1e-8).log()+self.residual(embedding,snapshot),-1)
