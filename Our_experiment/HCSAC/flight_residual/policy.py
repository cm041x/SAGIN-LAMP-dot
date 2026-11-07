"""Flight residual policy and shared current-observation graph encoding."""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
from Our_experiment.HCSAC.lamp_method.graph_observation import observe, CONTEXT_SCHEMA
from Our_experiment.HCSAC.lamp_method.policy import GCNEncoder
from Our_experiment.HCSAC.lamp_method.upstream_adapter import LAMPAdapter
from Our_experiment.HCSAC.lamp_method.lifecycle_lamp import LifecycleLAMP

ACTION_DIM=8
INACTIVE_ACTION=ACTION_DIM
LOCAL_DIM=128
NODE_COUNT=10
UAV_COUNT=6
VARIANTS=('gcn_flight_residual','lamp_plain_flight_residual',
          'lamp_lifecycle_flight_residual','lamp_lifecycle_ecc_flight_residual')


@dataclass(frozen=True)
class FlightSnapshot:
    x: Tensor
    context: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    node_ids: Tensor
    existence_mask: Tensor
    action_mask: Tensor
    local_features: Tensor
    base_probs: Tensor
    snapshot_id: int


def local_features_and_probs(cnn: nn.Module, state: Tensor, mask: Tensor) -> tuple[Tensor,Tensor]:
    """Exactly reproduce PolicyNet_CNN, with detached hidden FC1 features."""
    if state.shape != (UAV_COUNT,3,15,15) or mask.shape != (UAV_COUNT,):
        raise ValueError('Expected six fixed CNN observation/action slots')
    z=state.new_zeros((UAV_COUNT,LOCAL_DIM)); probs=state.new_zeros((UAV_COUNT,ACTION_DIM))
    with torch.no_grad():
        if mask.any():
            x=F.relu(cnn.conv1(state[mask]));x=F.relu(cnn.conv2(x))
            local=F.relu(cnn.fc1(x.reshape(x.shape[0],-1)))
            logits=cnn.fc2(local);logits=logits-logits.max(dim=1,keepdim=True)[0]
            z[mask]=local;probs[mask]=F.softmax(logits,dim=1)
    return z,probs


def flight_observe(env, state, cnn: nn.Module) -> FlightSnapshot:
    """Snapshot immediately before flight; ignore stale post-flight eligible."""
    graph=observe(env)
    flight_mask=torch.as_tensor(env.get_search_active_mask(),dtype=torch.bool)
    local,base=local_features_and_probs(cnn,torch.as_tensor(state,dtype=torch.float32),flight_mask)
    return FlightSnapshot(graph.x,graph.context,graph.edge_index,graph.edge_weight,
                          graph.node_ids,graph.existence_mask,flight_mask,local,base,env.step_number)


def make_flight_encoder(variant: str, hidden_dim: int=128) -> nn.Module:
    dim=6+len(CONTEXT_SCHEMA)+LOCAL_DIM
    if variant==VARIANTS[0]:return GCNEncoder(dim,hidden_dim)
    if variant==VARIANTS[1]:return LAMPAdapter(dim,hidden_dim)
    if variant in VARIANTS[2:]:return LifecycleLAMP(dim,len(CONTEXT_SCHEMA),hidden_dim=hidden_dim)
    raise ValueError('Unknown flight variant')


def encode(encoder: nn.Module, snapshot: FlightSnapshot, previous_hidden: Optional[Tensor]):
    local=torch.cat((snapshot.local_features,snapshot.x.new_zeros((NODE_COUNT-UAV_COUNT,LOCAL_DIM))),dim=0)
    x=torch.cat((snapshot.x,snapshot.context,local),dim=-1)
    return encoder.forward_snapshot(x,snapshot.edge_index,snapshot.edge_weight,previous_hidden,
                                    snapshot.node_ids,snapshot.context)


class FlightActor(nn.Module):
    def __init__(self, variant: str, hidden_dim: int=128):
        super().__init__();self.encoder=make_flight_encoder(variant,hidden_dim)
        self.residual_head=nn.Sequential(nn.Linear(LOCAL_DIM+hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,ACTION_DIM))
        nn.init.zeros_(self.residual_head[-1].weight);nn.init.zeros_(self.residual_head[-1].bias)

    def probabilities(self, embedding: Tensor, snapshot: FlightSnapshot) -> Tensor:
        residual=self.residual_head(torch.cat((snapshot.local_features,embedding[:UAV_COUNT]),dim=-1))
        return torch.softmax(snapshot.base_probs.clamp_min(1e-8).log()+residual,dim=-1)


class FlightMemory:
    def __init__(self):self.reset()
    def reset(self):self.hidden=None;self.snapshot_id=None;self.embeddings=None
    def advance(self, encoder: nn.Module, snapshot: FlightSnapshot) -> Tensor:
        if self.snapshot_id==snapshot.snapshot_id:return self.embeddings
        if self.snapshot_id is not None and snapshot.snapshot_id!=self.snapshot_id+1:
            raise ValueError('Flight snapshots must be sequential and advanced once')
        self.embeddings,self.hidden=encode(encoder,snapshot,self.hidden)
        self.snapshot_id=snapshot.snapshot_id
        return self.embeddings
