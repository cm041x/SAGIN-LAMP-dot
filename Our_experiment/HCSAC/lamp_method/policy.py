"""Shared-input GCN/LAMP graph encoders and per-UAV action actor."""
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from .graph_observation import CONTEXT_SCHEMA, GraphSnapshot
from .upstream_adapter import LAMPAdapter
from .lifecycle_lamp import LifecycleLAMP


class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.conv1=GCNConv(input_dim,hidden_dim); self.conv2=GCNConv(hidden_dim,hidden_dim)
    def forward_snapshot(self,x,edge_index,edge_weight,previous_hidden,node_ids,context=None):
        h=torch.tanh(self.conv1(x,edge_index,edge_weight))
        h=torch.tanh(self.conv2(h,edge_index,edge_weight))
        return h, None


def make_encoder(variant: str, hidden_dim: int = 128) -> nn.Module:
    dim=6+len(CONTEXT_SCHEMA)
    if variant == 'gcn_context': return GCNEncoder(dim,hidden_dim)
    if variant == 'lamp_plain': return LAMPAdapter(dim,hidden_dim)
    if variant in ('lamp_lifecycle','lamp_lifecycle_ecc'): return LifecycleLAMP(dim,len(CONTEXT_SCHEMA),hidden_dim=hidden_dim)
    raise ValueError(f'Unknown variant: {variant}')


def encode(encoder: nn.Module, snapshot: GraphSnapshot, previous_hidden):
    x=torch.cat((snapshot.x,snapshot.context),dim=-1)
    return encoder.forward_snapshot(x,snapshot.edge_index,snapshot.edge_weight,previous_hidden,snapshot.node_ids,snapshot.context)


class Actor(nn.Module):
    def __init__(self, variant: str, hidden_dim: int = 128):
        super().__init__(); self.encoder=make_encoder(variant,hidden_dim)
        self.head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,5))
    def probabilities(self, embeddings):
        return torch.softmax(self.head(embeddings[:6]),dim=-1)


class SnapshotMemory:
    """Single episode memory; repeated reads of a snapshot never advance it."""
    def __init__(self): self.reset()
    def reset(self): self.hidden=None; self.snapshot_id=None; self.embeddings=None
    def advance(self, encoder: nn.Module, snapshot: GraphSnapshot):
        if self.snapshot_id == snapshot.snapshot_id: return self.embeddings
        if self.snapshot_id is not None and snapshot.snapshot_id != self.snapshot_id+1:
            raise ValueError('Snapshots must advance exactly once and in order')
        self.embeddings,self.hidden=encode(encoder,snapshot,self.hidden)
        self.snapshot_id=snapshot.snapshot_id
        return self.embeddings
