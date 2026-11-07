"""CPU-owned replay records for executed slots and fixed-slot graph transitions."""
from collections import deque
import random
import numpy as np
import torch
from torch_geometric.data import Data

class CNNReplay:
    def __init__(self, capacity: int):
        self.capacity=capacity; self.items=deque(maxlen=capacity)
    def __len__(self) -> int: return len(self.items)
    def add(self, state, action: int, reward: float, next_state, done: bool, slot_id: int, episode_id: int, step_id: int, provenance: dict | None = None) -> None:
        if not 0<=slot_id<6: raise ValueError('Invalid slot id')
        a=np.asarray(state,dtype=np.float32).copy(); b=np.asarray(next_state,dtype=np.float32).copy()
        if a.shape!=(3,15,15) or b.shape!=(3,15,15): raise ValueError('Invalid CNN observation')
        self.items.append((a,int(action),float(reward),b,bool(done),int(slot_id),int(episode_id),int(step_id),dict(provenance or {})))
    def sample(self, count: int) -> list: return random.sample(self.items,count)
    def state_dict(self) -> dict: return {'capacity':self.capacity,'items':list(self.items)}
    def load_state_dict(self, state: dict) -> None:
        self.capacity=state['capacity'];self.items=deque(state['items'],maxlen=self.capacity)

class GraphReplay:
    def __init__(self, capacity: int):
        self.capacity=capacity; self.items=deque(maxlen=capacity)
    def __len__(self) -> int:return len(self.items)
    def add(self, graph: Data, actions, rewards, next_graph: Data, active_mask, next_active_mask, dones, episode_done: bool, episode_id: int, step_id: int) -> None:
        arrays=[np.asarray(a,dtype=dtype).copy() for a,dtype in ((actions,np.int64),(rewards,np.float32),(active_mask,np.bool_), (next_active_mask,np.bool_),(dones,np.bool_))]
        if any(x.shape!=(6,) for x in arrays): raise ValueError('Graph transitions require six slots')
        self.items.append((graph.clone().detach().cpu(),arrays[0],arrays[1],next_graph.clone().detach().cpu(),
                           arrays[2],arrays[3],arrays[4],bool(episode_done),int(episode_id),int(step_id)))
    def sample(self, count: int) -> list:return random.sample(self.items,count)
    def state_dict(self) -> dict:return {'capacity':self.capacity,'items':list(self.items)}
    def load_state_dict(self, state: dict) -> None:
        self.capacity=state['capacity'];self.items=deque(state['items'],maxlen=self.capacity)
