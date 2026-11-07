"""Deterministic, untrained grid responsibility and waypoint rules."""
from dataclasses import dataclass
from enum import Enum
from typing import Mapping
import numpy as np

class HandoverMode(str, Enum):
    NONE='none'
    UNCERTAINTY_WAYPOINT='uncertainty_waypoint'

@dataclass(frozen=True)
class HandoverPolicyConfig:
    separation_weight: float=0.5
    travel_weight: float=0.25
    distance_weight: float=1.0
    energy_weight: float=0.2
    occupancy_weight: float=2.0
    max_handover_steps: int=30

def distance(a: tuple[int,int], b: tuple[int,int]) -> int:
    return max(abs(a[0]-b[0]),abs(a[1]-b[1]))

def responsibility_partition(positions: Mapping[int,tuple[int,int]], size: int=20) -> np.ndarray:
    if not positions: raise ValueError('At least one active UAV is required')
    result=np.empty((size,size),dtype=int)
    for x in range(size):
        for y in range(size):
            result[x,y]=min(positions,key=lambda i:(distance((x,y),positions[i]),i))
    return result

def choose_target(owner: int, replacement_start: tuple[int,int], positions: Mapping[int,tuple[int,int]],
                  uncertainty: np.ndarray, vehicle: tuple[int,int], cfg: HandoverPolicyConfig) -> tuple[tuple[int,int] | None,bool,int]:
    region=responsibility_partition(positions,uncertainty.shape[0]); size=uncertainty.shape[0]
    occupied=set(positions.values()); others=[p for i,p in positions.items() if i!=owner]
    norm=max(1,size-1)
    def candidates(region_only: bool) -> list[tuple]:
        result=[]
        for x in range(size):
            for y in range(size):
                c=(x,y); u=float(uncertainty[c])
                if u<=0 or c==vehicle or c in occupied or (region_only and int(region[c])!=owner): continue
                separation=min((distance(c,p) for p in others),default=norm)/norm
                travel=distance(c,replacement_start)/norm
                score=u+cfg.separation_weight*separation-cfg.travel_weight*travel
                result.append((-score,-u,-separation,travel,x,y))
        return result
    local=candidates(True); fallback=not local
    choices=local or candidates(False)
    best=min(choices) if choices else None
    return ((best[-2],best[-1]) if best else None, fallback, int(np.sum(region==owner)))

def waypoint_action(env, slot: int, target: tuple[int,int], cfg: HandoverPolicyConfig) -> int:
    pos=tuple(env.uavs[slot]['position']); occupied={tuple(u['position']) for i,u in enumerate(env.uavs) if i!=slot and u['status'].value=='active'}
    candidates=[]; norm=max(1,env.Lx-1)
    costs=[env._cost(pos,d) for d in env.directions]; max_energy=max(costs) or 1.0
    for i,d in enumerate(env.directions):
        p=(pos[0]+d[0],pos[1]+d[1])
        if not (0<=p[0]<env.Lx and 0<=p[1]<env.Ly): continue
        cost=cfg.distance_weight*distance(p,target)/norm+cfg.energy_weight*costs[i]/max_energy+cfg.occupancy_weight*int(p in occupied)
        candidates.append((cost,i))
    if not candidates: raise ValueError('No legal waypoint action')
    return min(candidates)[1]
