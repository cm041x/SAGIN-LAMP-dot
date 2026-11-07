"""V1.5 responsibility handover and diagnostic instrumentation."""
from dataclasses import dataclass
from itertools import combinations
from typing import Any
import numpy as np
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import Status
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleetV1 import DynamicFleetV1Config, DynamicFleetV1Env
from Our_experiment.HCSAC.ENV.handover_policy import HandoverMode, HandoverPolicyConfig, choose_target, waypoint_action, distance

@dataclass
class DynamicFleetV15Config(DynamicFleetV1Config):
    handover_mode: HandoverMode = HandoverMode.NONE
    separation_weight: float = 0.5
    travel_weight: float = 0.25
    distance_weight: float = 1.0
    energy_weight: float = 0.2
    occupancy_weight: float = 2.0
    max_handover_steps: int = 30

class DynamicFleetV15Env(DynamicFleetV1Env):
    def __init__(self, config: DynamicFleetV15Config | None = None):
        config=config or DynamicFleetV15Config(); config.handover_mode=HandoverMode(config.handover_mode)
        if config.max_handover_steps<1: raise ValueError('max_handover_steps must be positive')
        self.policy=HandoverPolicyConfig(config.separation_weight,config.travel_weight,config.distance_weight,
                                        config.energy_weight,config.occupancy_weight,config.max_handover_steps)
        self._handover_ready=False
        super().__init__(config)

    def reset(self, seed: int | None = None, **kwargs: Any) -> np.ndarray:
        self._handover_ready=False
        obs=super().reset(seed=seed,**kwargs)
        self.visit_maps=np.zeros((self.N,self.Lx,self.Ly),dtype=np.int32)
        self.handover_records=[]; self.offload_timeline=[]; self.handover_timeline=[]; self.step_trajectory=[]
        self.total_visit_count=0; self.conflict_attempts=0; self.proximity_steps=0
        self.post_handover_unique_coverage_gain=0
        for u in self.uavs:
            u.update(responsibility_owner_id=None,handover_target_cell=None,handover_created_step=None,
                     handover_start_step=None,handover_end_step=None,handover_end_reason=None,handover_active=False,
                     handover_for_uav_id=None,handover_region_size=0,handover_initial_target_uncertainty=None,
                     handover_final_target_uncertainty=None,handover_distance_steps=0,handover_energy_j=0.0,
                     handover_unique_coverage_gain=0)
        self._handover_ready=True
        return obs

    def _log_event(self, event_type: str, index: int | None = None, old: Status | None = None, new: Status | None = None, **metadata: Any) -> None:
        super()._log_event(event_type,index,old,new,**metadata)
        if self._handover_ready and self.config.handover_mode==HandoverMode.UNCERTAINTY_WAYPOINT and event_type=='replacement_assigned':
            self._create_waypoint(index,metadata['replacement_uav_id'])

    def _create_waypoint(self, owner: int, replacement: int) -> None:
        u=self.uavs[replacement]
        if u['handover_active']: raise RuntimeError('Cannot replace an active waypoint')
        active={i:tuple(v['position']) for i,v in enumerate(self.uavs) if v['status']==Status.ACTIVE}
        target,fallback,region_size=choose_target(owner,tuple(u['position']),active,self.uncertainty_matrix,
                                                   tuple(self.config.charging_vehicle_position),self.policy)
        super()._log_event('responsibility_region_created',owner,Status.ACTIVE,Status.ACTIVE,
                           replacement_uav_id=replacement,region_size=region_size)
        if fallback: super()._log_event('fallback_global_target',replacement,None,None,owner=owner)
        if target is None:
            super()._log_event('no_valid_handover_target',replacement,None,None,owner=owner)
            return
        u.update(responsibility_owner_id=owner,handover_target_cell=target,handover_created_step=self.step_number,
                 handover_start_step=self.step_number,handover_end_step=None,handover_end_reason=None,
                 handover_active=True,handover_for_uav_id=owner,handover_region_size=region_size,
                 handover_initial_target_uncertainty=float(self.uncertainty_matrix[target]),
                 handover_final_target_uncertainty=None,handover_distance_steps=0,handover_energy_j=0.0,
                 handover_unique_coverage_gain=0)
        super()._log_event('handover_target_assigned',replacement,None,None,owner=owner,target=list(target),region_size=region_size)
        super()._log_event('handover_started',replacement,None,None,owner=owner,target=list(target))

    def _finish_waypoint(self, i: int, reason: str) -> None:
        u=self.uavs[i]
        if not u['handover_active']: return
        target=tuple(u['handover_target_cell'])
        u.update(handover_active=False,handover_end_step=self.step_number,handover_end_reason=reason,
                 handover_final_target_uncertainty=float(self.uncertainty_matrix[target]))
        record={'uav_id':i,'owner':u['handover_for_uav_id'],'target':list(target),'start_step':u['handover_start_step'],
                'end_step':self.step_number,'reason':reason,'travel_steps':u['handover_distance_steps'],
                'energy_j':u['handover_energy_j'],'initial_uncertainty':u['handover_initial_target_uncertainty'],
                'final_uncertainty':u['handover_final_target_uncertainty'],'unique_gain':u['handover_unique_coverage_gain']}
        self.handover_records.append(record)
        super()._log_event('handover_completed',i,u['status'],u['status'],**record)
        super()._log_event('handover_released',i,u['status'],u['status'],reason=reason,target=list(target))

    def step(self, actions: list[int]) -> tuple[np.ndarray,list[float],bool]:
        chosen=list(actions); before=[tuple(u['position']) for u in self.uavs]
        uncertainty_before=self.uncertainty_matrix.copy()
        energy_before=[float(u['energy']) for u in self.uavs]
        newly_by_uav={}
        active_waypoints=set()
        if self.config.handover_mode==HandoverMode.UNCERTAINTY_WAYPOINT:
            for i,u in enumerate(self.uavs):
                if not u['handover_active']: continue
                if u['status']!=Status.ACTIVE:
                    self._finish_waypoint(i,'replacement_returning'); continue
                target=tuple(u['handover_target_cell'])
                if tuple(u['position'])==target:
                    self._finish_waypoint(i,'target_reached'); continue
                if self.uncertainty_matrix[target]<=0:
                    self._finish_waypoint(i,'target_already_resolved'); continue
                if u['handover_distance_steps']>=self.policy.max_handover_steps:
                    self._finish_waypoint(i,'timeout'); continue
                chosen[i]=waypoint_action(self,i,target,self.policy); active_waypoints.add(i)
        obs,rewards,done=super().step(chosen)
        # Replay the original sequential occupancy rule to count attempted collisions.
        occupied={i:before[i] for i,u in enumerate(self.uavs) if u['status']==Status.ACTIVE}
        for i in sorted(self.eligible):
            d=self.directions[int(chosen[i])]; pos=before[i]; nxt=(pos[0]+d[0],pos[1]+d[1])
            if 0<=nxt[0]<self.Lx and 0<=nxt[1]<self.Ly:
                if any(j!=i and p==nxt for j,p in occupied.items()): self.conflict_attempts+=1
                else: occupied[i]=nxt
            final=tuple(self.uavs[i]['position']); self.visit_maps[i,final[0],final[1]]+=1; self.total_visit_count+=1
            newly_by_uav[i]=bool(uncertainty_before[final]>0)
            uncertainty_before[final]=0
        active=[i for i,u in enumerate(self.uavs) if u['status']==Status.ACTIVE]
        self.proximity_steps+=sum(distance(tuple(self.uavs[i]['position']),tuple(self.uavs[j]['position']))<=1 for i,j in combinations(active,2))
        for i in active_waypoints:
            u=self.uavs[i]
            if u['status']!=Status.ACTIVE:
                self._finish_waypoint(i,'replacement_returning'); continue
            if i in self.eligible:
                u['handover_distance_steps']+=1
                cost=min(energy_before[i],self._cost(before[i],self.directions[int(chosen[i])])); u['handover_energy_j']+=cost
                pos=tuple(u['position'])
                if newly_by_uav.get(i,False):
                    u['handover_unique_coverage_gain']+=1; self.post_handover_unique_coverage_gain+=1
                if pos!=before[i]:
                    super()._log_event('handover_move',i,Status.ACTIVE,Status.ACTIVE,from_position=list(before[i]),target=list(u['handover_target_cell']))
                target=tuple(u['handover_target_cell'])
                if pos==target: self._finish_waypoint(i,'target_reached')
                elif self.uncertainty_matrix[target]<=0: self._finish_waypoint(i,'target_already_resolved')
                elif u['handover_distance_steps']>=self.policy.max_handover_steps: self._finish_waypoint(i,'timeout')
        if done:
            for i,u in enumerate(self.uavs):
                if u['handover_active']: self._finish_waypoint(i,'episode_finished')
        self.handover_timeline.append({'step':self.step_number,'active_waypoints':[i for i,u in enumerate(self.uavs) if u['handover_active']]})
        self.step_trajectory.append({'step':self.step_number,'positions':[list(u['position']) for u in self.uavs],
                                     'statuses':self.get_uav_statuses(),'energies':[float(u['energy']) for u in self.uavs]})
        return obs,rewards,done

    def step_offload(self, actions: list[int]) -> tuple[Any,list[float],bool]:
        requested=[0]*5; effective=[0]*5
        for i in self.eligible:
            requested[int(actions[i])]+=1
            effective[self._effective_offload(i,int(actions[i]),actions)]+=1
        value=super().step_offload(actions)
        self.offload_timeline.append({'step':self.step_number,'requested':requested,'effective':effective,
                                      'active_decisions':len(self.eligible),
                                      'local_fallback_count':sum(requested[1:])-sum(effective[1:])})
        return value
