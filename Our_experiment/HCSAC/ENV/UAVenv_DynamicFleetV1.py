"""V1 scheduling and physical charging built on the preserved V0 environment."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import DynamicFleetConfig, DynamicFleetEnv, Status
from Our_experiment.HCSAC.ENV.fleet_scheduler import FleetSlot, ReturnRequest, SchedulerMode, make_scheduler

@dataclass
class DynamicFleetV1Config(DynamicFleetConfig):
    scheduler_mode: SchedulerMode = SchedulerMode.TARGET_AWARE
    target_search_uavs: int = 4
    charging_power_w: float = 600.0
    charging_efficiency: float = 0.90
    minimum_launch_energy_ratio: float = 0.95

class DynamicFleetV1Env(DynamicFleetEnv):
    def __init__(self, config: DynamicFleetV1Config | None = None):
        config = config or DynamicFleetV1Config()
        config.scheduler_mode = SchedulerMode(config.scheduler_mode)
        if not 0 <= config.target_search_uavs <= config.max_uavs:
            raise ValueError('target_search_uavs must fit fixed slots')
        if config.charging_power_w <= 0 or not 0 < config.charging_efficiency <= 1:
            raise ValueError('charging power and efficiency must be positive')
        if not 0 <= config.minimum_launch_energy_ratio <= 1:
            raise ValueError('invalid minimum launch energy')
        self.scheduler = make_scheduler(config.scheduler_mode)
        super().__init__(config)
        self.requests: dict[int, ReturnRequest] = {}
        self.request_history: list[dict[str, Any]] = []
        self.return_cost_cache: dict[tuple[int,int],float] = {}

    def reset(self, seed: int | None = None, **kwargs: Any) -> np.ndarray:
        state=super().reset(seed=seed, **kwargs)
        self.requests={}
        self.request_history=[]
        self.return_cost_cache={}
        self.energy_stats={'search_flight_energy_j':0.0,'return_flight_energy_j':0.0,
                           'offload_energy_j':0.0,'local_compute_energy_j':0.0,'communication_energy_j':0.0,'total_grid_energy_input_j':0.0}
        self.offload_target_counts=[0]*5
        self.effective_offload_counts=[0]*5
        self.waiting_charge_steps=[]
        for u in self.uavs:
            u.update(pending_return=False, return_request_step=None, replacement_assigned=False,
                     replacement_uav_id=None, expected_return_reason=None,
                     activation_reason=None, standby_since_step=0, available_for_launch=False,
                     launch_selection_rank=None, waiting_since_step=None)
        self._refresh_availability()
        return state

    def _refresh_availability(self) -> None:
        for u in self.uavs:
            u['available_for_launch']=(u['status']==Status.STANDBY and
                u['energy']/self.initial_energy>=self.config.minimum_launch_energy_ratio and
                self.config.scheduler_mode!=SchedulerMode.NO_REPLACEMENT)

    def _get_obs(self) -> np.ndarray:
        """Vectorize V0's 15x15 window while preserving its padding and occupancy."""
        radius=7
        uncertainty=np.pad(self.uncertainty_matrix,radius,constant_values=-1)
        wind_u=np.pad(self.wind_u,radius,constant_values=0)
        wind_v=np.pad(self.wind_v,radius,constant_values=0)
        observations=np.zeros((self.N,3,15,15),dtype=float)
        for i,u in enumerate(self.uavs):
            if u.get('status',Status.ACTIVE)!=Status.ACTIVE:
                continue
            x,y=u['position']
            observations[i,0]=uncertainty[x:x+15,y:y+15]
            observations[i,1]=wind_u[x:x+15,y:y+15]
            observations[i,2]=wind_v[x:x+15,y:y+15]
            for other in self.uavs:
                if other is u or other.get('status',Status.ACTIVE)!=Status.ACTIVE:
                    continue
                dx=other['position'][0]-x; dy=other['position'][1]-y
                if -radius<=dx<=radius and -radius<=dy<=radius:
                    observations[i,0,dx+radius,dy+radius]=-1
        return observations

    def _return_cost(self, pos: tuple[int,int]) -> float:
        if pos not in self.return_cost_cache:
            self.return_cost_cache[pos]=super()._return_cost(pos)
        return self.return_cost_cache[pos]

    def _arrive(self, i: int) -> None:
        self.uavs[i]['waiting_since_step']=self.step_number
        super()._arrive(i)

    def _assign_chargers(self) -> None:
        before={i for i,u in enumerate(self.uavs) if u['status']==Status.CHARGING}
        super()._assign_chargers()
        for i,u in enumerate(self.uavs):
            if u['status']==Status.CHARGING and i not in before and u['waiting_since_step'] is not None:
                self.waiting_charge_steps.append(self.step_number-u['waiting_since_step'])
                u['waiting_since_step']=None

    def _request(self, i: int, reason: str) -> None:
        u=self.uavs[i]
        if i in self.requests: return
        r=ReturnRequest(i,self.step_number,None,reason)
        self.requests[i]=r
        u.update(pending_return=True,return_request_step=self.step_number,
                 expected_return_reason=reason,replacement_assigned=False,replacement_uav_id=None)
        self._log_event('return_request_created',i,Status.ACTIVE,Status.ACTIVE,reason=reason)

    def _release_completed_bindings(self) -> None:
        for i,r in list(self.requests.items()):
            u=self.uavs[i]
            if u['status'] != Status.ACTIVE and u['return_start_step'] is not None and u['return_start_step'] < self.step_number:
                if r.replacement_uav_id is not None:
                    replacement=self.uavs[r.replacement_uav_id]
                    if replacement['replacement_for']==i: replacement['replacement_for']=None
                    self._log_event('replacement_binding_released',i,u['status'],u['status'],replacement_uav_id=r.replacement_uav_id)
                self.request_history.append({'uav_id':i,'request_step':r.request_step,
                    'replacement_uav_id':r.replacement_uav_id,'return_step':u['return_start_step'], 'reason':r.reason})
                del self.requests[i]
                u['pending_return']=False

    def step(self, actions: list[int]) -> tuple[np.ndarray,list[float],bool]:
        if len(actions)!=self.N or any(not 0<=int(a)<8 for a in actions):
            raise ValueError('Expected one legal action per UAV slot')
        c=self.config
        self.step_number+=1
        self._release_completed_bindings()
        self._refresh_availability()
        initially_active=set(np.flatnonzero(self.get_search_active_mask()).tolist())
        charging_before=[i for i,u in enumerate(self.uavs) if u['status']==Status.CHARGING]
        slots=[]
        for i,u in enumerate(self.uavs):
            safety_return=False
            if i in initially_active:
                pos=tuple(u['position']); direction=self.directions[int(actions[i])]
                candidate=(pos[0]+direction[0],pos[1]+direction[1])
                if not (0<=candidate[0]<self.Lx and 0<=candidate[1]<self.Ly): candidate=pos
                local_bound=self.kn*self.μ*float(np.max(self.task_matrix))*self.f_ln**2
                safety=max(self._return_cost(pos),self._return_cost(candidate))*(1+c.return_energy_margin_ratio)+self._cost(pos,direction)+local_bound
                safety_return=u['energy']<=safety
                if u['energy']/self.initial_energy<=c.launch_trigger_ratio or safety_return:
                    self._request(i,'safe_energy_boundary' if safety_return else 'low_energy')
            slots.append(FleetSlot(i,u['status'].value,u['energy']/self.initial_energy,
                u['standby_since_step'],u['full_charge_step'],safety_return,
                u['energy']/self.initial_energy<=c.return_trigger_ratio,u['replacement_launch_step']))
        active_requests=tuple(self.requests[i] for i in sorted(initially_active) if i in self.requests)
        decision=self.scheduler.decide(tuple(slots),active_requests,step=self.step_number,
            target=c.target_search_uavs,minimum_energy_ratio=c.minimum_launch_energy_ratio,handover_min_steps=c.handover_min_steps)
        if c.scheduler_mode==SchedulerMode.TARGET_AWARE and decision.launch_deficit:
            self._log_event('target_deficit_detected',projected_search_count=decision.projected_search_count,launch_deficit=decision.launch_deficit)
        for i in decision.unavailable_request_ids:
            self._log_event('replacement_unavailable',i,Status.ACTIVE,Status.ACTIVE)
        inverse={replacement:request for request,replacement in decision.request_to_replacement_mapping.items()}
        for j in decision.launch_uav_ids:
            u=self.uavs[j]; owner=inverse.get(j); reason=decision.decision_reason[j]
            u.update(replacement_for=owner,activation_reason=reason,launch_selection_rank=decision.launch_selection_rank[j],
                     available_for_launch=False,activation_count=u['activation_count']+1,launch_step=self.step_number)
            if owner is not None:
                old=self.requests[owner]
                self.requests[owner]=ReturnRequest(owner,old.request_step,j,old.reason)
                source=self.uavs[owner]
                source.update(replacement_requested=True,replacement_id=j,replacement_launch_step=self.step_number,
                              replacement_assigned=True,replacement_uav_id=j)
                self._log_event('replacement_requested',owner,Status.ACTIVE,Status.ACTIVE,replacement_id=j)
                self._log_event('replacement_assigned',owner,Status.ACTIVE,Status.ACTIVE,replacement_uav_id=j)
            self._transition(j,Status.ACTIVE,'standby_to_active')
            self._log_event('reserve_launched',j,Status.STANDBY,Status.ACTIVE,replacement_for=owner,
                            activation_reason=reason,launch_selection_rank=decision.launch_selection_rank[j], activation_count=u['activation_count'])
            if c.scheduler_mode==SchedulerMode.TARGET_AWARE:
                self._log_event('target_replenishment_launched',j,Status.STANDBY,Status.ACTIVE,reason=reason)
        for i in decision.return_uav_ids:
            u=self.uavs[i]
            # A replacement that itself returns can no longer hold a binding.
            owner=u['replacement_for']
            if owner is not None and owner in self.requests and self.requests[owner].replacement_uav_id==i:
                req=self.requests[owner]
                self.requests[owner]=ReturnRequest(owner,req.request_step,None,req.reason)
                self.uavs[owner].update(replacement_requested=False,replacement_assigned=False,
                                        replacement_id=None,replacement_uav_id=None,replacement_launch_step=None)
                u['replacement_for']=None
                self._log_event('replacement_binding_released',owner,Status.ACTIVE,Status.ACTIVE,
                                replacement_uav_id=i,reason='replacement_returning')
            u['return_start_step']=self.step_number
            self._transition(i,Status.RETURNING,'active_to_returning')
        self.eligible={i for i in initially_active if self.uavs[i]['status']==Status.ACTIVE}
        rewards=[0.0]*self.N
        for i,u in enumerate(self.uavs):
            status=u['status']
            if status==Status.RETURNING:
                pos=tuple(u['position']); home=c.charging_vehicle_position
                if pos==home:
                    self._arrive(i); continue
                d=tuple(int(np.sign(home[k]-pos[k])) for k in (0,1)); cost=self._cost(pos,d)
                if u['energy']<cost:
                    if not u['exhausted']:
                        u['exhausted']=True; self._log_event('energy_exhausted_before_return',i,status,status)
                    continue
                u['energy']-=cost; self.energy_stats['return_flight_energy_j']+=cost
                u['position']=(pos[0]+d[0],pos[1]+d[1])
                self._log_event('return_move',i,status,status,from_position=list(pos))
                if u['position']==home: self._arrive(i)
            elif i in self.eligible:
                d=self.directions[int(actions[i])]; pos=tuple(u['position'])
                nxt=(pos[0]+d[0],pos[1]+d[1]); penalty=0.0
                before=float(self.uncertainty_matrix.sum())
                if not (0<=nxt[0]<self.Lx and 0<=nxt[1]<self.Ly): penalty+=0.05
                elif any(tuple(v['position'])==nxt for v in self.uavs if v is not u and v['status']==Status.ACTIVE): penalty+=0.05
                else:
                    if self.uncertainty_matrix[nxt]==1: penalty-=0.01
                    u['position']=nxt
                cost=self._cost(pos,d); used=min(u['energy'],cost)
                u['energy']-=used; self.energy_stats['search_flight_energy_j']+=used
                self._update_uncertainty_matrix(u['position'])
                rewards[i]=(before-float(self.uncertainty_matrix.sum()))/100-penalty-cost/5e5
        for i in charging_before:
            u=self.uavs[i]
            increment=min(self.initial_energy-u['energy'],c.charging_power_w*c.charging_efficiency*self.T)
            u['energy']+=increment; self.energy_stats['total_grid_energy_input_j']+=increment/c.charging_efficiency
            if u['energy']>=self.initial_energy:
                u['full_charge_step']=self.step_number
                self._log_event('charging_completed',i,Status.CHARGING,Status.CHARGING)
                u.update(replacement_requested=False,replacement_id=None,replacement_launch_step=None,
                         replacement_for=None,replacement_assigned=False,replacement_uav_id=None,
                         pending_return=False,standby_since_step=self.step_number)
                self._transition(i,Status.STANDBY,'charging_to_standby')
        self._assign_chargers(); self._refresh_availability()
        self.remaining_time-=self.T
        if c.log_every_step: self._log_event('fleet_snapshot')
        done=self._check_done()
        if done: self._log_event('episode_finished')
        return self._get_obs(),rewards,done

    def _effective_offload(self, i: int, action: int, actions: list[int]) -> int:
        """Mirror the original service feasibility test, including local fallback."""
        if action == 0:
            return 0
        u=self.uavs[i]
        mu=self.μ
        complexity=self.task_matrix[u['position'][0],u['position'][1]]
        gbs=sum(int(actions[j])==1 for j in self.eligible)
        cloud=sum(int(actions[j])==4 for j in self.eligible)
        count=sum(int(actions[j])==action for j in self.eligible)
        if action in (1,4):
            d2=self._squared_link_distance_m(u['position'],self.gbs_position,self.H)
            rate=self.L*self.B*np.log2(1+10**(self.Pn/10-3)*(10**(self.h0/10)/d2)/self.sigma2)
            delay=(gbs+cloud)*mu/rate*7/6
            delay += (gbs*mu*complexity/self.f_B if action==1 else cloud*mu*complexity/self.f_c+2*self.d_c/3e8)
            return 0 if delay>self.T or u['link']==0 else action
        d2=self._squared_link_distance_m(u['position'],self.haps_position,self.haps_height) if action==2 else self.leo_height**2
        rate=self.L*self.B*np.log2(1+10**(self.Pn/10-3)*(10**(self.h0/10)/d2)/self.sigma2)
        compute=self.f_h if action==2 else self.f_l
        delay=count*mu/rate*7/6+count*mu*complexity/compute
        return 0 if delay>self.T else action

    def step_offload(self, actions: list[int]) -> tuple[Any,list[float],bool]:
        if len(actions)!=self.N or any(not 0<=int(a)<5 for a in actions):
            raise ValueError('Expected one offload action per UAV slot')
        before=[u['energy'] for u in self.uavs]
        effective={i:self._effective_offload(i,int(actions[i]),actions) for i in self.eligible}
        for i in self.eligible:
            self.offload_target_counts[int(actions[i])]+=1
            self.effective_offload_counts[effective[i]]+=1
        state,reward,done=super().step_offload(actions)
        for i in self.eligible:
            consumed=before[i]-self.uavs[i]['energy']
            self.energy_stats['offload_energy_j']+=consumed
            category='local_compute_energy_j' if effective[i]==0 else 'communication_energy_j'
            self.energy_stats[category]+=consumed
        return state,reward,done
