"""Charging-assisted fixed-slot fleet; original physics and service model are retained."""
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
import torch
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import UAVEnv

class Status(str, Enum):
    STANDBY = 'standby'
    ACTIVE = 'active'
    RETURNING = 'returning'
    WAITING = 'waiting_charge'
    CHARGING = 'charging'

@dataclass
class DynamicFleetConfig:
    max_uavs: int = 6
    initial_active_uavs: int = 4
    reserve_uavs: int = 2
    charging_vehicle_position: tuple[int, int] = (10, 10)
    charging_capacity: int = 1
    charge_rate_per_step: float = 10000.0
    launch_trigger_ratio: float = 0.45
    return_trigger_ratio: float = 0.30
    return_energy_margin_ratio: float = 0.20
    handover_min_steps: int = 2
    max_episode_steps: int = 200
    stop_uncertainty_threshold: float | None = None
    seed: int = 42
    enable_pretrained_controller: bool = False
    log_every_step: bool = True

class DynamicFleetEnv(UAVEnv):
    def __init__(self, config: DynamicFleetConfig | None = None):
        self.config = config or DynamicFleetConfig()
        c = self.config
        if c.max_uavs != c.initial_active_uavs + c.reserve_uavs or not 1 <= c.max_uavs <= 8:
            raise ValueError('Fleet counts must add up and fit original starting positions')
        if not (0 <= c.return_trigger_ratio <= c.launch_trigger_ratio <= 1):
            raise ValueError('Invalid energy thresholds')
        if c.charging_capacity < 1 or c.charge_rate_per_step <= 0 or c.handover_min_steps < 0:
            raise ValueError('Invalid charging or handover configuration')
        if not all(0 <= v < 20 for v in c.charging_vehicle_position):
            raise ValueError('Vehicle position outside grid')
        self.events: list[dict[str, Any]] = []
        self.step_number = 0
        self.queue: deque[int] = deque()
        self.eligible: set[int] = set()
        super().__init__(N=c.max_uavs)

    def get_uav_statuses(self) -> list[str]:
        return [u.get('status', Status.ACTIVE).value for u in self.uavs]

    def get_search_active_mask(self) -> np.ndarray:
        return np.array([u.get('status', Status.ACTIVE) == Status.ACTIVE for u in self.uavs], dtype=bool)

    @property
    def active_mask(self) -> np.ndarray:
        return self.get_search_active_mask()

    def get_active_mask(self) -> np.ndarray:
        return self.get_search_active_mask()

    def get_airborne_mask(self) -> np.ndarray:
        return np.array([u.get('status', Status.ACTIVE) in (Status.ACTIVE, Status.RETURNING) for u in self.uavs], dtype=bool)

    def get_fleet_counts(self) -> dict[str, int]:
        statuses = self.get_uav_statuses()
        return {'search_active_count': statuses.count(Status.ACTIVE.value), 'airborne_count': sum(self.get_airborne_mask()).item(),
                'standby_count': statuses.count(Status.STANDBY.value), 'returning_count': statuses.count(Status.RETURNING.value),
                'waiting_charge_count': statuses.count(Status.WAITING.value), 'charging_count': statuses.count(Status.CHARGING.value)}

    def get_event_log(self) -> list[dict[str, Any]]:
        return list(self.events)

    def _log_event(self, event_type: str, index: int | None = None, old: Status | None = None, new: Status | None = None, **metadata: Any) -> None:
        u = self.uavs[index] if index is not None else None
        self.events.append({'step': self.step_number, 'simulation_time': self.max_search_time-self.remaining_time,
            'uav_id': index, 'event_type': event_type, 'old_status': old.value if old else None,
            'new_status': new.value if new else None, 'position': list(u['position']) if u else None,
            'energy': float(u['energy']) if u else None, 'energy_ratio': float(u['energy']/self.initial_energy) if u else None,
            **self.get_fleet_counts(), 'average_uncertainty': float(np.mean(self.uncertainty_matrix)), 'metadata': metadata})

    def _transition(self, i: int, new: Status, event: str) -> None:
        u = self.uavs[i]
        old = u['status']
        u['status'] = new
        u['done'] = new != Status.ACTIVE
        u['last_transition_step'] = self.step_number
        self._log_event(event, i, old, new)

    def reset(self, seed: int | None = None, **kwargs: Any) -> np.ndarray:
        c = self.config
        positions = kwargs.pop('positions', None)
        kwargs.pop('destinations', None)
        if positions is None:
            positions = list(self.gird_position[:c.initial_active_uavs]) + [tuple(c.charging_vehicle_position)] * c.reserve_uavs
        if len(positions) != self.N:
            raise ValueError('Expected a position for each UAV slot')
        positions = list(positions[:c.initial_active_uavs]) + [tuple(c.charging_vehicle_position)] * c.reserve_uavs
        super().reset(seed=c.seed if seed is None else seed, positions=positions,
                      destinations=[tuple(c.charging_vehicle_position)] * self.N, **kwargs)
        # Grounded standby aircraft do not mark their parking cell as searched.
        self.uncertainty_matrix.fill(1.0)
        for u in self.uavs[:c.initial_active_uavs]:
            self.uncertainty_matrix[tuple(u['position'])] = 0.0
        self.G = self.generate_sagin_graph()
        self.step_number = 0
        self.events = []
        self.queue = deque()
        self.eligible = set()
        for i, u in enumerate(self.uavs):
            status = Status.ACTIVE if i < c.initial_active_uavs else Status.STANDBY
            if status == Status.STANDBY:
                u['position'] = tuple(c.charging_vehicle_position)
            u.update(destination=tuple(c.charging_vehicle_position), status=status, done=status != Status.ACTIVE,
                     replacement_requested=False, replacement_for=None, replacement_id=None, replacement_launch_step=None, activation_count=int(status == Status.ACTIVE),
                     charging_count=0, last_transition_step=0, launch_step=0 if status == Status.ACTIVE else None,
                     return_start_step=None, charging_start_step=None, full_charge_step=None, exhausted=False)
        self._log_event('episode_reset')
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs()
        for i, u in enumerate(self.uavs):
            if u.get('status', Status.ACTIVE) != Status.ACTIVE:
                obs[i] = 0
        return obs

    def get_obs_2(self) -> Any:
        data = super().get_obs_2()
        # Node insertion order is UAV_0..UAV_N-1 followed by infrastructure.
        inactive = torch.tensor(~self.get_search_active_mask(), dtype=torch.bool)
        data.x[:self.N][inactive] = 0
        mask = torch.ones(data.num_nodes, dtype=torch.bool)
        mask[:self.N] = ~inactive
        keep = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        data.edge_index = data.edge_index[:, keep]
        data.edge_weight = data.edge_weight[keep]
        data.slot_id = torch.cat((torch.arange(self.N), torch.full((data.num_nodes-self.N,), -1)))
        for i in np.where(inactive.numpy())[0]:
            self.uavs[int(i)]['link'] = 0
        return data

    def _cost(self, pos: tuple[int, int], direction: tuple[int, int]) -> float:
        x, y = pos
        return self._flight_energy(self._get_real_v(np.array(direction), np.array([self.wind_u[x,y], self.wind_v[x,y]])), self.T)

    def _return_cost(self, pos: tuple[int, int]) -> float:
        home = self.config.charging_vehicle_position
        total = 0.0
        while pos != home:
            d = tuple(int(np.sign(home[k]-pos[k])) for k in (0,1))
            total += self._cost(pos, d)
            pos = (pos[0]+d[0], pos[1]+d[1])
        return total

    def _assign_chargers(self) -> None:
        while self.queue and self.get_fleet_counts()['charging_count'] < self.config.charging_capacity:
            i = self.queue.popleft()
            u = self.uavs[i]
            u['charging_start_step'] = self.step_number
            u['charging_count'] += 1
            self._transition(i, Status.CHARGING, 'charging_started')

    def _arrive(self, i: int) -> None:
        self._transition(i, Status.WAITING, 'returned_to_vehicle')
        self.queue.append(i)
        self._log_event('entered_charging_queue', i, Status.WAITING, Status.WAITING)
        self._assign_chargers()

    def _check_done(self) -> bool:
        threshold = self.config.stop_uncertainty_threshold
        return self.step_number >= self.config.max_episode_steps or self.remaining_time <= 0 or (threshold is not None and float(np.mean(self.uncertainty_matrix)) <= threshold)

    def step(self, actions: list[int]) -> tuple[np.ndarray, list[float], bool]:
        if len(actions) != self.N or any(not 0 <= int(a) < 8 for a in actions):
            raise ValueError('Expected one legal action per fixed UAV slot')
        c = self.config
        self.step_number += 1
        initially_active = set(np.flatnonzero(self.get_search_active_mask()).tolist())
        charging_before = [i for i,u in enumerate(self.uavs) if u['status'] == Status.CHARGING]
        # Launch once per replacement request. New aircraft become eligible next step.
        for i in sorted(initially_active):
            u = self.uavs[i]
            if u['energy']/self.initial_energy <= c.launch_trigger_ratio and not u['replacement_requested']:
                reserve = next((j for j,v in enumerate(self.uavs) if v['status'] == Status.STANDBY), None)
                if reserve is not None:
                    v = self.uavs[reserve]
                    u['replacement_requested'] = True
                    u['replacement_id'] = reserve
                    u['replacement_launch_step'] = self.step_number
                    v['replacement_for'] = i
                    v['activation_count'] += 1
                    v['launch_step'] = self.step_number
                    self._log_event('replacement_requested', i, Status.ACTIVE, Status.ACTIVE, replacement_id=reserve)
                    self._transition(reserve, Status.ACTIVE, 'standby_to_active')
                    self._log_event('reserve_launched', reserve, Status.STANDBY, Status.ACTIVE, replacement_for=i)
            return_cost = self._return_cost(tuple(u['position']))
            # Account for the planned move, worst-case local compute, and the
            # actual wind-dependent return path from its prospective position.
            pos = tuple(u['position'])
            direction = self.directions[int(actions[i])]
            candidate = (pos[0] + direction[0], pos[1] + direction[1])
            if not (0 <= candidate[0] < self.Lx and 0 <= candidate[1] < self.Ly):
                candidate = pos
            local_compute_bound = self.kn * self.μ * float(np.max(self.task_matrix)) * self.f_ln ** 2
            after_move_return = max(return_cost, self._return_cost(candidate))
            safety = after_move_return * (1+c.return_energy_margin_ratio) + self._cost(pos, direction) + local_compute_bound
            handover_done = not u['replacement_requested'] or self.step_number - u['replacement_launch_step'] >= c.handover_min_steps
            if u['energy'] <= safety or (u['energy']/self.initial_energy <= c.return_trigger_ratio and handover_done):
                u['return_start_step'] = self.step_number
                self._transition(i, Status.RETURNING, 'active_to_returning')
        self.eligible = {i for i in initially_active if self.uavs[i]['status'] == Status.ACTIVE}
        rewards = [0.0]*self.N
        for i,u in enumerate(self.uavs):
            status = u['status']
            if status == Status.RETURNING:
                pos = tuple(u['position'])
                home = c.charging_vehicle_position
                if pos == home:
                    self._arrive(i)
                    continue
                d = tuple(int(np.sign(home[k]-pos[k])) for k in (0,1))
                cost = self._cost(pos, d)
                if u['energy'] < cost:
                    if not u['exhausted']:
                        u['exhausted'] = True
                        self._log_event('energy_exhausted_before_return', i, status, status)
                    continue
                u['energy'] -= cost
                u['position'] = (pos[0]+d[0], pos[1]+d[1])
                self._log_event('return_move', i, status, status, from_position=list(pos))
                if u['position'] == home:
                    self._arrive(i)
            elif i in self.eligible:
                d = self.directions[int(actions[i])]
                pos = tuple(u['position'])
                next_pos = (pos[0]+d[0], pos[1]+d[1])
                penalty = 0.0
                before = float(self.uncertainty_matrix.sum())
                if not (0 <= next_pos[0] < self.Lx and 0 <= next_pos[1] < self.Ly):
                    penalty += 0.05
                elif any(tuple(v['position']) == next_pos for v in self.uavs if v is not u and v['status'] == Status.ACTIVE):
                    penalty += 0.05
                else:
                    if self.uncertainty_matrix[next_pos] == 1:
                        penalty -= 0.01
                    u['position'] = next_pos
                cost = self._cost(pos, d)
                u['energy'] = max(0.0, u['energy']-cost)
                self._update_uncertainty_matrix(u['position'])
                rewards[i] = (before-float(self.uncertainty_matrix.sum()))/100-penalty-cost/5e5
        for i in charging_before:
            u = self.uavs[i]
            u['energy'] = min(self.initial_energy, u['energy']+c.charge_rate_per_step)
            if u['energy'] >= self.initial_energy:
                u['full_charge_step'] = self.step_number
                self._log_event('charging_completed', i, Status.CHARGING, Status.CHARGING)
                u['replacement_requested'] = False
                u['replacement_id'] = None
                u['replacement_launch_step'] = None
                u['replacement_for'] = None
                self._transition(i, Status.STANDBY, 'charging_to_standby')
        self._assign_chargers()
        self.remaining_time -= self.T
        if c.log_every_step:
            self._log_event('fleet_snapshot')
        done = self._check_done()
        if done:
            self._log_event('episode_finished')
        return self._get_obs(), rewards, done

    # Adapted from the original offload model; fixed-slot masks replace terminal teleport.
    def step_offload(self, actions: list[int]) -> tuple[Any, list[float], bool]:
        """处理无人机的卸载决策, 0:本地处理, 1:卸载到GBS, 2:HAPS, 3:LEO, 4:云服务器"""
        # en: Process UAV offloading decisions, 0: local processing, 1: offload to GBS, 2: HAPS, 3: LEO, 4: cloud server
        if len(actions) != self.N or any(not 0 <= int(a) < 5 for a in actions):
            raise ValueError('Expected one offload action per slot')
        reward = [0] * self.N

        X_t_B = sum(actions[u]==1 for u in range(self.N) if u in self.eligible)  #卸载到GBS的数量 en: Number of UAVs offloading to GBS
        X_t_C = sum(actions[u]==4 for u in range(self.N) if u in self.eligible)  #卸载到云服务器的数量 en: Number of UAVs offloading to cloud server
        #无人机卸载到GBS的数量（这里假设卸载到云服务器需要经过GBS） en: Number of UAVs offloading to GBS (assuming offloading to cloud server requires passing through GBS)
        X_t = X_t_B + X_t_C
        #无人机卸载到HAPS的数量 en: Number of UAVs offloading to HAPS
        X_h = sum(actions[u]==2 for u in range(self.N) if u in self.eligible)
        #无人机卸载到LEO的数量 e
        X_l = sum(actions[u]==3 for u in range(self.N) if u in self.eligible)

        for u, uav in enumerate(self.uavs):
            offload = actions[u]
            if u not in self.eligible:
                continue
            uav['offload'] = offload

            μ = self.μ  #任务大小 en: Task size
            self.Clx_ly = self.task_matrix[uav['position'][0], uav['position'][1]]  #任务难度 e

            #兼容之前只有两种卸载方式的情况 en: Compatible with previous cases with only two offloading methods
            if offload == 0:
                #计算卸载本地产生的能耗 en: Calculate energy consumption for local processing
                T = μ * self.Clx_ly / self.f_ln
                E = self.kn * μ * self.Clx_ly * self.f_ln**2
                uav['energy'] -= E
                reward[u] = (self.T - T)/1e4 - E/1e4    #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            elif offload == 1:
                ε = 0
                #计算与基站的距离 en: Calculate distance to base station
                d2 = self._squared_link_distance_m(uav['position'], self.gbs_position, self.H)
                h_ngt=10**(self.h0/10)/d2 #信道功率增益 en: Channel power gain
                #计算卸载到GBS产生的能耗 en: Calculate energy consumption for offloading to GBS
                E = X_t*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2))*(1+1/6)
                uav['energy'] -= E
                #计算处理时间 = 发送时间+处理时间+接收时间 en: Calculate processing time = transmission time + processing time + reception time
                T = X_t * μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2)) *7/6 + X_t_B * μ * self.Clx_ly / self.f_B
                if T > self.T or uav['link'] == 0:
                    # 如果处理时间超过最大时间，或者没有连接基站，增加惩罚，改为本地处理 en: If processing time exceeds maximum time, or not connected to base station, increase penalty and switch to local processing
                    ε = 0.01
                    uav['energy'] += E
                    #改用本地卸载 en: Switch to local processing
                    T = μ * self.Clx_ly / self.f_ln
                    E = self.kn * μ * self.Clx_ly * self.f_ln**2
                    uav['energy'] -= E

                reward[u] = (self.T - T)/1e4 - E/1e4 - ε   #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            #剩下的卸载方式后面补充 en: The remaining offloading methods will be added later
            elif offload == 2:
                ε = 0
                # 计算与HAPS的距离（水平距离+垂直高度） en: Calculate distance to HAPS (horizontal + vertical)
                d2 = self._squared_link_distance_m(uav['position'], self.haps_position, self.haps_height)
                # HAPS信道功率增益可能与GBS不同，一般空地链路衰减更小 en: HAPS channel power gain may differ from GBS, generally lower free-space path loss
                h_haps = 10**(self.h0/10)/d2  # 可能需要不同的信道模型 en: HAPS channel model may need to be different
                # 计算卸载到HAPS产生的能耗 en: Calculate energy consumption for offloading to HAPS
                E = X_h*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_haps/self.sigma2))*(1+1/6)
                uav['energy'] -= E
                # 计算处理时间 en: Calculate processing time
                T = X_h * μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_haps/self.sigma2))*7/6 + X_h * μ * self.Clx_ly / self.f_h
                if T > self.T:
                    ε = 0.01
                    uav['energy'] += E
                    #改用本地卸载 en: Switch to local processing
                    T = μ * self.Clx_ly / self.f_ln
                    E = self.kn * μ * self.Clx_ly * self.f_ln**2
                    uav['energy'] -= E
                reward[u] = (self.T - T)/1e4 - E/1e4 - ε    #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            elif offload == 3:
                ε = 0
                # 计算与LEO的距离 - LEO轨道高度显著高于HAPS en: Calculate distance to LEO - LEO orbit height is significantly higher than HAPS
                d2 = self.leo_height**2
                # LEO卫星信道模型可能需要考虑更多的自由空间路径损耗 en: LEO satellite channel model may need to consider more free-space path loss
                h_leo = 10**(self.h0/10)/d2  # 可能需要更特殊的信道模型 en: LEO channel model may need to be more specialized
                # 计算卸载到LEO产生的能耗 en: Calculate energy consumption for offloading to LEO
                E = X_l*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_leo/self.sigma2))*(1+1/6)
                uav['energy'] -= E
                # 计算处理时间 en: Calculate processing time
                T = X_l * μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_leo/self.sigma2))*7/6 + X_l * μ * self.Clx_ly / self.f_l
                if T > self.T:
                    ε = 0.01
                    uav['energy'] += E
                    #改用本地卸载 en: Switch to local processing
                    T = μ * self.Clx_ly / self.f_ln
                    E = self.kn * μ * self.Clx_ly * self.f_ln**2
                    uav['energy'] -= E
                reward[u] = (self.T - T)/1e4 - E/1e4 - ε   #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            elif offload == 4:
                ε = 0
                #计算与基站的距离 en: Calculate distance to base station
                d2 = self._squared_link_distance_m(uav['position'], self.gbs_position, self.H)
                h_ngt=10**(self.h0/10)/d2 #信道功率增益 en: Channel power gain
                #计算卸载到GBS产生的能耗 en: Calculate energy consumption for offloading to GBS
                E = X_t*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2))*(1+1/6)
                uav['energy'] -= E
                #计算处理时间 en: Calculate processing time
                T = X_t * μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2))*7/6 + X_t_C * μ * self.Clx_ly / self.f_c + 2*self.d_c/3e8  #多一个来回传输时间
                if T > self.T or uav['link'] == 0:
                    # 如果处理时间超过最大时间，或者没有连接基站，增加惩罚，改为本地处理 en: If processing time exceeds maximum time, or not connected to base station, increase penalty and switch to local processing
                    ε = 0.01
                    uav['energy'] += E
                    #改用本地卸载 en: Switch to local processing
                    T = μ * self.Clx_ly / self.f_ln
                    E = self.kn * μ * self.Clx_ly * self.f_ln**2
                    uav['energy'] -= E
                reward[u] = (self.T - T)/1e4 - E/1e4 - ε   #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            if uav['energy'] < 0:
                uav['energy'] = 0.0
                if not uav['exhausted']:
                    uav['exhausted'] = True
                    self._log_event('energy_exhausted_before_return', u, uav['status'], uav['status'])

        # 卸载是在飞行中进行的，所以不需要计算时间        en: Offloading is done during flight, so no need to calculate time
        self.remaining_time -= 0

        # 检查是否达到终止条件 en: Check if termination conditions are met
        done = self._check_done()

        # 返回状态：图结构 en: Return state: graph structure
        state = self.get_obs_2()

        return state, reward , done
