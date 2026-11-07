"""Current-state-only lifecycle context with stable node and action masks."""
from dataclasses import dataclass
import torch
from torch import Tensor
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import Status

STATUSES = (Status.STANDBY, Status.ACTIVE, Status.RETURNING, Status.WAITING, Status.CHARGING)
CONTEXT_SCHEMA = ('is_uav', 'type_uav', 'type_gbs', 'type_haps', 'type_leo', 'type_cloud',
                  'standby', 'active', 'returning', 'waiting_charge', 'charging',
                  'energy_ratio', 'x', 'y', 'time_since_transition', 'waiting_time',
                  'charger_occupancy', 'queue_length', 'remaining_time')


@dataclass(frozen=True)
class GraphSnapshot:
    x: Tensor
    context: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    node_ids: Tensor
    existence_mask: Tensor
    action_mask: Tensor
    snapshot_id: int


def observe(env) -> GraphSnapshot:
    data = env.get_obs_2()
    if data.num_nodes != 10 or not torch.equal(data.slot_id[:6], torch.arange(6)):
        raise ValueError('Expected fixed UAV slots 0..5 and four infrastructure nodes')
    out = torch.zeros((10, len(CONTEXT_SCHEMA)), dtype=data.x.dtype, device=data.x.device)
    counts = env.get_fleet_counts()
    horizon = float(env.config.max_episode_steps)
    for i in range(10):
        kind = int(data.node_type[i])
        out[i, 0] = float(kind == 0)
        out[i, 1 + kind] = 1
        out[i, 16] = counts['charging_count'] / env.config.charging_capacity
        out[i, 17] = counts['waiting_charge_count'] / env.N
        out[i, 18] = max(0.0, env.remaining_time / env.max_search_time)
        if kind == 0:
            u = env.uavs[i]
            out[i, 6 + STATUSES.index(u['status'])] = 1
            out[i, 11] = u['energy'] / env.initial_energy
            out[i, 12] = u['position'][0] / (env.Lx-1)
            out[i, 13] = u['position'][1] / (env.Ly-1)
            out[i, 14] = (env.step_number-u['last_transition_step']) / horizon
            wait = u.get('waiting_since_step')
            out[i, 15] = 0 if wait is None else (env.step_number-wait) / horizon
    mask = torch.tensor([i in env.eligible for i in range(6)], dtype=torch.bool)
    return GraphSnapshot(data.x.float().clone(), out.float(), data.edge_index.clone(),
                         data.edge_weight.float().clone(), torch.arange(10),
                         torch.ones(10, dtype=torch.bool), mask, env.step_number)
