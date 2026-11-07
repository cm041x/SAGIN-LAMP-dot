"""Pending-transition collection between actual offloading decision snapshots."""
from dataclasses import dataclass
import random
import numpy as np
import torch
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import PolicyNet_CNN
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import HCSAC, new_env
from .graph_observation import observe, STATUSES
from .critics import event_category
from .policy import SnapshotMemory


def frozen_flight_actor():
    actor=PolicyNet_CNN(15,128,8)
    checkpoint=torch.load(HCSAC/'data/sac_model_fly.pt',map_location='cpu',weights_only=True)
    actor.load_state_dict(checkpoint['actor_state_dict'],strict=True)
    actor.eval()
    for p in actor.parameters(): p.requires_grad_(False)
    return actor


@dataclass
class CollectedEpisode:
    transitions: list
    prefix_flight_reward: float
    total_physical_reward: float
    metrics: dict


def collect_episode(actor, flight_actor, seeds: dict, max_steps: int,
                    *, random_actions: bool=False, greedy: bool=False) -> CollectedEpisode:
    seed=int(seeds['terrain_seed']); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env=new_env(max_steps,seeds); state=env._get_obs(); memory=SnapshotMemory()
    initial_uncertainty=float(env.uncertainty_matrix.sum())
    pending=None; transitions=[]; prefix=0.0; physical_reward=0.0; coverage=[]; counts=[]
    hidden_norms=[]; dampings={s.value:[] for s in STATUSES}
    for step in range(max_steps):
        ids=np.flatnonzero(env.get_search_active_mask()).tolist(); fly_actions=[0]*6
        if ids:
            with torch.no_grad():
                chosen=flight_actor(torch.as_tensor(state[ids],dtype=torch.float32)).argmax(-1).tolist()
            for i,a in zip(ids,chosen): fly_actions[i]=a
        state,fly_reward,flight_done=env.step(fly_actions)
        fly_sum=sum(fly_reward); physical_reward+=fly_sum
        snapshot=observe(env)
        if pending is None:
            prefix=fly_sum
        else:
            pending.update(next_snapshot=snapshot,reward=pending.pop('offload_reward')+fly_sum,
                           event=event_category(env.events[pending.pop('event_start'):]),terminal=False)
            transitions.append(pending)
        with torch.no_grad():
            embedding=memory.advance(actor.encoder,snapshot)
            probs=actor.probabilities(embedding)
            hidden_norms.append(float(embedding.norm()))
            if hasattr(actor.encoder,'damping'):
                d=actor.encoder.damping(snapshot.context,10)
                for i,u in enumerate(env.uavs): dampings[u['status'].value].append(float(d[i].mean()))
            actions=torch.zeros(6,dtype=torch.long)
            for i in torch.where(snapshot.action_mask)[0].tolist():
                actions[i]=random.randrange(5) if random_actions else int(probs[i].argmax() if greedy else torch.distributions.Categorical(probs[i]).sample())
        events_before_offload=len(env.events)
        energy_before=[u['energy'] for u in env.uavs]
        _,off_rewards,off_done=env.step_offload(actions.tolist())
        for i in range(env.N):
            if not snapshot.action_mask[i] and (off_rewards[i] != 0 or env.uavs[i]['energy'] != energy_before[i]):
                raise RuntimeError('inactive UAV was processed during offloading')
        off_sum=sum(off_rewards);physical_reward+=off_sum
        pending={'snapshot':snapshot,'actions':actions.clone(),'offload_reward':off_sum,'event_start':events_before_offload}
        coverage.append(1-float(env.uncertainty_matrix.mean()))
        counts.append(env.get_fleet_counts()['search_active_count'])
        if flight_done or off_done:
            pending.update(next_snapshot=None,reward=pending.pop('offload_reward'),event=0,terminal=True)
            pending.pop('event_start');transitions.append(pending); pending=None
            break
    if pending is not None: raise RuntimeError('episode did not flush terminal transition')
    assigned=prefix+sum(t['reward'] for t in transitions)
    if abs(assigned-physical_reward)>1e-6: raise RuntimeError('physical reward accounting mismatch')
    event_types=[e['event_type'] for e in env.events]
    return CollectedEpisode(transitions,prefix,physical_reward,{
        'steps':len(coverage),'final_coverage':coverage[-1], 'coverage_auc':float(np.mean(coverage)),
        'returns':event_types.count('active_to_returning'), 'launches':event_types.count('standby_to_active'),
        'charges_completed':event_types.count('charging_to_standby'),
        'energy_exhausted':event_types.count('energy_exhausted_before_return'),
        'uav_energy_j':sum(env.energy_stats[k] for k in ('search_flight_energy_j','return_flight_energy_j','offload_energy_j')),
        'repeated_visit_ratio':(env.total_visit_count-(initial_uncertainty-env.uncertainty_matrix.sum()))/max(1,env.total_visit_count),
        'integrated_fleet_deficit':sum(max(0,env.config.target_search_uavs-c) for c in counts),
        'mean_hidden_norm':float(np.mean(hidden_norms)),
        'damping_by_status':{k:float(np.mean(v)) if v else None for k,v in dampings.items()},
        'damping_by_status_samples':dampings,
        'inactive_offload_count':0})
