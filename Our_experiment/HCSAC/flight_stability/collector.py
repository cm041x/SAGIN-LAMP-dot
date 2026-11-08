"""Pre-flight team decisions, rule-controlled returns, and fixed HAPS offload."""
from dataclasses import dataclass
import random
import numpy as np
import torch
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import new_env
from Our_experiment.HCSAC.lamp_method.critics import event_category
from Our_experiment.HCSAC.lamp_method.graph_observation import STATUSES
from .policy import ACTION_DIM, UAV_COUNT, FlightMemory, flight_observe
from .diagnose import reward_components

HAPS_ACTION=2


@dataclass
class FlightEpisode:
    transitions: list
    metrics: dict
    trajectory: list
    actions: list


def _safety_boundary(env, pos, energy, direction) -> bool:
    """Audit the V1.5 decision's same pre-flight safety inequality."""
    candidate=(pos[0]+direction[0],pos[1]+direction[1])
    if not (0<=candidate[0]<env.Lx and 0<=candidate[1]<env.Ly):candidate=pos
    local_bound=env.kn*env.μ*float(np.max(env.task_matrix))*env.f_ln**2
    safety=max(env._return_cost(pos),env._return_cost(candidate))*(1+env.config.return_energy_margin_ratio)
    safety+=env._cost(pos,direction)+local_bound
    return bool(energy<=safety)


def collect_episode(actor,cnn,seeds: dict,max_steps: int,*,random_actions: bool=False,
                    greedy: bool=False,env_override=None) -> FlightEpisode:
    seed=int(seeds['terrain_seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=env_override if env_override is not None else new_env(max_steps,seeds)
    state=env._get_obs();memory=FlightMemory();pending=None;transitions=[];timeline=[];trajectory=[]
    initial_uncertainty=float(env.uncertainty_matrix.sum());cover=[];counts=[];hidden=[];residual_norm=[]
    dampings={s.value:[] for s in STATUSES}; reward_total=0.0
    kl_values=[];ranges=[];saturation=[];raw_hidden=[];team_rewards=[];reward_parts=[]
    for _ in range(max_steps):
        snapshot=flight_observe(env,state,cnn)
        if pending is not None:
            pending['next_snapshot']=snapshot;transitions.append(pending);pending=None
        with torch.no_grad():
            if actor is None:
                probabilities=snapshot.base_probs
            else:
                embedding=memory.advance(actor.encoder,snapshot)
                probabilities=actor.probabilities(embedding,snapshot)
                hidden.append(float(embedding.norm()))
                offsets=actor.residual(embedding,snapshot)
                head_input=torch.cat((actor.encoder.normalized_local(snapshot.local_features),embedding[:UAV_COUNT]),dim=-1)
                pre=actor.residual_head[0](head_input)
                if snapshot.action_mask.any():
                    saturation.append(float((pre[snapshot.action_mask].abs()>3).float().mean()))
                    ranges.extend((offsets[snapshot.action_mask].max(-1).values-offsets[snapshot.action_mask].min(-1).values).tolist())
                    p=probabilities[snapshot.action_mask].clamp_min(1e-8);b=snapshot.base_probs[snapshot.action_mask].clamp_min(1e-8)
                    kl_values.extend((p*(p.log()-b.log())).sum(-1).tolist())
                raw_hidden.append(float((memory.hidden if memory.hidden is not None else embedding).square().mean().sqrt()))
                if snapshot.action_mask.any():residual_norm.extend(offsets[snapshot.action_mask].norm(dim=-1).tolist())
                if hasattr(actor.encoder,'damping'):
                    d=actor.encoder.damping(snapshot.context,10)
                    for i,u in enumerate(env.uavs):dampings[u['status'].value].append(float(d[i].mean()))
            proposed=torch.zeros(UAV_COUNT,dtype=torch.long)
            for i in torch.where(snapshot.action_mask)[0].tolist():
                proposed[i]=random.randrange(ACTION_DIM) if random_actions else int(probabilities[i].argmax() if greedy else torch.distributions.Categorical(probabilities[i]).sample())
        before_positions=[tuple(u['position']) for u in env.uavs]
        uncertainty_before=env.uncertainty_matrix.copy()
        before_energy=[float(u['energy']) for u in env.uavs]
        before_status=[u['status'].value for u in env.uavs]
        first_event=len(env.events)
        state,fly_rewards,flight_done=env.step(proposed.tolist())
        offload_mask=torch.tensor([i in env.eligible for i in range(UAV_COUNT)],dtype=torch.bool)
        reward_parts.append(reward_components(env,before_positions,uncertainty_before,proposed,fly_rewards))
        # Refresh current-position links after flight before executing fixed HAPS.
        env.get_obs_2()
        offload_actions=[HAPS_ACTION if offload_mask[i] else 0 for i in range(UAV_COUNT)]
        energy_before=[float(u['energy']) for u in env.uavs]
        _,off_rewards,off_done=env.step_offload(offload_actions)
        for i in range(UAV_COUNT):
            if not offload_mask[i] and (off_rewards[i]!=0 or env.uavs[i]['energy']!=energy_before[i]):
                raise RuntimeError('Inactive UAV offloaded')
        event_slice=env.events[first_event:];returned={e['uav_id'] for e in event_slice if e['event_type']=='active_to_returning'}
        positions=[tuple(u['position']) for u in env.uavs];executed=[];reasons=[];actual_directions=[]
        for i in range(UAV_COUNT):
            delta=(positions[i][0]-before_positions[i][0],positions[i][1]-before_positions[i][1])
            actual_directions.append(env.directions.index(delta) if delta in env.directions else None)
            if snapshot.action_mask[i] and i in returned:
                executed.append('rule_return')
                reasons.append('safety_energy_boundary' if _safety_boundary(env,before_positions[i],before_energy[i],env.directions[int(proposed[i])]) else 'fleet_return_threshold')
            elif offload_mask[i]:
                executed.append('search_move' if delta!=(0,0) else 'search_blocked');reasons.append(None)
            elif before_status[i]=='returning':executed.append('rule_return');reasons.append(None)
            else:executed.append('no_policy_flight');reasons.append(None)
        reward=float(sum(fly_rewards)+sum(off_rewards));reward_total+=reward;team_rewards.append(reward)
        done=bool(flight_done or off_done)
        transition={'snapshot':snapshot,'actions':proposed.clone(),'reward':reward,
                    'fly_reward':float(sum(fly_rewards)),'offload_reward':float(sum(off_rewards)),
                    'offload_mask':offload_mask.clone(),'event':event_category(event_slice),
                    'terminal':done,'next_snapshot':None,'execution_type':executed,'override_reason':reasons}
        if done:transitions.append(transition)
        else:pending=transition
        base_argmax=snapshot.base_probs.argmax(dim=-1)
        timeline.append({'step':env.step_number,'flight_action_mask':snapshot.action_mask.tolist(),
                         'offload_action_mask':offload_mask.tolist(),
                         'proposed_action':[int(proposed[i]) if snapshot.action_mask[i] else None for i in range(UAV_COUNT)],
                         'base_action':[int(base_argmax[i]) if snapshot.action_mask[i] else None for i in range(UAV_COUNT)],
                         'execution_type':executed,'override_reason':reasons,'actual_direction':actual_directions,
                         'requested_offload':env.offload_timeline[-1]['requested'],
                         'effective_offload':env.offload_timeline[-1]['effective']})
        trajectory.append({'step':env.step_number,'positions':[list(p) for p in positions],
                           'statuses':env.get_uav_statuses()})
        cover.append(1-float(env.uncertainty_matrix.mean()));counts.append(env.get_fleet_counts()['search_active_count'])
        if done:break
    if pending is not None:raise RuntimeError('Missing terminal flush')
    if abs(sum(t['reward'] for t in transitions)-reward_total)>1e-6:raise RuntimeError('Team reward accounting mismatch')
    actives=sum(sum(x['flight_action_mask']) for x in timeline)
    different=sum(sum(a is not None and a!=b for a,b in zip(x['proposed_action'],x['base_action'])) for x in timeline)
    overrides=sum(sum(r is not None for r in x['override_reason']) for x in timeline)
    safety_overrides=sum(sum(r=='safety_energy_boundary' for r in x['override_reason']) for x in timeline)
    request=[sum(x['requested_offload'][i] for x in timeline) for i in range(5)]
    effective=[sum(x['effective_offload'][i] for x in timeline) for i in range(5)]
    event_types=[e['event_type'] for e in env.events]
    stats={'steps':len(cover),'coverage_auc':float(np.mean(cover)),'final_coverage':cover[-1],
           'repeated_visit_ratio':float((env.total_visit_count-(initial_uncertainty-env.uncertainty_matrix.sum()))/max(1,env.total_visit_count)),
           'total_uav_energy_j':float(sum(env.energy_stats[k] for k in ('search_flight_energy_j','return_flight_energy_j','offload_energy_j'))),
           'return_energy_j':float(env.energy_stats['return_flight_energy_j']),
           'integrated_fleet_deficit':sum(max(0,env.config.target_search_uavs-c) for c in counts),
           'flight_proposals':actives,'changed_from_base':different,'action_change_ratio':different/max(1,actives),
           'flight_overrides':overrides,'safety_overrides':safety_overrides,'safety_override_ratio':safety_overrides/max(1,actives),
           'requested_offload_counts':request,'effective_offload_counts':effective,
           'inactive_uav_offload_count':0,'energy_exhausted_events':event_types.count('energy_exhausted_before_return'),
           'return_events':event_types.count('active_to_returning'),'launch_events':event_types.count('standby_to_active'),
           'charging_complete_events':event_types.count('charging_to_standby'),
           'mean_residual_logits_norm':float(np.mean(residual_norm)) if residual_norm else 0.,
           'mean_hidden_norm':float(np.mean(hidden)) if hidden else None,
           'damping_by_status':{k:float(np.mean(v)) if v else None for k,v in dampings.items()},
           'total_team_reward':reward_total,'discounted_team_reward':sum((.99**i)*r for i,r in enumerate(team_rewards)),
           'mean_policy_kl_to_cnn':float(np.mean(kl_values)) if kl_values else 0.,
           'mean_residual_range':float(np.mean(ranges)) if ranges else 0.,
           'mean_actor_tanh_saturation':float(np.mean(saturation)) if saturation else 0.,
           'mean_raw_hidden_rms':float(np.mean(raw_hidden)) if raw_hidden else 0.,
           'direction_counts':[sum(a==i for row in timeline for a in row['proposed_action'] if a is not None) for i in range(8)],
           'direction_counts_last100':[sum(a==i for row in timeline[-100:] for a in row['proposed_action'] if a is not None) for i in range(8)],
           'boundary_blocks':sum(x['boundary_blocks'] for x in reward_parts),
           'occupied_blocks':sum(x['occupied_blocks'] for x in reward_parts),
           'coverage_gain_reward':sum(x['coverage_gain_reward'] for x in reward_parts),
           'first_visit_bonus':sum(x['first_visit_bonus'] for x in reward_parts),
           'block_penalty':sum(x['block_penalty'] for x in reward_parts),
           'search_energy_penalty':sum(x['search_energy_penalty'] for x in reward_parts),
           'total_offload_reward':sum(t['offload_reward'] for t in transitions)}
    return FlightEpisode(transitions,stats,trajectory,timeline)
