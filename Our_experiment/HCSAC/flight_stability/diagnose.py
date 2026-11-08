"""Read-only replay of old checkpoints on validation scenarios."""
from __future__ import annotations
import collections
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from Our_experiment.HCSAC.flight_residual.policy import FlightMemory, flight_observe, encode
from Our_experiment.HCSAC.flight_residual.trainer import TeamSAC
from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import new_env
from Our_experiment.HCSAC.training.training_split import build_split

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/flight_residual_stability'
OLD=ROOT/'outputs/lamp_flight_residual_pilot'
VARIANTS=('gcn_flight_residual','lamp_plain_flight_residual')
OBSERVE_STEPS={1,50,100,200,500}

def jwrite(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def make_agent(variant,trained):
    torch.manual_seed(3001)
    agent=TeamSAC(variant)
    if trained:
        ck=torch.load(OLD/'training'/variant/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        agent.load_state_dict(ck['agent'])
    return agent

def rms(x):
    return float(x.square().mean().sqrt()) if x.numel() else 0.

def fixed_q(agent,snap,actions,h1,h2):
    ids=torch.where(snap.action_mask)[0]
    if not len(ids):return None
    i=int(ids[0]); arr=actions.repeat(8,1);arr[:,i]=torch.arange(8)
    mask=snap.action_mask.expand(8,-1)
    q1=agent.q1.head(h1.expand(8,-1,-1),arr,mask)[0]
    q2=agent.q2.head(h2.expand(8,-1,-1),arr,mask)[0]
    q=torch.minimum(q1,q2)
    return {'uav_id':i,'q1':q1.tolist(),'q2':q2.tolist(),'twin_min':q.tolist(),
            'span':float(q.max()-q.min()),'argmax':int(q.argmax())}

def reward_components(env, before, uncert, actions, fly):
    # Reconstruct the environment's sequential collision and uncertainty logic.
    occupied={i:before[i] for i,u in enumerate(env.uavs) if u['status'].value=='active'}
    gains=bonuses=boundary=occupied_blocks=energy_penalty=0.
    for i in sorted(env.eligible):
        d=env.directions[int(actions[i])];p=before[i];n=(p[0]+d[0],p[1]+d[1])
        if not(0<=n[0]<env.Lx and 0<=n[1]<env.Ly):boundary+=1;n=p
        elif any(j!=i and v==n for j,v in occupied.items()):occupied_blocks+=1;n=p
        else:
            if uncert[n]==1:bonuses+=.01
            occupied[i]=n
        previous=float(uncert[n]); uncert[n]*=(1-env.delta)
        gains+=(previous-float(uncert[n]))/100
        energy_penalty+=env._cost(p,d)/5e5
    reconstructed=gains+bonuses-.05*(boundary+occupied_blocks)-energy_penalty
    if abs(reconstructed-sum(fly))>1e-5:
        raise AssertionError(f'Flight reward decomposition mismatch: {reconstructed}, {sum(fly)}')
    return {'coverage_gain_reward':gains,'first_visit_bonus':bonuses,'boundary_blocks':int(boundary),
            'occupied_blocks':int(occupied_blocks),'block_penalty':.05*(boundary+occupied_blocks),
            'search_energy_penalty':energy_penalty}

def rollout(scenario,variant=None):
    random.seed(scenario['terrain_seed']);np.random.seed(scenario['terrain_seed']);torch.manual_seed(scenario['terrain_seed'])
    cnn=frozen_flight_actor();agent=make_agent(variant,True) if variant else None
    env=new_env(500,scenario);state=env._get_obs();memory=FlightMemory();h1=h2=None
    steps=[];snaps=[];rewards=[];positions_seen=[collections.Counter() for _ in range(6)]
    old_init=float(env.uncertainty_matrix.sum());snap_actions=[]
    for t in range(500):
        snap=flight_observe(env,state,cnn);snaps.append(snap)
        with torch.no_grad():
            if agent:
                h=memory.advance(agent.actor.encoder,snap)
                p=agent.actor.probabilities(h,snap)
                raw=agent.actor.residual_head(torch.cat((snap.local_features,h[:6]),-1))
                qh1,h1=encode(agent.q1.encoder,snap,h1);qh2,h2=encode(agent.q2.encoder,snap,h2)
            else:p=snap.base_probs;raw=torch.zeros((6,8))
            actions=torch.zeros(6,dtype=torch.long);actions[snap.action_mask]=p[snap.action_mask].argmax(-1)
            qdiag=fixed_q(agent,snap,actions,qh1,qh2) if agent else None
            ids=torch.where(snap.action_mask)[0]
            p_active=p[ids].clamp_min(1e-8);base=snap.base_probs[ids].clamp_min(1e-8)
            kl=float((p_active*(p_active.log()-base.log())).sum(-1).mean()) if len(ids) else 0.
            top=base.log().topk(2,dim=-1).values
            margin=(top[:,0]-top[:,1]).tolist() if len(ids) else []
        pre=[tuple(u['position']) for u in env.uavs];uncert=env.uncertainty_matrix.copy()
        state,fly,done=env.step(actions.tolist());eligible=set(env.eligible)
        component=reward_components(env,pre,uncert,actions,fly)
        env.get_obs_2();off=[2 if i in eligible else 0 for i in range(6)]
        _,offreward,done2=env.step_offload(off);reward=float(sum(fly)+sum(offreward));rewards.append(reward)
        post=[tuple(u['position']) for u in env.uavs]
        for i in eligible:positions_seen[i][post[i]]+=1
        active_ids=ids.tolist();diff=[int(actions[i]!=snap.base_probs[i].argmax()) for i in active_ids]
        steps.append({'step':t+1,'ids':active_ids,'actions':[int(actions[i]) for i in active_ids],
                      'positions':[list(post[i]) for i in active_ids],'kl':kl,'base_margin':margin,
                      'residual_range':(raw[ids].max(-1).values-raw[ids].min(-1).values).tolist(),
                      'residual_logits':raw[ids].tolist(),'local_rms':rms(snap.local_features[ids]),
                      'action_diff':diff,'q':qdiag,'fly_reward':float(sum(fly)),'offload_reward':float(sum(offreward)),
                      'coverage':1-float(env.uncertainty_matrix.mean()),**component})
        snap_actions.append(actions)
        if done or done2:break
    n=sum(len(s['ids']) for s in steps);counts=[sum(s['actions'].count(a) for s in steps) for a in range(8)]
    discounted=sum((.99**t)*r for t,r in enumerate(rewards))
    summary={'scenario':scenario,'policy':variant or 'frozen_cnn_reference','steps':len(steps),
      'final_coverage':steps[-1]['coverage'],'coverage_auc':float(np.mean([s['coverage'] for s in steps])),
      'team_reward':sum(rewards),'discounted_team_reward':discounted,'coverage_gain':old_init-float(env.uncertainty_matrix.sum()),
      'coverage_gain_reward':sum(s['coverage_gain_reward'] for s in steps),'first_visit_bonus':sum(s['first_visit_bonus'] for s in steps),
      'block_penalty':sum(s['block_penalty'] for s in steps),'boundary_blocks':sum(s['boundary_blocks'] for s in steps),
      'occupied_blocks':sum(s['occupied_blocks'] for s in steps),'search_energy_penalty':sum(s['search_energy_penalty'] for s in steps),
      'offload_reward':sum(s['offload_reward'] for s in steps),'return_energy_j':float(env.energy_stats['return_flight_energy_j']),
      'return_energy_in_reward':False,'action_counts':counts,'action_diff_ratio':sum(sum(s['action_diff']) for s in steps)/max(n,1),
      'repeated_position_visits':sum(sum(c.values())-len(c) for c in positions_seen),
      'unique_positions_per_uav':[len(c) for c in positions_seen],
      'max_3x3_visit_fraction_per_uav':[max((sum(n for (x,y),n in visits.items() if abs(x-cx)<=1 and abs(y-cy)<=1)
            for cx in range(env.Lx) for cy in range(env.Ly)),default=0)/max(1,sum(visits.values())) for visits in positions_seen]}
    return summary,steps,snaps,snap_actions

def segment_stats(steps):
    out=[]
    for start in range(0,len(steps),100):
        chunk=steps[start:start+100]; acts=[a for s in chunk for a in s['actions']]
        ranges=[v for s in chunk for v in s['residual_range']];margins=[v for s in chunk for v in s['base_margin']]
        out.append({'start':start+1,'end':start+len(chunk),'direction_counts':[acts.count(i) for i in range(8)],
                    'boundary_blocks':sum(s['boundary_blocks'] for s in chunk),'occupied_blocks':sum(s['occupied_blocks'] for s in chunk),
                    'mean_kl':float(np.mean([s['kl'] for s in chunk])),
                    'mean_residual_range':float(np.mean(ranges)) if ranges else None,
                    'mean_base_margin':float(np.mean(margins)) if margins else None,
                    'changed':sum(sum(s['action_diff']) for s in chunk),'proposals':len(acts)})
    # Max same-action run, tracked per UAV and reset across inactive intervals.
    for row in out:
        lengths=[]
        for i in range(6):
            last=None;run=0
            for s in steps[row['start']-1:row['end']]:
                a=s['actions'][s['ids'].index(i)] if i in s['ids'] else None
                if a is not None and a==last:run+=1
                else:
                    if run:lengths.append(run)
                    run=1 if a is not None else 0;last=a
            if run:lengths.append(run)
        row['longest_same_action_run']=max(lengths,default=0)
    return out

def replays(snaps,actions,variant,trained,xi):
    agent=make_agent(variant,trained)
    if variant=='lamp_plain_flight_residual':
        for net in (agent.actor,agent.q1,agent.q2):net.encoder.xi=xi
    hidden=qh1=qh2=None;records=[]
    for t,(s,a) in enumerate(zip(snaps,actions),1):
        local=torch.cat((s.local_features,s.x.new_zeros((4,128))),0)
        x=torch.cat((s.x,s.context,local),-1)
        with torch.no_grad():
            previous=hidden
            h,hidden=encode(agent.actor.encoder,s,hidden)
            q1,qh1=encode(agent.q1.encoder,s,qh1);q2,qh2=encode(agent.q2.encoder,s,qh2)
            if t not in OBSERVE_STEPS:continue
            node_pre=agent.actor.residual_head[0](torch.cat((s.local_features,h[:6]),-1))
            act=torch.where(s.action_mask,a,torch.full_like(a,8));allact=torch.cat((act,torch.full((4,),8)))
            critic_pre=agent.q1.head.node_fusion[0](torch.cat((q1,agent.q1.head.action_embedding(allact)),-1))
            injected=x@agent.actor.encoder.U if hasattr(agent.actor.encoder,'U') else None
            initial=torch.zeros_like(h) if previous is None else previous
            before_update=(injected+initial)/xi if injected is not None else None
            records.append({'step':t,'x_base_rms':rms(s.x),'x_context_rms':rms(s.context),
                'local_rms':rms(s.local_features),'local_max_abs':float(s.local_features.abs().max()),
                'x_injection_rms':rms(injected) if injected is not None else None,
                'hidden_norm':float(h.norm()),'hidden_node_rms':h.square().mean(-1).sqrt().tolist(),
                'previous_hidden_norm':float(initial.norm()),'previous_hidden_node_rms':initial.square().mean(-1).sqrt().tolist(),
                'pre_propagation_hidden_norm':float(before_update.norm()) if before_update is not None else None,
                'pre_propagation_hidden_node_rms':before_update.square().mean(-1).sqrt().tolist() if before_update is not None else None,
                'hidden_change_rms':rms(h-initial),
                'post_propagation_minus_injected_rms':rms(h-before_update) if before_update is not None else None,
                'actor_tanh_saturation':float((node_pre.abs()>3).float().mean()),
                'critic_tanh_saturation':float((critic_pre.abs()>3).float().mean()),
                'q':fixed_q(agent,s,a,q1,q2)})
    return records

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vals=build_split()['validation_combinations'];scenario=vals[0]
    all_rows={};base_snaps=base_actions=None
    for name in (None,*VARIANTS):
        summary,steps,snaps,actions=rollout(scenario,name);key=name or 'frozen_cnn_reference'
        all_rows[key]={'summary':summary,'segments':segment_stats(steps)}
        jwrite(OUT/'diagnostics'/f'{key}_steps.json',steps)
        if name is None:base_snaps,base_actions=snaps,actions
        print('diagnosed',key,summary['final_coverage'],flush=True)
    replay_results={}
    for variant in VARIANTS:
        for trained in (False,True):
            for xi in ((1.,2.) if variant==VARIANTS[1] else (1.,)):
                key=f'{variant}_{"trained" if trained else "initial"}_xi{xi:g}'
                replay_results[key]=replays(base_snaps,base_actions,variant,trained,xi)
                print('replayed',key,flush=True)
    jwrite(OUT/'diagnostics'/'validation_diagnosis.json',{'scenario':scenario,'behavior':all_rows,'same_cnn_trajectory_replay':replay_results})
    torch.save({'scenario':scenario,'snapshots':base_snaps,'actions':base_actions},OUT/'diagnostics'/'cnn_validation_trajectory.pt')
    print('diagnosis complete',flush=True)

if __name__=='__main__':main()
