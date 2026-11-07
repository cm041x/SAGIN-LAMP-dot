"""Bounded, resumable training of unchanged HCSAC networks in V1.5."""
import csv
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleetV15 import DynamicFleetV15Config,DynamicFleetV15Env
from Our_experiment.HCSAC.ENV.handover_policy import HandoverMode
from Our_experiment.HCSAC.ENV.fleet_scheduler import SchedulerMode
from Our_experiment.HCSAC.dynamic_fleet_v15_experiment import ROOT,HCSAC,load_positions
from Our_experiment.HCSAC.dynamic_fleet_v15_metrics import summarize_v15
from Our_experiment.HCSAC.dynamic_fleet_v1_experiment import fleet_deficit
from Our_experiment.HCSAC.training.masked_discrete_sac import MaskedDiscreteSAC
from Our_experiment.HCSAC.training.replay_buffers import CNNReplay,GraphReplay
from Our_experiment.HCSAC.training.checkpointing import agent_state,restore_agent,rng_state,restore_rng,atomic_save,module_hash,file_hash
from Our_experiment.HCSAC.training.training_split import build_split,episode_manifest

class TrainingMode(str,Enum):
    SCRATCH='scratch'
    FINETUNE='finetune'

def select_device(choice: str) -> str:
    if choice=='auto':return 'cuda' if torch.cuda.is_available() else 'cpu'
    if choice=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    if choice not in ('cpu','cuda'):raise ValueError('Unsupported device')
    return choice

def manifest_hash(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:return
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def choose_flight(agent: MaskedDiscreteSAC, state: np.ndarray, active: np.ndarray, *, random_actions: bool, greedy: bool) -> list[int]:
    actions=[0]*6;ids=np.flatnonzero(active).tolist()
    if random_actions:
        for i in ids:actions[i]=random.randrange(8)
    elif ids:
        with torch.no_grad():
            p=agent.actor(torch.as_tensor(state[ids],dtype=torch.float32,device=agent.device))
            chosen=p.argmax(-1) if greedy else torch.distributions.Categorical(p).sample()
        for i,a in zip(ids,chosen.tolist()):actions[i]=a
    return actions

def choose_offload(agent: MaskedDiscreteSAC, graph, eligible: set[int], *, random_actions: bool, greedy: bool) -> list[int]:
    actions=[0]*6
    if random_actions:
        for i in eligible:actions[i]=random.randrange(5)
    elif eligible:
        ids=sorted(eligible)
        with torch.no_grad():
            p=agent._graph(agent.actor,[graph])[0][ids]
            chosen=p.argmax(-1) if greedy else torch.distributions.Categorical(p).sample()
        for i,a in zip(ids,chosen.tolist()):actions[i]=a
    return actions

def collect_step(env, state, fly, off, cnn_buffer, graph_buffer, episode: int, step: int,
                 *, warmup: bool=False, greedy: bool=False, store: bool=True) -> tuple:
    previous_mask=env.get_search_active_mask();actions=choose_flight(fly,state,previous_mask,random_actions=warmup,greedy=greedy)
    next_state,fly_reward,done=env.step(actions);executed=set(env.eligible)
    next_active=env.get_search_active_mask()
    if any(not previous_mask[i] for i in executed):raise RuntimeError('New/inactive slot executed a search')
    if store:
        for i in executed:
            cnn_buffer.add(state[i],actions[i],fly_reward[i],next_state[i],done or not next_active[i],i,episode,step,
                           provenance={'active_before':bool(previous_mask[i]),'executed_slots':sorted(executed),'next_active':bool(next_active[i])})
    graph=env.get_obs_2();mask=np.array([i in executed for i in range(6)],dtype=bool)
    off_actions=choose_offload(off,graph,executed,random_actions=warmup,greedy=greedy)
    before=[u['energy'] for u in env.uavs]
    next_graph,off_reward,off_done=env.step_offload(off_actions)
    for i in set(range(6))-executed:
        if off_reward[i]!=0 or env.uavs[i]['energy']!=before[i]:raise RuntimeError('Inactive offload detected')
    if not all(np.isfinite(u['energy']) and 0<=u['energy']<=env.initial_energy for u in env.uavs):raise FloatingPointError('Invalid energy')
    nextmask=env.get_search_active_mask();dones=np.array([done or off_done or not nextmask[i] for i in range(6)],dtype=bool)
    if store:graph_buffer.add(graph,off_actions,off_reward,next_graph,mask,nextmask,dones,done or off_done,episode,step)
    return next_state,fly_reward,off_reward,bool(done or off_done),int((~previous_mask).sum()),int(previous_mask.sum()-len(executed))

def new_env(max_steps: int, seeds: dict) -> DynamicFleetV15Env:
    env=DynamicFleetV15Env(DynamicFleetV15Config(scheduler_mode=SchedulerMode.TARGET_AWARE,
        handover_mode=HandoverMode.NONE,target_search_uavs=4,charging_capacity=1,charging_power_w=600,
        charging_efficiency=.9,minimum_launch_energy_ratio=.95,max_episode_steps=max_steps))
    env.reset(seed=seeds['terrain_seed'],wind_seed=seeds['wind_seed'],terrain_seed=seeds['terrain_seed'],
              infra_seed=seeds['infra_seed'],positions=load_positions())
    return env

def evaluate_episode(fly,off,seeds,max_steps: int) -> dict:
    env=new_env(max_steps,seeds);state=env._get_obs();initial=float(env.uncertainty_matrix.sum());counts=[];covers=[]
    for step in range(max_steps):
        state,_,_,done,_,_=collect_step(env,state,fly,off,None,None,0,step,greedy=True,store=False)
        counts.append(env.get_fleet_counts()['search_active_count']);covers.append(1-float(env.uncertainty_matrix.mean()))
        if done:break
    stats=summarize_v15(env,counts,initial);energy=env.energy_stats
    return {**stats,'final_coverage':covers[-1],'coverage_auc':float(np.mean(covers)),
            'integrated_fleet_deficit':fleet_deficit(counts,4),'target_met_ratio':sum(x>=4 for x in counts)/len(counts),
            'total_uav_energy_consumed_j':energy['search_flight_energy_j']+energy['return_flight_energy_j']+energy['offload_energy_j'],
            'steps':len(counts),'inactive_uav_offload_count':0}

def mean_validation(fly,off,split,max_steps: int) -> dict:
    saved=rng_state()
    try:
        scores=[evaluate_episode(fly,off,s,max_steps) for s in split['validation_combinations']]
        keys=('coverage_auc','final_coverage','repeated_visit_ratio','total_uav_energy_consumed_j')
        return {k:float(np.mean([s[k] for s in scores])) for k in keys}
    finally:restore_rng(saved)

def initialize_agents(mode: TrainingMode, device: str) -> tuple:
    fly=MaskedDiscreteSAC('CNN',device);off=MaskedDiscreteSAC('GCN',device)
    if mode==TrainingMode.FINETUNE:
        fly.load_original(str(HCSAC/'data/sac_model_fly.pt'));off.load_original(str(HCSAC/'data/sac_model_offload.pt'))
        # Verify the saved actors through forward passes without consuming training RNG.
        saved=rng_state()
        try:
            from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import PolicyNet_CNN,PolicyNet_GCN
            from torch_geometric.data import Data
            ref_cnn=PolicyNet_CNN(15,128,8).to(device)
            ref_gcn=PolicyNet_GCN(6,128,5).to(device)
            ref_cnn.load_state_dict(torch.load(HCSAC/'data/sac_model_fly.pt',map_location=device,weights_only=True)['actor_state_dict'],strict=True)
            ref_gcn.load_state_dict(torch.load(HCSAC/'data/sac_model_offload.pt',map_location=device,weights_only=True)['actor_state_dict'],strict=True)
            probe=torch.zeros((1,3,15,15),device=device)
            graph=Data(x=torch.tensor([[0.,1,0,0,0,0]]*6+[[0.,0,1,0,0,0],[0.,0,0,1,0,0],[0.,0,0,0,1,0],[0.,0,0,0,0,1]],device=device),
                       edge_index=torch.tensor([[0,6,1,7],[6,0,7,1]],dtype=torch.long,device=device),
                       edge_weight=torch.ones(4,device=device),node_type=torch.tensor([0]*6+[1,2,3,4],device=device))
            with torch.no_grad():
                if not torch.equal(fly.actor(probe),ref_cnn(probe)) or not torch.equal(off.actor(graph),ref_gcn(graph)):
                    raise RuntimeError('Finetune actor forward differs from pretrained checkpoint')
            if fly.actor_optimizer.state or off.actor_optimizer.state:raise RuntimeError('Finetune optimizers are not fresh')
        finally:restore_rng(saved)
    return fly,off

def train_run(mode: TrainingMode, seed: int, episodes: int, max_steps: int, warmup: int, batch_size: int,
              validation_every: int, checkpoint_every: int, output: Path, device: str, resume: bool) -> dict:
    started=time.monotonic();root=output/mode.value/f'seed_{seed}';root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    split=build_split(); schedules=episode_manifest(seed,episodes,split)
    output.mkdir(parents=True,exist_ok=True)
    (output/'training_split.json').write_text(json.dumps(split,indent=2)+'\n')
    (output/'validation_manifest.json').write_text(json.dumps(split['validation_combinations'],indent=2)+'\n')
    write_csv(output/f'episode_manifest_seed_{seed}.csv',schedules)
    write_csv(root/'episode_manifest.csv',schedules)
    code_files=sorted((HCSAC/'training').glob('*.py'))+[HCSAC/'ENV/UAVenv_SAC_Original.py',HCSAC/'ENV/UAVenv_DynamicFleetV15.py',HCSAC/'ENV/UAVenv_DynamicFleetV1.py',HCSAC/'ENV/UAVenv_DynamicFleet.py',HCSAC/'ENV/fleet_scheduler.py']
    code_fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in code_files)).hexdigest()
    config={'code_fingerprint':code_fingerprint,'mode':mode.value,'seed':seed,'max_steps':max_steps,'warmup_steps':warmup,'batch_size':batch_size,
            'validation_every':validation_every,'checkpoint_every':checkpoint_every,'device':device,
            'actor_lr':3e-4,'critic_lr':3e-4,'alpha_lr':1e-4,'gamma':.99,'tau':.005,
            'cnn_capacity':20000,'gcn_capacity':20000,'initial_energy_j':360000,'target':4,
            'charging_capacity':1,'charging_power_w':600,'charging_efficiency':.9,'initial_positions':load_positions()}
    cfg=root/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=json.loads(json.dumps(config)):raise ValueError('Resume configuration mismatch')
    cfg.write_text(json.dumps(config,indent=2)+'\n')
    fly,off=initialize_agents(mode,device);cnn_buffer=CNNReplay(20000);graph_buffer=GraphReplay(20000)
    fingerprints={'mode':mode.value,'original_fly_checkpoint_sha256':file_hash(HCSAC/'data/sac_model_fly.pt'),
       'original_offload_checkpoint_sha256':file_hash(HCSAC/'data/sac_model_offload.pt'),
       'fly_actor_sha256':module_hash(fly.actor),'offload_actor_sha256':module_hash(off.actor),
       'fly_critic_1_sha256':module_hash(fly.critic_1),'offload_critic_1_sha256':module_hash(off.critic_1),
       'pretrained_forward_verified':mode==TrainingMode.FINETUNE,
       'finetune_start_fly_actor_sha256':module_hash(fly.actor) if mode==TrainingMode.FINETUNE else None,
       'finetune_start_offload_actor_sha256':module_hash(off.actor) if mode==TrainingMode.FINETUNE else None}
    start_episode=0;global_step=0;history=[];validations=[];best=None;best_episode=None
    last=root/'last_checkpoint.pt'
    if last.exists():
        if not resume:raise ValueError('Run exists; use --resume')
        ck=torch.load(last,map_location='cpu',weights_only=False);start_episode=ck['episode'];global_step=ck['global_step']
        if ck['manifest_prefix_sha256']!=manifest_hash(schedules[:start_episode]):raise ValueError('Episode manifest changed')
        restore_agent(fly,ck['fly']);restore_agent(off,ck['off']);cnn_buffer.load_state_dict(ck['cnn_replay']);graph_buffer.load_state_dict(ck['gcn_replay'])
        history=ck['history'];validations=ck['validations'];best=ck['best'];best_episode=ck['best_episode'];restore_rng(ck['rng'])
        print(f'{mode.value}/{seed}: resumed at episode {start_episode}, step {global_step}, replay {len(cnn_buffer)}/{len(graph_buffer)}',flush=True)
    else:(root/'initial_model_fingerprint.json').write_text(json.dumps(fingerprints,indent=2)+'\n')
    for episode in range(start_episode,episodes):
        t=time.monotonic();seeds=schedules[episode];env=new_env(max_steps,seeds);state=env._get_obs()
        initial=float(env.uncertainty_matrix.sum());counts=[];covers=[];fly_return=off_return=0.;fl=[];ol=[];skip=nonexecuted=collected=0
        for step in range(max_steps):
            global_step+=1;state,fr,orr,done,s,n=collect_step(env,state,fly,off,cnn_buffer,graph_buffer,episode+1,step,
                                                            warmup=global_step<=warmup)
            skip+=s;nonexecuted+=n;collected+=len(env.eligible);fly_return+=sum(fr);off_return+=sum(orr)
            counts.append(env.get_fleet_counts()['search_active_count']);covers.append(1-float(env.uncertainty_matrix.mean()))
            if global_step>warmup:
                try:
                    if len(cnn_buffer)>=batch_size:fl.append(fly.update(cnn_buffer.sample(batch_size)))
                    if len(graph_buffer)>=batch_size:ol.append(off.update(graph_buffer.sample(batch_size)))
                except Exception:
                    with (root/'failed_updates.jsonl').open('a') as f:f.write(json.dumps({'episode':episode+1,'step':global_step,'traceback':traceback.format_exc()})+'\n')
                    raise
            if done:break
        stats=summarize_v15(env,counts,initial);energy=env.energy_stats
        def avg(rows,key):
            vals=[x[key] for x in rows if not x.get('skipped',False)];return float(np.mean(vals)) if vals else None
        row={'episode':episode+1,'global_step':global_step,'episode_length':len(counts),'seconds':time.monotonic()-t,
             'fly_return':float(fly_return),'offload_return':float(off_return),'total_return':float(fly_return+off_return),
             'final_coverage':covers[-1],'coverage_auc':float(np.mean(covers)),'repeated_visit_ratio':stats['repeated_visit_ratio'],
             'coverage_gain_per_active_uav_step':stats['coverage_gain_per_active_uav_step'],'integrated_fleet_deficit':fleet_deficit(counts,4),
             'target_met_ratio':sum(x>=4 for x in counts)/len(counts),
             'total_uav_energy_consumed_j':energy['search_flight_energy_j']+energy['return_flight_energy_j']+energy['offload_energy_j'],
             'collected_cnn_transitions':collected,'skipped_inactive_cnn_slots':skip,'skipped_nonexecuted_active_actions':nonexecuted,
             'fly_actor_loss':avg(fl,'actor_loss'),'fly_critic_loss':avg(fl,'critic_loss'),'fly_alpha':fly.log_alpha.exp().item(),'fly_entropy':avg(fl,'entropy'),
             'off_actor_loss':avg(ol,'actor_loss'),'off_critic_loss':avg(ol,'critic_loss'),'off_alpha':off.log_alpha.exp().item(),'off_entropy':avg(ol,'entropy'),
             'cnn_replay_size':len(cnn_buffer),'gcn_replay_size':len(graph_buffer)}
        history.append(row);write_csv(root/'train_history.csv',history)
        score=None
        if (episode+1)%validation_every==0:
            score=mean_validation(fly,off,split,max_steps);v={'episode':episode+1,**score};validations.append(v);write_csv(root/'validation_history.csv',validations)
            candidate=(score['coverage_auc'],score['final_coverage'],-score['repeated_visit_ratio'],-score['total_uav_energy_consumed_j'])
            if best is None or candidate>tuple(best):best=candidate;best_episode=episode+1
            else:score=None
        checkpoint={'episode':episode+1,'global_step':global_step,'fly':agent_state(fly),'off':agent_state(off),
                    'cnn_replay':cnn_buffer.state_dict(),'gcn_replay':graph_buffer.state_dict(),'rng':rng_state(),
                    'best':best,'best_episode':best_episode,'history':history,'validations':validations,
                    'manifest_prefix_sha256':manifest_hash(schedules[:episode+1]),'training_manifest_sha256':manifest_hash(schedules),'code_fingerprint':code_fingerprint,'config':config}
        atomic_save(last,checkpoint)
        if score is not None:
            atomic_save(root/'best_checkpoint.pt',checkpoint);atomic_save(root/'best_fly_actor.pt',fly.actor.state_dict());atomic_save(root/'best_offload_actor.pt',off.actor.state_dict())
        (root/'events.jsonl').open('a').write(json.dumps({'episode':episode+1,'step':global_step,'validation':score is not None})+'\n')
        (root/'failed_updates.jsonl').touch(exist_ok=True)
        print(f'{mode.value}/{seed} episode {episode+1}/{episodes} steps={global_step} replay={len(cnn_buffer)}/{len(graph_buffer)} seconds={row["seconds"]:.1f}',flush=True)
    initial=json.loads((root/'initial_model_fingerprint.json').read_text())
    final={'fly_actor_sha256':module_hash(fly.actor),'offload_actor_sha256':module_hash(off.actor),
           'fly_critic_1_sha256':module_hash(fly.critic_1),'offload_critic_1_sha256':module_hash(off.critic_1),
           'fly_critic_1_changed':module_hash(fly.critic_1)!=initial['fly_critic_1_sha256'],
           'offload_critic_1_changed':module_hash(off.critic_1)!=initial['offload_critic_1_sha256'],
           'fly_actor_changed':module_hash(fly.actor)!=initial['fly_actor_sha256'],
           'offload_actor_changed':module_hash(off.actor)!=initial['offload_actor_sha256']}
    (root/'final_model_fingerprint.json').write_text(json.dumps(final,indent=2)+'\n')
    manifest={'status':'completed','episodes':episodes,'global_step':global_step,'best_episode':best_episode,
              'best_validation':best,'elapsed_seconds_this_invocation':time.monotonic()-started,'device':device,
              'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'manifest_sha256':manifest_hash(schedules)}
    (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
