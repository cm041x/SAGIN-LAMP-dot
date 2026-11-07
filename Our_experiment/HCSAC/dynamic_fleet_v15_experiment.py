"""Shared V1.5 paired diagnostic runner; all episodes use fixed pretrained actors."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; HCSAC=ROOT/'Our_experiment/HCSAC'; os.chdir(HCSAC)
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import Status
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleetV15 import DynamicFleetV15Config, DynamicFleetV15Env
from Our_experiment.HCSAC.ENV.handover_policy import HandoverMode
from Our_experiment.HCSAC.ENV.fleet_scheduler import SchedulerMode
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import PolicyNet_CNN, PolicyNet_GCN
from Our_experiment.HCSAC.dynamic_fleet_v1_experiment import fleet_deficit, fleet_surplus, recovery_delays
from Our_experiment.HCSAC.dynamic_fleet_v15_metrics import summarize_v15
from Our_experiment.HCSAC.fleet_metrics import request_accounting

WIND={11:'Low Wind',23:'Moderate Wind',4800:'Strong Wind'}
GA_SOURCE='Our_experiment/GA/data/ga_best_deployment_random10_w23_g10_i999999_pool0_200.json'
FILES=['Our_experiment/HCSAC/ENV/UAVenv_SAC_Original.py','Our_experiment/HCSAC/ENV/UAVenv_DynamicFleet.py',
 'Our_experiment/HCSAC/ENV/UAVenv_DynamicFleetV1.py','Our_experiment/HCSAC/ENV/fleet_scheduler.py',
 'Our_experiment/HCSAC/ENV/UAVenv_DynamicFleetV15.py','Our_experiment/HCSAC/ENV/handover_policy.py',
 'Our_experiment/HCSAC/dynamic_fleet_v15_metrics.py','Our_experiment/HCSAC/dynamic_fleet_v15_experiment.py',
 'Our_experiment/HCSAC/wind.py','Our_experiment/HCSAC/wind.json',GA_SOURCE,
 'Our_experiment/HCSAC/data/sac_model_fly.pt','Our_experiment/HCSAC/data/sac_model_offload.pt']

def fingerprint() -> str:
    return hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in FILES)).hexdigest()

def load_positions() -> list[tuple[int,int]]:
    data=json.loads((ROOT/GA_SOURCE).read_text()); positions=[tuple(map(int,p)) for p in data['best']['deployment_positions']]
    if len(positions)!=4: raise ValueError('Expected four pre-existing GA deployment positions')
    return positions+[(10,10)]*2

def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value,allow_nan=False,ensure_ascii=False)+'\n')

def run_rollout(spec: dict) -> dict:
    torch.set_num_threads(spec['threads']); seed=spec['eval_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mode=spec['method']; scheduler=SchedulerMode.NAIVE_THRESHOLD if mode=='naive_threshold' else SchedulerMode.TARGET_AWARE
    handover=HandoverMode.UNCERTAINTY_WAYPOINT if mode=='target_aware_handover' else HandoverMode.NONE
    config=DynamicFleetV15Config(scheduler_mode=scheduler,handover_mode=handover,target_search_uavs=spec['target'],
        charging_capacity=spec['capacity'],charging_power_w=spec['power'],charging_efficiency=spec['efficiency'],
        max_episode_steps=spec['max_steps'],seed=seed)
    env=DynamicFleetV15Env(config); state=env.reset(seed=seed,wind_seed=spec['wind_seed'],terrain_seed=seed,infra_seed=seed,positions=load_positions())
    initial_uncertainty=float(env.uncertainty_matrix.sum()); initial_positions=[list(u['position']) for u in env.uavs]
    cnn=PolicyNet_CNN(15,128,8).eval(); gcn=PolicyNet_GCN(6,128,5).eval()
    cnn.load_state_dict(torch.load(HCSAC/'data/sac_model_fly.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
    gcn.load_state_dict(torch.load(HCSAC/'data/sac_model_offload.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
    counts=[]; coverage=[]; queues=[]; fleet=[]; occupied_steps=0; idle=0; overload=0
    for _ in range(spec['max_steps']):
        active=np.flatnonzero(env.get_search_active_mask()).tolist(); actions=[0]*env.N
        if active:
            with torch.no_grad(): chosen=cnn(torch.as_tensor(state[active],dtype=torch.float32)).argmax(1).tolist()
            for i,a in zip(active,chosen): actions[i]=a
        charging=sum(u['status']==Status.CHARGING for u in env.uavs)
        occupied_steps+=charging; idle+=int(charging<config.charging_capacity); overload+=int(charging>config.charging_capacity)
        state,_,done=env.step(actions)
        if not np.isfinite(state).all(): raise ValueError('Nonfinite CNN state')
        graph=env.get_obs_2()
        if graph.slot_id[:env.N].tolist()!=list(range(env.N)): raise ValueError('Graph slot mismatch')
        for i,u in enumerate(env.uavs):
            if u['status']!=Status.ACTIVE and ((graph.edge_index==i).any() or graph.x[i].any()): raise ValueError('Inactive graph not isolated')
        offloads=[0]*env.N
        if env.eligible:
            with torch.no_grad(): offloads=gcn(graph).argmax(1).tolist()
        before=[u['energy'] for u in env.uavs]; _,r,_=env.step_offload(offloads)
        if any(r[i]!=0 or env.uavs[i]['energy']!=before[i] for i in set(range(env.N))-env.eligible): raise ValueError('Inactive UAV offloaded')
        if any(not np.isfinite(u['energy']) or not 0<=u['energy']<=env.initial_energy for u in env.uavs): raise ValueError('Invalid energy')
        count=env.get_fleet_counts(); counts.append(count['search_active_count']); queues.append(len(env.queue)); coverage.append(1-float(env.uncertainty_matrix.mean()))
        fleet.append({'step':env.step_number,**count,'queue_length':len(env.queue),'coverage':coverage[-1]})
        if done: break
    events=env.get_event_log(); types=[e['event_type'] for e in events]; n=len(counts)
    if any(t=='energy_exhausted_before_return' for t in types): raise ValueError('Energy exhausted before return')
    if mode!='target_aware_handover' and any(t.startswith('handover_') or t=='responsibility_region_created' for t in types): raise ValueError('Handover polluted baseline')
    handover_assign=sum(t=='handover_target_assigned' for t in types); handover_release=sum(t=='handover_released' for t in types)
    if handover_assign!=handover_release: raise ValueError('Unclosed handover')
    stats=summarize_v15(env,counts,initial_uncertainty)
    recoveries,censored=recovery_delays(counts,config.target_search_uavs)
    requests={}; delays=[]
    for e in events:
        if e['event_type']=='return_request_created': requests[e['uav_id']]=e['step']
        if e['event_type']=='replacement_assigned' and e['uav_id'] in requests: delays.append(e['step']-requests[e['uav_id']])
    energy=env.energy_stats
    total=energy['search_flight_energy_j']+energy['return_flight_energy_j']+energy['offload_energy_j']
    mean=lambda x: float(np.mean(x)) if x else None
    metrics={'run_id':spec['run_id'],'experiment_type':spec['stage'],'method':mode,'scheduler_mode':scheduler.value,
        'handover_mode':handover.value,'target_search_uavs':config.target_search_uavs,'charging_capacity':config.charging_capacity,
        'charging_power_w':config.charging_power_w,'charging_efficiency':config.charging_efficiency,
        'wind_seed':spec['wind_seed'],'wind_class':WIND.get(spec['wind_seed'],'Other'),'eval_seed':seed,
        'terrain_seed':seed,'infra_seed':seed,'trajectory_seed':seed,'code_fingerprint':spec['fingerprint'],
        'initial_positions':initial_positions,'ga_source':GA_SOURCE,'steps':n,'initial_energy_j':env.initial_energy,
        'step_duration_seconds':env.T,'charged_energy_per_step_j':config.charging_power_w*config.charging_efficiency*env.T,
        'final_average_uncertainty':float(env.uncertainty_matrix.mean()),'final_coverage':coverage[-1],
        'coverage_auc':float(np.mean(coverage)),'mean_search_active_count':float(np.mean(counts)),
        'min_search_active_count':min(counts),'max_search_active_count':max(counts),
        'target_met_steps':sum(x>=config.target_search_uavs for x in counts),
        'target_met_ratio':sum(x>=config.target_search_uavs for x in counts)/n,
        'below_target_steps':sum(x<config.target_search_uavs for x in counts),
        'above_target_steps':sum(x>config.target_search_uavs for x in counts),
        'integrated_fleet_deficit':fleet_deficit(counts,config.target_search_uavs),
        'integrated_fleet_surplus':fleet_surplus(counts,config.target_search_uavs),
        'total_launch_events':types.count('reserve_launched'),'total_return_requests':types.count('return_request_created'),
        'total_return_events':types.count('active_to_returning'),'total_charging_start_events':types.count('charging_started'),
        'total_charging_complete_events':types.count('charging_completed'), 'mean_replacement_delay_steps':mean(delays),
        'mean_target_recovery_delay_steps':mean(recoveries),'target_recovery_right_censored':censored,
        'mean_charging_queue_length':mean(queues),'max_charging_queue_length':max(queues),
        'mean_waiting_charge_steps':mean(env.waiting_charge_steps),'waiting_charge_observed':len(env.waiting_charge_steps),
        'charger_utilization_ratio':occupied_steps/(n*config.charging_capacity),'charging_idle_steps':idle,'charging_overload_steps':overload,
        **energy,'total_uav_energy_consumed_j':total,**request_accounting(events),**stats}
    return {'summary':metrics,'events':events,'trajectory':env.step_trajectory,'fleet_timeline':fleet,
            'offload_timeline':env.offload_timeline,'handover_timeline':env.handover_timeline,'visit_maps':env.visit_maps}

def execute(spec: dict) -> dict:
    try:
        run=run_rollout(spec); root=Path(spec['output_dir']); final=root/'runs'/spec['run_id']
        temp=root/'runs'/(spec['run_id']+'.incomplete'); temp.mkdir(parents=True,exist_ok=True)
        for name in ('summary','trajectory','fleet_timeline','offload_timeline','handover_timeline'):
            json_write(temp/(name+'.json'),run[name])
        with (temp/'events.jsonl').open('w') as f:
            for e in run['events']: f.write(json.dumps(e,allow_nan=False,ensure_ascii=False)+'\n')
        np.savez_compressed(temp/'visit_maps.npz',visits=run['visit_maps'])
        os.rename(temp,final)
        return {'ok':True,'run_id':spec['run_id']}
    except Exception:
        return {'ok':False,'run_id':spec['run_id'],'traceback':traceback.format_exc(),'spec':spec}

def parse(stage: str) -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument('--wind-seeds',default='11,23,4800'); p.add_argument('--eval-seeds',default=','.join(map(str,range(42,52))))
    p.add_argument('--max-steps',type=int,default=500); p.add_argument('--charging-power-w',type=float,default=600)
    p.add_argument('--charging-efficiency',type=float,default=.9); p.add_argument('--output-dir',required=True)
    p.add_argument('--resume',action='store_true'); p.add_argument('--workers',type=int,default=1)
    if stage=='capacity':
        p.add_argument('--target-search-uavs',default='3,4'); p.add_argument('--charging-capacities',default='1,2')
    else:
        p.add_argument('--methods',default='naive_threshold,target_aware,target_aware_handover')
        p.add_argument('--target-search-uavs',type=int,default=4); p.add_argument('--charging-capacity',type=int,default=1)
    a=p.parse_args()
    if not 1<=a.workers<=4: p.error('workers must be 1..4')
    return a

def main(stage: str) -> None:
    a=parse(stage); winds=list(map(int,a.wind_seeds.split(','))); seeds=list(map(int,a.eval_seeds.split(',')))
    if stage=='capacity': configs=[('target_aware',t,c) for t in map(int,a.target_search_uavs.split(',')) for c in map(int,a.charging_capacities.split(','))]
    else: configs=[(m,a.target_search_uavs,a.charging_capacity) for m in a.methods.split(',')]
    if any(m not in ('naive_threshold','target_aware','target_aware_handover') for m,_,_ in configs): raise ValueError('Unknown method')
    if len(set(configs))!=len(configs) or len(set(winds))!=len(winds) or len(set(seeds))!=len(seeds): raise ValueError('Duplicate configuration')
    out=Path(a.output_dir); out=out if out.is_absolute() else ROOT/out; (out/'runs').mkdir(parents=True,exist_ok=True)
    fp=fingerprint(); settings={'stage':stage,'configs':configs,'wind_seeds':winds,'eval_seeds':seeds,'max_steps':a.max_steps,
        'charging_power_w':a.charging_power_w,'charging_efficiency':a.charging_efficiency,'code_fingerprint':fp,
        'ga_source':GA_SOURCE,'initial_positions':[list(p) for p in load_positions()],
        'controller':'pretrained_argmax','initial_energy_j':360000,'minimum_launch_energy_ratio':0.95,
        'seed_mapping':'wind=wind seed; terrain=infra=trajectory=eval seed','threads_per_worker':max(1,4//a.workers)}
    config_path=out/'experiment_config.json'
    if config_path.exists():
        old=json.loads(config_path.read_text()); canonical=json.loads(json.dumps(settings))
        if old!=canonical: raise SystemExit('Configuration changed: choose a new output directory')
        if not a.resume: raise SystemExit('Output exists; use --resume')
    else: json_write(config_path,settings)
    specs=[]
    for w in winds:
        for seed in seeds:
            for method,target,capacity in configs:
                rid=f'{stage}_{method}_t{target}_c{capacity}_p{a.charging_power_w:g}_w{w}_s{seed}_{fp[:12]}'
                if a.resume and (out/'runs'/rid/'summary.json').is_file(): continue
                specs.append({'stage':stage,'method':method,'target':target,'capacity':capacity,'power':a.charging_power_w,
                    'efficiency':a.charging_efficiency,'wind_seed':w,'eval_seed':seed,'max_steps':a.max_steps,
                    'fingerprint':fp,'run_id':rid,'output_dir':str(out),'threads':max(1,4//a.workers)})
    print(f'{stage}: pending {len(specs)} runs, {a.workers} workers',flush=True)
    failed=out/'failed_runs.jsonl'; failed.touch(exist_ok=True)
    def save(result):
        if result['ok']: print('completed',result['run_id'],flush=True)
        else:
            with failed.open('a') as f: f.write(json.dumps(result,ensure_ascii=False)+'\n')
            print('FAILED',result['run_id'],result['traceback'],flush=True)
    if a.workers==1:
        for spec in specs: save(execute(spec))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in as_completed([pool.submit(execute,s) for s in specs]): save(f.result())
    summaries=[]
    for path in sorted((out/'runs').glob('*/summary.json')):
        if any(part.startswith('._') for part in path.parts): continue
        summaries.append(json.loads(path.read_text()))
    with (out/'raw_results.jsonl').open('w') as f:
        for x in summaries: f.write(json.dumps(x,allow_nan=False)+'\n')
    if summaries:
        with (out/'per_run_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in m.items()} for m in summaries)
    if stage=='handover':
        with (out/'handover_events.jsonl').open('w') as dst:
            for run in summaries:
                for line in (out/'runs'/run['run_id']/'events.jsonl').read_text().splitlines():
                    e=json.loads(line)
                    if e['event_type'].startswith('handover_') or e['event_type'] in ('responsibility_region_created','fallback_global_target','no_valid_handover_target'):
                        dst.write(json.dumps({'run_id':run['run_id'],**e})+'\n')
    expected=len(configs)*len(winds)*len(seeds); failures=sum(1 for _ in failed.open())
    json_write(out/'run_manifest.json',{'expected_runs':expected,'completed_runs':len(summaries),'failed_attempts':failures,
        'remaining_runs':expected-len(summaries),'workers':a.workers,'config':settings})
    if len(summaries)!=expected: raise SystemExit(f'Incomplete {len(summaries)}/{expected}; resume failed runs')
