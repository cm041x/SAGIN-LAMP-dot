"""Bounded paired V1 rollouts; writes isolated per-run files for safe resume."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HCSAC=ROOT/'Our_experiment/HCSAC'
os.chdir(HCSAC)
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleetV1 import DynamicFleetV1Config, DynamicFleetV1Env
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import Status
from Our_experiment.HCSAC.ENV.fleet_scheduler import SchedulerMode
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import PolicyNet_CNN, PolicyNet_GCN

WIND_CLASSES={11:'Low Wind',23:'Moderate Wind',4800:'Strong Wind'}

def fleet_deficit(counts: list[int], target: int) -> int:
    return sum(max(0,target-n) for n in counts)

def fleet_surplus(counts: list[int], target: int) -> int:
    return sum(max(0,n-target) for n in counts)

def sample_seeds(wind_seed: int, eval_seed: int) -> dict:
    return {'wind_seed':wind_seed,'terrain_seed':eval_seed,'infra_seed':eval_seed,'trajectory_seed':eval_seed}

def recovery_delays(counts: list[int], target: int) -> tuple[list[int],int]:
    start=None; delays=[]
    for i,n in enumerate(counts):
        if n<target and start is None: start=i
        elif n>=target and start is not None:
            delays.append(i-start); start=None
    return delays,int(start is not None)

def average(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None

def run_rollout(spec: dict) -> dict:
    torch.set_num_threads(spec['threads'])
    seed=spec['eval_seed']; seeds=sample_seeds(spec['wind_seed'],seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    c=DynamicFleetV1Config(scheduler_mode=SchedulerMode(spec['scheduler_mode']),target_search_uavs=spec['target_search_uavs'],
        max_episode_steps=spec['max_steps'],charging_power_w=spec['charging_power_w'],charging_capacity=spec['charging_capacity'],
        charging_efficiency=spec['charging_efficiency'],minimum_launch_energy_ratio=spec['minimum_launch_energy_ratio'],seed=seed)
    env=DynamicFleetV1Env(c)
    state=env.reset(seed=seed, wind_seed=seeds['wind_seed'],terrain_seed=seeds['terrain_seed'],infra_seed=seeds['infra_seed'])
    cnn=PolicyNet_CNN(15,128,8).eval(); gcn=PolicyNet_GCN(6,128,5).eval()
    cnn.load_state_dict(torch.load(HCSAC/'data/sac_model_fly.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
    gcn.load_state_dict(torch.load(HCSAC/'data/sac_model_offload.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
    initial_uncertainty=float(env.uncertainty_matrix.sum()); search_counts=[]; coverages=[]; queues=[]; trajectories=[]
    search_visits=0; charging_occupied=0; charging_idle=0; charging_overload=0; inactive_offload=0
    for _ in range(spec['max_steps']):
        slots=np.flatnonzero(env.get_search_active_mask()).tolist(); actions=[0]*env.N
        if slots:
            with torch.no_grad(): selected=cnn(torch.as_tensor(state[slots],dtype=torch.float32)).argmax(1).tolist()
            for i,a in zip(slots,selected): actions[i]=a
        occupied=sum(u['status']==Status.CHARGING for u in env.uavs)
        charging_occupied+=occupied; charging_idle+=int(occupied<c.charging_capacity); charging_overload+=int(occupied>c.charging_capacity)
        state,_,done=env.step(actions)
        if not np.isfinite(state).all(): raise ValueError('nonfinite flight state')
        graph=env.get_obs_2()
        if graph.slot_id[:env.N].tolist()!=list(range(env.N)): raise ValueError('slot mapping mismatch')
        for i,u in enumerate(env.uavs):
            if u['status']!=Status.ACTIVE and ((graph.edge_index==i).any() or graph.x[i].any()):
                raise ValueError('inactive offload graph edge or feature')
        offloads=[0]*env.N
        if env.eligible:
            with torch.no_grad(): offloads=gcn(graph).argmax(1).tolist()
        before=[u['energy'] for u in env.uavs]
        _,r,_=env.step_offload(offloads)
        for i in set(range(env.N))-env.eligible:
            if r[i]!=0 or env.uavs[i]['energy']!=before[i]: inactive_offload+=1
        if inactive_offload: raise ValueError('inactive UAV offloaded')
        if not all(np.isfinite(u['energy']) and 0<=u['energy']<=env.initial_energy for u in env.uavs):
            raise ValueError('invalid UAV energy')
        if any(e['event_type']=='energy_exhausted_before_return' for e in env.events[-15:]):
            raise ValueError('energy exhausted before return; event retained in failed run')
        search_visits+=len(env.eligible)
        count=env.get_fleet_counts()['search_active_count']; search_counts.append(count)
        coverages.append(1-float(env.uncertainty_matrix.mean())); queues.append(len(env.queue))
        trajectories.append({'step':env.step_number,'search_active_count':count,'queue_length':len(env.queue),
                             'coverage':coverages[-1], 'charging_count':env.get_fleet_counts()['charging_count']})
        if done: break
    events=env.get_event_log(); types=[e['event_type'] for e in events]; steps=len(search_counts)
    requests={}; binding={}; delays=[]; duplicates=0
    for ev in events:
        typ=ev['event_type']; uid=ev['uav_id']
        if typ=='return_request_created': requests[uid]=ev['step']
        elif typ=='replacement_assigned':
            rid=ev['metadata']['replacement_uav_id']
            if uid in binding or rid in binding.values(): duplicates+=1
            binding[uid]=rid
            if uid in requests: delays.append(ev['step']-requests[uid])
        elif typ=='replacement_binding_released': binding.pop(uid,None)
    recoveries,censored=recovery_delays(search_counts,c.target_search_uavs)
    newly=int(initial_uncertainty-env.uncertainty_matrix.sum())
    total=env.energy_stats['search_flight_energy_j']+env.energy_stats['return_flight_energy_j']+env.energy_stats['offload_energy_j']
    result={'run_id':spec['run_id'],'scheduler_mode':spec['scheduler_mode'],'wind_seed':spec['wind_seed'],
        'wind_class':WIND_CLASSES.get(spec['wind_seed'],'Other'),'eval_seed':seed,**seeds,'steps':steps,
        'mean_wind_speed_m_s':float(np.sqrt(env.wind_u**2+env.wind_v**2).mean()),
        'final_average_uncertainty':float(env.uncertainty_matrix.mean()),'final_coverage':coverages[-1],
        'coverage_auc':float(np.mean(coverages)),'newly_covered_cells':newly,'repeated_visit_count':search_visits-newly,
        'repeated_visit_ratio':(search_visits-newly)/search_visits if search_visits else 0.0,
        'mean_search_active_count':average(search_counts),'min_search_active_count':min(search_counts),'max_search_active_count':max(search_counts),
        'target_met_steps':sum(n>=c.target_search_uavs for n in search_counts),
        'target_met_ratio':sum(n>=c.target_search_uavs for n in search_counts)/steps,
        'below_target_steps':sum(n<c.target_search_uavs for n in search_counts),'above_target_steps':sum(n>c.target_search_uavs for n in search_counts),
        'integrated_fleet_deficit':fleet_deficit(search_counts,c.target_search_uavs),'integrated_fleet_surplus':fleet_surplus(search_counts,c.target_search_uavs),
        'total_return_requests':types.count('return_request_created'),'total_launch_events':types.count('reserve_launched'),
        'total_return_events':types.count('active_to_returning'),'total_charging_start_events':types.count('charging_started'),
        'total_charging_complete_events':types.count('charging_completed'),
        'total_reactivation_events':sum(e['event_type']=='reserve_launched' and e['metadata']['activation_count']>1 for e in events),
        'unresolved_return_requests':sum(env.uavs[i]['status']==Status.ACTIVE for i in env.requests),
        'duplicate_replacement_attempts':duplicates,'mean_replacement_delay_steps':average(delays),
        'max_replacement_delay_steps':max(delays) if delays else None,'replacement_delays_observed':len(delays),
        'mean_target_recovery_delay_steps':average(recoveries),'max_target_recovery_delay_steps':max(recoveries) if recoveries else None,
        'target_recovery_delays_observed':len(recoveries),'target_recovery_right_censored':censored,
        'total_grid_energy_input_j':env.energy_stats['total_grid_energy_input_j'],
        'mean_charging_queue_length':average(queues),'max_charging_queue_length':max(queues),
        'mean_waiting_charge_steps':average(env.waiting_charge_steps),'waiting_charge_observed':len(env.waiting_charge_steps),
        'charger_utilization_ratio':charging_occupied/(steps*c.charging_capacity),'charging_idle_steps':charging_idle,
        'charging_overload_steps':charging_overload,'search_flight_energy_j':env.energy_stats['search_flight_energy_j'],
        'return_flight_energy_j':env.energy_stats['return_flight_energy_j'],'local_compute_energy_j':env.energy_stats['local_compute_energy_j'],'communication_energy_j':env.energy_stats['communication_energy_j'],
        'offload_energy_j':env.energy_stats['offload_energy_j'],'total_uav_energy_consumed_j':total,
        'active_offload_decisions':sum(env.offload_target_counts),'offload_target_counts':env.offload_target_counts,
        'effective_offload_target_counts':env.effective_offload_counts, 'local_processing_count':env.effective_offload_counts[0], 'local_requested_count':env.offload_target_counts[0], 'inactive_uav_offload_count':inactive_offload,
        'charging_power_w':c.charging_power_w,'charging_efficiency':c.charging_efficiency,'step_duration_seconds':env.T,
        'charged_energy_per_step_j':c.charging_power_w*c.charging_efficiency*env.T,
        'initial_energy_j':env.initial_energy,'target_search_uavs':c.target_search_uavs}
    assert not duplicates, 'duplicate binding'
    assert spec['scheduler_mode']!='no_replacement' or result['total_launch_events']==0
    return {'metrics':result,'events':events,'trajectory':trajectories}

def atomic_json(path: Path, obj: dict) -> None:
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,allow_nan=False,ensure_ascii=False)+'\n'); os.replace(tmp,path)

def run_one(spec: dict) -> dict:
    try: return {'ok':True, 'run':run_rollout(spec)}
    except Exception: return {'ok':False,'run_id':spec['run_id'],'traceback':traceback.format_exc()}

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--methods',default='no_replacement,naive_threshold,target_aware')
    p.add_argument('--wind-seeds',default='11,23,4800'); p.add_argument('--eval-seeds',default=','.join(map(str,range(42,52))))
    p.add_argument('--max-steps',type=int,default=500); p.add_argument('--charging-power-w',type=float,default=600)
    p.add_argument('--charging-capacity',type=int,default=1); p.add_argument('--target-search-uavs',type=int,default=4)
    p.add_argument('--output-dir',default='Our_experiment/HCSAC/data/dynamic_fleet_v1'); p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=1); args=p.parse_args()
    if not 1<=args.workers<=4: p.error('--workers must be in 1..4')
    methods=[SchedulerMode(x).value for x in args.methods.split(',')]; winds=list(map(int,args.wind_seeds.split(','))); seeds=list(map(int,args.eval_seeds.split(',')))
    if len(set(methods))!=len(methods) or len(set(winds))!=len(winds) or len(set(seeds))!=len(seeds): p.error('Duplicate methods or seeds')
    out=Path(args.output_dir); out=out if out.is_absolute() else ROOT/out; (out/'runs').mkdir(parents=True,exist_ok=True)
    config={'methods':methods,'wind_seeds':winds,'eval_seeds':seeds,'max_steps':args.max_steps,'charging_power_w':args.charging_power_w,
        'charging_capacity':args.charging_capacity,'charging_efficiency':0.9,'minimum_launch_energy_ratio':0.95,
        'target_search_uavs':args.target_search_uavs,'controller':'pretrained_argmax', 'initial_energy_j':360000,
        'seed_mapping':'wind=wind_seed; terrain=infra=trajectory=eval_seed', 'no_smoke_overrides':True}
    files=['Our_experiment/HCSAC/ENV/UAVenv_DynamicFleet.py','Our_experiment/HCSAC/ENV/UAVenv_DynamicFleetV1.py',
           'Our_experiment/HCSAC/ENV/fleet_scheduler.py','Our_experiment/HCSAC/dynamic_fleet_v1_experiment.py',
           'Our_experiment/HCSAC/data/sac_model_fly.pt','Our_experiment/HCSAC/data/sac_model_offload.pt']
    config['code_weights_sha256']=hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in files)).hexdigest()
    cfg=out/'experiment_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config: raise SystemExit('Output directory belongs to a different configuration; choose another')
    if cfg.exists() and not args.resume: raise SystemExit('Output exists; use --resume or a new directory')
    atomic_json(cfg,config)
    specs=[]
    for wind in winds:
        for seed in seeds:
            for mode in methods:
                rid=f'{mode}_w{wind}_s{seed}'
                if args.resume and (out/'runs'/f'{rid}.json').exists(): continue
                specs.append({'run_id':rid,'scheduler_mode':mode,'wind_seed':wind,'eval_seed':seed,
                    'max_steps':args.max_steps,'charging_power_w':args.charging_power_w,'charging_capacity':args.charging_capacity,
                    'charging_efficiency':0.9,'minimum_launch_energy_ratio':0.95,'target_search_uavs':args.target_search_uavs,
                    'threads':max(1,4//args.workers)})
    print(f'Pending {len(specs)} runs in {out}',flush=True)
    def save(result: dict) -> None:
        if result['ok']:
            run=result['run']; rid=run['metrics']['run_id']; atomic_json(out/'runs'/f'{rid}.json',run)
            print('completed',rid,flush=True)
        else:
            with (out/'failed_runs.jsonl').open('a') as f: f.write(json.dumps(result,ensure_ascii=False)+'\n')
            print('FAILED',result['run_id'],result['traceback'],flush=True)
    if args.workers==1:
        for spec in specs: save(run_one(spec))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_one,s) for s in specs]): save(future.result())
    rows=[]
    for path in sorted((out/'runs').glob('*.json')):
        if path.name.startswith('._'): continue
        rows.append(json.loads(path.read_text()))
    with (out/'raw_results.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row['metrics'],allow_nan=False)+'\n')
    if rows:
        with (out/'per_run_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]['metrics']))
            writer.writeheader(); writer.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in row['metrics'].items()} for row in rows)
    (out/'failed_runs.jsonl').touch(exist_ok=True)
    expected=len(methods)*len(winds)*len(seeds)
    atomic_json(out/'run_manifest.json',{'expected_runs':expected,'completed_runs':len(rows),'remaining_runs':expected-len(rows),
        'workers':args.workers,'failed_attempts':sum(1 for _ in (out/'failed_runs.jsonl').open()),'config':config})
    if len(rows)!=expected: raise SystemExit(f'Incomplete: {len(rows)}/{expected}; rerun with --resume')

if __name__=='__main__': main()
