"""Evaluate only saved validation-selected actors on fixed environment combinations."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import ROOT,HCSAC,TrainingMode,initialize_agents,new_env,collect_step
from Our_experiment.HCSAC.training.checkpointing import file_hash
from Our_experiment.HCSAC.dynamic_fleet_v15_metrics import summarize_v15
from Our_experiment.HCSAC.dynamic_fleet_v1_experiment import fleet_deficit,fleet_surplus

CLASSES={11:'Low Wind',23:'Moderate Wind',4800:'Strong Wind'}

def evaluation_rollout(fly,off,wind: int,seed: int,max_steps: int) -> tuple[dict,list]:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=new_env(max_steps,{'wind_seed':wind,'terrain_seed':seed,'infra_seed':seed});state=env._get_obs()
    initial=float(env.uncertainty_matrix.sum());counts=[];coverage=[]
    for t in range(max_steps):
        state,_,_,done,_,_=collect_step(env,state,fly,off,None,None,0,t,greedy=True,store=False)
        counts.append(env.get_fleet_counts()['search_active_count']);coverage.append(1-float(env.uncertainty_matrix.mean()))
        if done:break
    stats=summarize_v15(env,counts,initial);energy=env.energy_stats;types=[e['event_type'] for e in env.events]
    metrics={**stats,'final_average_uncertainty':float(env.uncertainty_matrix.mean()),'final_coverage':coverage[-1],
        'coverage_auc':float(np.mean(coverage)),'mean_search_active_count':float(np.mean(counts)),
        'min_search_active_count':min(counts),'max_search_active_count':max(counts),
        'integrated_fleet_deficit':fleet_deficit(counts,4),'integrated_fleet_surplus':fleet_surplus(counts,4),
        'target_met_ratio':sum(c>=4 for c in counts)/len(counts),'below_target_steps':sum(c<4 for c in counts),
        'total_launch_events':types.count('reserve_launched'),'total_return_events':types.count('active_to_returning'),
        'total_charging_complete_events':types.count('charging_completed'),**energy,
        'total_uav_energy_consumed_j':energy['search_flight_energy_j']+energy['return_flight_energy_j']+energy['offload_energy_j'],
        'sequential_blocked_move_attempts':env.conflict_attempts,'steps':len(counts)}
    return metrics,env.step_trajectory

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--training-dir',required=True);p.add_argument('--scratch-dir');p.add_argument('--finetune-dir')
    p.add_argument('--include-zero-shot',action='store_true');p.add_argument('--modes',default='scratch,finetune');p.add_argument('--train-seeds',default='3001,3002,3003')
    p.add_argument('--wind-seeds',default='11,23,4800');p.add_argument('--eval-seeds',default=','.join(map(str,range(42,52))))
    p.add_argument('--max-steps',type=int,default=500);p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.workers!=1:p.error('Only workers=1 is currently supported; no unverified parallel evaluation')
    output=Path(a.output_dir);output=output if output.is_absolute() else ROOT/output;(output/'runs').mkdir(parents=True,exist_ok=True)
    train=Path(a.training_dir);train=train if train.is_absolute() else ROOT/train
    def root_path(x):
        p=Path(x);return p if p.is_absolute() else ROOT/p
    models=[]
    if a.include_zero_shot:models.append(('original_zero_shot',None,None,None))
    for mode in map(TrainingMode,a.modes.split(',')):
        base=root_path(a.scratch_dir) if mode==TrainingMode.SCRATCH and a.scratch_dir else root_path(a.finetune_dir) if mode==TrainingMode.FINETUNE and a.finetune_dir else train
        for seed in map(int,a.train_seeds.split(',')):
            folder=base/mode.value/f'seed_{seed}'
            if not (folder/'best_fly_actor.pt').exists():raise FileNotFoundError(f'Missing validation-selected actor: {folder}')
            models.append((f'{mode.value}_seed_{seed}',mode,seed,folder))
    winds=list(map(int,a.wind_seeds.split(',')));seeds=list(map(int,a.eval_seeds.split(',')))
    config={'models':[x[0] for x in models],'wind_seeds':winds,'eval_seeds':seeds,'max_steps':a.max_steps,
            'environment':'V1.5 target-aware, no handover, target 4, capacity 1, 600 W, 0.90 efficiency, GA deployment',
            'model_sha256':{name:[file_hash((folder/'best_fly_actor.pt') if folder else HCSAC/'data/sac_model_fly.pt'),file_hash((folder/'best_offload_actor.pt') if folder else HCSAC/'data/sac_model_offload.pt')] for name,_,_,folder in models}}
    environment={'class':'DynamicFleetV15Env','scheduler':'target_aware','handover':'none','target':4,'capacity':1,
                 'power_w':600,'efficiency':.9,'minimum_launch_energy_ratio':.95,'initial_energy_j':360000,
                 'max_steps':a.max_steps,'initial_deployment':'GA w23 frozen'}
    signature=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()
    code_files=sorted((HCSAC/'training').glob('*.py'))+[HCSAC/'evaluate_dynamic_hcsac.py',
        HCSAC/'ENV/UAVenv_DynamicFleetV15.py',HCSAC/'ENV/UAVenv_DynamicFleetV1.py',HCSAC/'ENV/UAVenv_DynamicFleet.py',
        HCSAC/'ENV/UAVenv_SAC_Original.py',HCSAC/'ENV/fleet_scheduler.py']
    code_fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in code_files)).hexdigest()
    config['environment_config_hash']=signature;config['code_fingerprint']=code_fingerprint
    cfg=output/'experiment_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Evaluation configuration changed')
    if cfg.exists() and not a.resume:raise ValueError('Output exists; use --resume')
    cfg.write_text(json.dumps(config,indent=2)+'\n');failures=output/'failed_runs.jsonl';failures.touch(exist_ok=True)
    for name,mode,train_seed,folder in models:
        fly,off=initialize_agents(TrainingMode.FINETUNE if mode is None else TrainingMode.SCRATCH,'cpu');torch.set_num_threads(4)
        if folder:
            fly.actor.load_state_dict(torch.load(folder/'best_fly_actor.pt',map_location='cpu',weights_only=True),strict=True)
            off.actor.load_state_dict(torch.load(folder/'best_offload_actor.pt',map_location='cpu',weights_only=True),strict=True)
            manifest=json.loads((folder/'run_manifest.json').read_text());best_episode=manifest['best_episode'];val=manifest['best_validation'][0]
        else:best_episode=None;val=None
        for wind in winds:
            for seed in seeds:
                rid=f'{name}_w{wind}_s{seed}';dest=output/'runs'/f'{rid}.json'
                if a.resume and dest.exists():continue
                try:
                    metrics,trajectory=evaluation_rollout(fly,off,wind,seed,a.max_steps)
                    row={'run_id':rid,'policy_name':name,'training_mode':mode.value if mode else 'zero_shot','training_seed':train_seed,
                         'checkpoint_path':str(folder) if folder else 'original checkpoints','checkpoint_sha256':config['model_sha256'][name],
                         'best_episode':best_episode,'validation_coverage_auc':val,'code_fingerprint':code_fingerprint,
                         'environment_config_hash':signature,'wind_seed':wind,'wind_class':CLASSES.get(wind,'Other'),'eval_seed':seed,**metrics}
                    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps({'metrics':row,'trajectory':trajectory},allow_nan=False)+'\n');os.replace(tmp,dest)
                    print('completed',rid,flush=True)
                except Exception as e:
                    import traceback
                    with failures.open('a') as f:f.write(json.dumps({'run_id':rid,'error':str(e),'traceback':traceback.format_exc()})+'\n')
                    raise
    allrows=[json.loads(x.read_text())['metrics'] for x in sorted((output/'runs').glob('*.json')) if not x.name.startswith('._')]
    with (output/'raw_results.jsonl').open('w') as f:
        for row in allrows:f.write(json.dumps(row,allow_nan=False)+'\n')
    if allrows:
        with (output/'per_run_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in allrows)
    (output/'run_manifest.json').write_text(json.dumps({'expected_runs':len(models)*len(winds)*len(seeds),'completed_runs':len(allrows),'config':config},indent=2)+'\n')
if __name__=='__main__':main()
