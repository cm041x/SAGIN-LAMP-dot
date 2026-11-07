"""Reproducible fixed-budget flight residual pilot; no import-time work."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/lamp_flight_residual_pilot'
OLD=ROOT/'outputs/lamp_prototype'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def old_output_check():
    snapshot=json.loads((OUT/'old_output_snapshot.json').read_text());errors=[]
    expected={x['path'] for x in snapshot['files']}
    current={str(p.relative_to(OLD)) for p in OLD.rglob('*') if p.is_file() and not any(x.startswith('._') for x in p.relative_to(OLD).parts)}
    if current!=expected:errors.append('Prior output file set changed')
    for row in snapshot['files']:
        p=OLD/row['path']
        if not p.is_file() or sha(p)!=row['sha256']:errors.append(row['path'])
    return {'passed':not errors,'checked_files':len(expected),'errors':errors}


def source_files():
    dirs=('Our_experiment/HCSAC/flight_residual','Our_experiment/HCSAC/lamp_method')
    paths=[]
    for name in dirs:paths.extend(sorted(p for p in (ROOT/name).glob('*.py') if not p.name.startswith('._')))
    paths += [ROOT/'Our_experiment/HCSAC/ENV'/n for n in ('UAVenv_SAC_Original.py','UAVenv_DynamicFleet.py',
              'UAVenv_DynamicFleetV1.py','UAVenv_DynamicFleetV15.py','fleet_scheduler.py','handover_policy.py')]
    paths += [ROOT/'Our_experiment/HCSAC'/n for n in ('wind.py','dynamic_fleet_v15_experiment.py','training/training_split.py')]
    return paths


def fingerprint():return hashlib.sha256(b''.join(p.read_bytes() for p in source_files())).hexdigest()


def run_tests():
    results=[]
    for pattern in ('test_dynamic*.py','test_lamp*.py','test_flight_residual.py'):
        cmd=[sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v']
        r=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        name='test_'+pattern.replace('*','all')+'.log';(OUT/name).write_text(r.stdout)
        print(r.stdout[-1200:],flush=True);results.append({'pattern':pattern,'returncode':r.returncode,'log':name})
    status={'passed':all(x['returncode']==0 for x in results),'suites':results}
    write_json(OUT/'test_results.json',status)
    if not status['passed']:raise RuntimeError('Regression or flight tests failed')


def initial_check():
    from .policy import VARIANTS,FlightMemory,flight_observe
    from .trainer import TeamSAC
    from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
    from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import new_env
    from Our_experiment.HCSAC.training.checkpointing import module_hash
    cnn=frozen_flight_actor();env=new_env(10,{'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000})
    snap=flight_observe(env,env._get_obs(),cnn);active=snap.action_mask;errors={}
    with torch.no_grad():
        base=cnn(torch.as_tensor(env._get_obs(),dtype=torch.float32)[active])
        for v in VARIANTS:
            torch.manual_seed(3001);actor=TeamSAC(v).actor;h=FlightMemory().advance(actor.encoder,snap)
            probs=actor.probabilities(h,snap);errors[v]=float((probs[active]-base).abs().max())
            if probs.shape!=(6,8):raise ValueError('Flight action dimension is not eight')
    result={'passed':all(v<1e-6 for v in errors.values()),'max_absolute_errors':errors,
            'cnn_file_sha256':sha(ROOT/'Our_experiment/HCSAC/data/sac_model_fly.pt'),
            'cnn_parameter_hash':module_hash(cnn),'tolerance':1e-6,'observation_schema':'unchanged V1.5 graph plus current lifecycle and frozen CNN FC1 features'}
    write_json(OUT/'initialization.json',result)
    if not result['passed']:raise RuntimeError('Zero residual does not reproduce CNN')


def require(name):
    p=OUT/name
    if not p.exists() or not json.loads(p.read_text()).get('passed'):raise RuntimeError(f'Gate not passed: {name}')


def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
def restore_rng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])


def pilot(resume):
    require('test_results.json');require('initialization.json');require('smoke_results.json')
    from .policy import VARIANTS
    from .trainer import TeamSAC
    from .collector import collect_episode
    from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
    from Our_experiment.HCSAC.lamp_method.sequence_replay import SequenceReplay
    from Our_experiment.HCSAC.training.training_split import build_split,episode_manifest
    from Our_experiment.HCSAC.training.checkpointing import module_hash
    split=build_split();manifest=episode_manifest(3001,20,split)
    if {m['wind_seed'] for m in manifest} & set(split['excluded_test_wind_seeds']):raise RuntimeError('Training/test wind overlap')
    if {m['wind_seed'] for m in manifest} & set(split['excluded_validation_wind_seeds']):raise RuntimeError('Training/validation wind overlap')
    cfg={'variants':list(VARIANTS),'train_seed':3001,'episodes':20,'max_steps':200,'warmup_episodes':2,
         'updates_per_episode':20,'unroll_length':16,'sequence_batch_size':4,'actor_lr':3e-4,'critic_lr':3e-4,
         'gamma':.99,'tau':.005,'temperature_mode':'fixed','alpha':.01,'fixed_offload':'HAPS',
         'decision_time':'pre_flight','task_feature_refresh':False,'training_fingerprint':fingerprint(),
         'cnn_sha256':sha(ROOT/'Our_experiment/HCSAC/data/sac_model_fly.pt'),
         'wind_sha256':sha(ROOT/'Our_experiment/HCSAC/wind.json'),
         'catalog_sha256':sha(ROOT/'Our_experiment/HCSAC/Our_experiment/HCSAC/data/wind_seed_classes_5000.json'),
         'ga_sha256':sha(ROOT/'Our_experiment/GA/data/ga_best_deployment_random10_w23_g10_i999999_pool0_200.json')}
    config_path=OUT/'experiment_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=cfg:raise ValueError('Configuration changed; preserve existing outputs')
    write_json(config_path,cfg)
    with (OUT/'episode_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    write_json(OUT/'validation_manifest.json',split['validation_combinations'])
    cnn=frozen_flight_actor();cnn_initial=module_hash(cnn)
    for variant in VARIANTS:
        folder=OUT/'training'/variant;folder.mkdir(parents=True,exist_ok=True);path=folder/'last_checkpoint.pt'
        random.seed(3001);np.random.seed(3001);torch.manual_seed(3001)
        agent=TeamSAC(variant);replay=SequenceReplay();history=[];start=0
        initial={'actor':module_hash(agent.actor),'q1':module_hash(agent.q1),'q2':module_hash(agent.q2)}
        if path.exists():
            if not resume:raise FileExistsError('Existing training output: use --resume')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if ck['config']!=cfg:raise ValueError('Checkpoint configuration changed')
            agent.load_state_dict(ck['agent']);replay.load_state_dict(ck['replay']);history=ck['history']
            start=ck['episode'];initial=ck['initial'];restore_rng(ck['rng'])
        for index in range(start,20):
            start_time=time.monotonic();episode=collect_episode(agent.actor,cnn,manifest[index],200,random_actions=index<2)
            replay.add_episode(episode.transitions);updates=[]
            if index>=2:
                for _ in range(20):updates.append(agent.update(replay.sample(16,4)))
            if module_hash(cnn)!=cnn_initial:raise RuntimeError('Frozen CNN parameters changed')
            row={'episode':index+1,'steps':episode.metrics['steps'],'metrics':episode.metrics,
                 'updates':updates,'seconds':time.monotonic()-start_time};history.append(row)
            write_json(folder/'episodes'/f'{index+1:03}.json',{'metrics':episode.metrics,'trajectory':episode.trajectory,'actions':episode.actions})
            state={'config':cfg,'episode':index+1,'agent':agent.state_dict(),'replay':replay.state_dict(),
                   'history':history,'rng':rng_state(),'initial':initial,'cnn_parameter_hash':cnn_initial}
            tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
            write_json(folder/'history.json',history)
            print('pilot',variant,'episode',index+1,'updates',agent.updates,'seconds',round(row['seconds'],1),flush=True)


def _evaluate_one(variant,agent,cnn,scenario,max_steps,dest,resume,identity):
    from .collector import collect_episode
    if resume and dest.exists():
        existing=json.loads(dest.read_text())
        if existing.get('identity')!=identity:raise ValueError('Existing evaluation has different model or configuration')
        return existing
    e=collect_episode(agent.actor if agent is not None else None,cnn,scenario,max_steps,greedy=True)
    event_metrics={'event_ce':None,'event_brier':None,'event_accuracy':None,'event_majority_accuracy':None,'event_counts':None}
    if variant=='lamp_lifecycle_ecc_flight_residual':
        from .policy import encode
        hidden1=hidden2=None;predictions=[];labels=[]
        with torch.no_grad():
            for t in e.transitions:
                snap=t['snapshot'];h1,hidden1=encode(agent.q1.encoder,snap,hidden1);h2,hidden2=encode(agent.q2.encoder,snap,hidden2)
                for q,h in ((agent.q1,h1),(agent.q2,h2)):
                    _,logits,_=q.head(h,t['actions'],snap.action_mask)
                    predictions.append(torch.softmax(logits,dim=-1));labels.append(t['event'])
        probs=torch.stack(predictions);lab=torch.tensor(labels);truth=torch.nn.functional.one_hot(lab,8)
        counts=[labels.count(i) for i in range(8)]
        event_metrics={'event_ce':float(-probs[torch.arange(len(lab)),lab].clamp_min(1e-8).log().mean()),
                       'event_brier':float((probs-truth).square().sum(-1).mean()),
                       'event_accuracy':float((probs.argmax(-1)==lab).float().mean()),
                       'event_majority_accuracy':max(counts)/len(labels),'event_counts':counts}
    e.metrics.update(event_metrics)
    result={'policy':variant,**scenario,'identity':identity,'metrics':e.metrics,'trajectory':e.trajectory,'actions':e.actions}
    write_json(dest,result);print('evaluated',variant,scenario['wind_seed'],scenario['terrain_seed'],flush=True)
    return result


def evaluate(resume):
    require('test_results.json');require('initialization.json')
    from .policy import VARIANTS
    from .trainer import TeamSAC
    from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
    from Our_experiment.HCSAC.training.training_split import build_split
    from Our_experiment.HCSAC.training.checkpointing import module_hash
    cnn=frozen_flight_actor();cnn_hash=module_hash(cnn);cases=[]
    for variant in ('frozen_cnn_reference',*VARIANTS):
        agent=None; checkpoint_hash=None
        if variant!='frozen_cnn_reference':
            checkpoint_path=OUT/'training'/variant/'last_checkpoint.pt';checkpoint_hash=sha(checkpoint_path)
            ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            if ck['episode']!=20 or ck['config']['training_fingerprint']!=fingerprint():raise RuntimeError('Missing or stale finished training')
            agent=TeamSAC(variant);agent.load_state_dict(ck['agent'])
        identity={'cnn_parameter_hash':cnn_hash,'checkpoint_sha256':checkpoint_hash,'code_fingerprint':fingerprint(),'max_steps':500}
        for scenario in build_split()['validation_combinations']:
            dest=OUT/'validation'/variant/f"w{scenario['wind_seed']}_t{scenario['terrain_seed']}.json"
            _evaluate_one(variant,agent,cnn,scenario,500,dest,resume,identity)
        for wind in (11,23,4800):
            for seed in (42,43,44):
                scenario={'wind_seed':wind,'terrain_seed':seed,'infra_seed':seed}
                dest=OUT/'evaluation'/variant/f'w{wind}_s{seed}.json'
                cases.append(_evaluate_one(variant,agent,cnn,scenario,500,dest,resume,identity))
    if module_hash(cnn)!=cnn_hash:raise RuntimeError('CNN weights changed during evaluation')
    reference={(r['wind_seed'],r['terrain_seed']):r for r in cases if r['policy']=='frozen_cnn_reference'}
    rows=[]
    for r in cases:
        ref=reference[(r['wind_seed'],r['terrain_seed'])]['trajectory'];trajectory=r['trajectory']
        compared=min(len(ref),len(trajectory))
        changed=sum(ref[i]['positions']!=trajectory[i]['positions'] for i in range(compared))
        row={'policy':r['policy'],'wind_seed':r['wind_seed'],'eval_seed':r['terrain_seed'],**r['metrics'],
             'trajectory_changed':bool(changed or len(ref)!=len(trajectory)),
             'trajectory_changed_steps':changed,'trajectory_compared_steps':compared}
        rows.append(row)
    keys=list(rows[0])
    with (OUT/'per_run_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    metrics=('coverage_auc','final_coverage','repeated_visit_ratio','total_uav_energy_j','return_energy_j',
             'integrated_fleet_deficit','action_change_ratio','safety_override_ratio','mean_residual_logits_norm')
    aggregate=[]
    for v in ('frozen_cnn_reference',*VARIANTS):
        subset=[r for r in rows if r['policy']==v];entry={'policy':v,'valid_runs':len(subset),
                 'trajectories_changed':sum(x['trajectory_changed'] for x in subset)}
        for name in metrics:
            values=[float(x[name]) for x in subset];entry[name+'_mean']=float(np.mean(values));entry[name+'_std']=float(np.std(values,ddof=1))
        aggregate.append(entry)
    with (OUT/'aggregate_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=aggregate[0]);w.writeheader();w.writerows(aggregate)
    write_json(OUT/'evaluation_manifest.json',{'completed':len(rows),'expected':45,'validation_per_policy':9})
    if len(rows)!=45:raise RuntimeError('Not all evaluation rollouts completed')


def report():
    require('test_results.json');require('initialization.json')
    from .policy import VARIANTS
    from .trainer import TeamSAC
    from Our_experiment.HCSAC.training.checkpointing import module_hash
    prior=old_output_check();write_json(OUT/'prior_output_integrity.json',prior)
    if not prior['passed']:raise RuntimeError('Prior offloading output changed')
    manifest=json.loads((OUT/'evaluation_manifest.json').read_text())
    if manifest['completed']!=45:raise RuntimeError('Evaluation incomplete')
    rows=list(csv.DictReader((OUT/'aggregate_metrics.csv').open()));lines=[
        '# Graph-assisted flight residual pilot','',
        'Short pilot only: 20 × 200 training steps versus 500-step evaluation; one training seed.', '',
        '| Policy | Coverage | Coverage AUC | Repeat | UAV energy J | Return energy J | Deficit | Changed actions | Changed trajectories |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['policy']} | {float(r['final_coverage_mean']):.4f} | {float(r['coverage_auc_mean']):.4f} | {float(r['repeated_visit_ratio_mean']):.4f} | {float(r['total_uav_energy_j_mean']):.0f} | {float(r['return_energy_j_mean']):.0f} | {float(r['integrated_fleet_deficit_mean']):.1f} | {float(r['action_change_ratio_mean']):.4f} | {r['trajectories_changed']}/9 |")
    lines+=['','## Training diagnostics','']
    audit={'passed':True,'prior_output':prior,'models':{},'evaluation_runs':45,'errors':[]}
    for variant in VARIANTS:
        folder=OUT/'training'/variant;h=json.loads((folder/'history.json').read_text())
        ck=torch.load(folder/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        torch.manual_seed(3001);initial=TeamSAC(variant)
        changes={k:any(not torch.equal(x,ck['agent'][k][name]) for name,x in getattr(initial,k).state_dict().items()) for k in ('actor','q1','q2')}
        residual_change=float((ck['agent']['actor']['residual_head.2.weight']-initial.actor.state_dict()['residual_head.2.weight']).norm())
        encoder_change=any(not torch.equal(x,ck['agent']['actor']['encoder.'+name]) for name,x in initial.actor.encoder.state_dict().items())
        updates=[u for e in h for u in e['updates']]
        def avg(name):
            values=[u[name] for u in updates if u[name] is not None]
            return float(np.mean(values)) if values else None
        event_counts=[sum(u['event_counts'][i] for u in updates) for i in range(8)]
        event_weight=sum(event_counts)
        def weighted(name):return sum(u[name]*sum(u['event_counts']) for u in updates if u[name] is not None)/event_weight if event_weight else None
        condition=[u['conditioner_gradient_norm'] for u in updates if u['conditioner_gradient_norm'] is not None]
        entry={'episode':ck['episode'],'environment_steps':sum(e['steps'] for e in h),'updates':ck['agent']['updates'],
               'parameter_changes':changes,'actor_encoder_changed':encoder_change,'residual_weight_delta_norm':residual_change,
               'mean_actor_encoder_gradient':avg('actor_encoder_gradient_norm'),
               'mean_residual_gradient':avg('residual_gradient_norm'),
               'mean_critic_encoder_gradients':[float(np.mean([u['critic_encoder_gradient_norms'][i] for u in updates])) for i in range(2)],
               'mean_actor_loss':avg('actor_loss'),'mean_critic_loss':avg('critic_loss'),
               'mean_conditioner_gradient':float(np.mean(condition)) if condition else None,
               'event_counts_both_critics':event_counts,'event_ce':weighted('event_ce'),
               'event_brier':weighted('event_brier'),'event_accuracy':weighted('event_accuracy'),
               'majority_accuracy':max(event_counts)/event_weight if event_weight else None}
        audit['models'][variant]=entry
        if ck['episode']!=20 or ck['agent']['updates']!=360 or entry['environment_steps']!=4000 or not all(changes.values()) or not encoder_change or residual_change<=0:
            audit['errors'].append('Incomplete or unchanged parameters: '+variant)
        if 'lifecycle' in variant and not (entry['mean_conditioner_gradient'] and entry['mean_conditioner_gradient']>0):
            audit['errors'].append('No lifecycle conditioner gradient: '+variant)
        lines.append(f"- {variant}: {entry}")
    evaluation_rows=list(csv.DictReader((OUT/'per_run_metrics.csv').open()))
    inactive=sum(int(x['inactive_uav_offload_count']) for x in evaluation_rows)
    exhausted=sum(int(x['energy_exhausted_events']) for x in evaluation_rows)
    audit.update(inactive_offloads=inactive,energy_exhausted_events=exhausted)
    if inactive:audit['errors'].append('Inactive UAV executed offload')
    lines+=['','## Evaluation action, safety and event details','']
    for policy in ('frozen_cnn_reference',*VARIANTS):
        group=[r for r in evaluation_rows if r['policy']==policy]
        requested=[sum(json.loads(r['requested_offload_counts'])[i] for r in group) for i in range(5)]
        effective=[sum(json.loads(r['effective_offload_counts'])[i] for r in group) for i in range(5)]
        proposals=sum(int(r['flight_proposals']) for r in group)
        changed=sum(int(r['changed_from_base']) for r in group)
        safety=sum(int(r['safety_overrides']) for r in group)
        if any(requested[i] for i in (0,1,3,4)):audit['errors'].append('Non-HAPS request: '+policy)
        lines.append(f'- {policy}: {changed}/{proposals} proposals differ from the frozen CNN on encountered states; '
                     f'{safety}/{proposals} overridden by safety boundary; '
                     f'requested 0..4={requested}, effective 0..4={effective}.')
        if policy=='lamp_lifecycle_ecc_flight_residual':
            ce=float(np.mean([float(r['event_ce']) for r in group]));brier=float(np.mean([float(r['event_brier']) for r in group]))
            accuracy=float(np.mean([float(r['event_accuracy']) for r in group]));majority=float(np.mean([float(r['event_majority_accuracy']) for r in group]))
            lines.append(f'  ECC independent evaluation: mean CE={ce:.5f}, Brier={brier:.5f}, accuracy={accuracy:.5f}, per-run majority reference={majority:.5f}; both critics counted.')
    config=json.loads((OUT/'experiment_config.json').read_text())
    if sha(ROOT/'Our_experiment/HCSAC/data/sac_model_fly.pt')!=config['cnn_sha256']:
        audit['errors'].append('Frozen CNN checkpoint file changed')
    audit['passed']=not audit['errors'];write_json(OUT/'audit.json',audit)
    lines+=['','CNN weights are frozen and hashed; action masks use pre-flight active slots. Flight proposals overridden by a return rule remain in replay.',
            'The fixed offload request is HAPS, with the original environment fallback. Requested and effective counts are in each run.',
            'Parameter change does not establish action change; action change does not establish performance gains.',
            'The inherited stale graph task feature and uncertainty-before-offload behavior remain unmodified.',
            'Previous offloading-only four variants chose HAPS in the evaluated states; this is not proof of global optimality.',
            'ECC scores are sampled training diagnostics, not independent calibration; no novelty or convergence claim.']
    (OUT/'FLIGHT_RESIDUAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    if not audit['passed']:raise RuntimeError('Flight residual audit failed: '+str(audit['errors']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('test','initial-check','pilot','evaluate','report'),required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    try:
        {'test':run_tests,'initial-check':initial_check,'pilot':lambda:pilot(a.resume),
         'evaluate':lambda:evaluate(a.resume),'report':report}[a.stage]()
    except Exception:
        with (OUT/'failures.jsonl').open('a') as f:f.write(json.dumps({'stage':a.stage,'traceback':traceback.format_exc()})+'\n')
        raise

if __name__=='__main__':main()
