"""Bounded stages for the LAMP prototype; import has no training side effects."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/lamp_prototype'
VARIANTS=('gcn_context','lamp_plain','lamp_lifecycle','lamp_lifecycle_ecc')
UPSTREAM='69f6eb53316ef19296fce4fd25e0a57cb128c231'
UPSTREAM_FILES=('src/models/lamp.py','src/layers/convolutions/lamp_conv.py',
 'src/layers/convolutions/simple_conv.py','src/layers/parameterizations/antisymmetric.py',
 'src/layers/parameterizations/symmetric.py','src/layers/parameterizations/dissipative_antisymmetric.py',
 'src/layers/attention/dissipative.py','src/layers/discretizations/euler.py','LICENSE','pyproject.toml')


def write_json(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def python_source_paths(directory: Path) -> list[Path]:
    """Exclude macOS AppleDouble sidecars from reproducibility fingerprints."""
    return sorted(p for p in directory.glob('*.py') if not p.name.startswith('._'))


def manifests():
    upstream=ROOT/'external/LAMP'
    files={p:sha(upstream/p) for p in UPSTREAM_FILES}
    write_json(OUT/'upstream_manifest.json',{'url':'https://github.com/FabriDeCastelli/LAMP','commit':UPSTREAM,'files':files})
    snapshot=json.loads((ROOT/'SOURCE_SNAPSHOT_SHA256.json').read_text())
    write_json(OUT/'source_snapshot_manifest.json',snapshot)
    for name in ('Our_experiment/HCSAC/data/sac_model_fly.pt','Our_experiment/HCSAC/data/sac_model_offload.pt',
                 'Our_experiment/GA/data/ga_best_deployment_random10_w23_g10_i999999_pool0_200.json'):
        expected={x['path']:x['sha256'] for x in snapshot['files']}[name]
        if sha(ROOT/name)!=expected:raise RuntimeError(f'Fixed source input changed: {name}')


def tests():
    manifests();results=[]
    for pattern in ('test_dynamic*.py','test_lamp*.py'):
        cmd=[sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v']
        result=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (OUT/f'{pattern.replace("*","all")}.log').write_text(result.stdout)
        results.append({'pattern':pattern,'returncode':result.returncode,'log':f'{pattern.replace("*","all")}.log'})
        print(result.stdout[-1200:],flush=True)
    ok=all(r['returncode']==0 for r in results)
    write_json(OUT/'test_results.json',{'passed':ok,'suites':results})
    if not ok:raise RuntimeError('Unit or regression tests failed')
    import importlib.util
    spec=importlib.util.spec_from_file_location('lamp_parity_test',ROOT/'tests/test_lamp_prototype.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    test=module.LampParity();test.setUp()
    actual,_=test.model.forward_snapshot(test.x,test.edges,test.weights,test.h,torch.arange(5))
    expected=module.dense_reference(test.model,test.x,test.edges,test.weights,test.h)
    error=float((actual-expected).abs().max())
    write_json(OUT/'parity_results.json',{'passed':error<2e-6,'method':'independent dense reconstruction of upstream conventions',
                'max_absolute_error':error,'tolerance':2e-6,'direct_official_execution':False,
                'reason':'torch_scatter is not installed; no dependency changes made'})
    if error>=2e-6:raise RuntimeError('Upstream-convention parity failed')


def require_gates(*names):
    for name in names:
        path=OUT/name
        if not path.exists() or not json.loads(path.read_text()).get('passed'):
            raise RuntimeError(f'Required gate has not passed: {name}')


def math_check():
    from .math_checks import run_checks
    checks=run_checks();write_json(OUT/'math_checks.json',checks)
    if not checks['passed']:raise RuntimeError('Mathematical checks failed')


def smoke():
    require_gates('test_results.json','parity_results.json')
    from .collector import collect_episode,frozen_flight_actor
    from .policy import Actor
    flight=frozen_flight_actor();data={}
    for variant in VARIANTS:
        torch.manual_seed(3001);t=time.monotonic()
        ep=collect_episode(Actor(variant),flight,{'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000},500,greedy=True)
        data[variant]={**ep.metrics,'seconds':time.monotonic()-t}
        print('smoke',variant,'steps',ep.metrics['steps'],'returns',ep.metrics['returns'],flush=True)
        if ep.metrics['steps']!=500 or ep.metrics['inactive_offload_count'] or ep.metrics['energy_exhausted']:
            raise RuntimeError('Smoke did not complete safely')
    write_json(OUT/'smoke_results.json',{'passed':True,'variants':data})


def snapshot_rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
def restore_rng(rng):random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])


def pilot(resume: bool):
    require_gates('test_results.json','parity_results.json','math_checks.json','smoke_results.json')
    from .collector import collect_episode,frozen_flight_actor
    from .trainer import TeamSAC
    from .sequence_replay import SequenceReplay
    from Our_experiment.HCSAC.training.training_split import build_split,episode_manifest
    split=build_split();manifest=episode_manifest(3001,20,split)
    assert not set(split['excluded_test_wind_seeds']) & {r['wind_seed'] for r in manifest}
    source_files=python_source_paths(ROOT/'Our_experiment/HCSAC/lamp_method')
    source_files += [ROOT/'Our_experiment/HCSAC/ENV'/name for name in ('UAVenv_SAC_Original.py',
        'UAVenv_DynamicFleet.py','UAVenv_DynamicFleetV1.py','UAVenv_DynamicFleetV15.py','fleet_scheduler.py')]
    code_fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest()
    config={'seed':3001,'episodes':20,'max_steps':200,'warmup_episodes':2,'updates_per_episode':20,
            'unroll_length':16,'sequence_batch_size':4,'learning_rate':3e-4,'gamma':.99,'tau':.005,
            'alpha':.01,'temperature_mode':'fixed','variants':VARIANTS,'code_fingerprint':code_fingerprint}
    config_path=OUT/'experiment_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=json.loads(json.dumps(config)):raise ValueError('Pilot configuration changed')
    write_json(config_path,config)
    with (OUT/'episode_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    flight=frozen_flight_actor()
    for variant in VARIANTS:
        folder=OUT/'training'/variant;folder.mkdir(parents=True,exist_ok=True);path=folder/'last_checkpoint.pt'
        random.seed(3001);np.random.seed(3001);torch.manual_seed(3001)
        agent=TeamSAC(variant); replay=SequenceReplay();history=[];start=0
        initial={k:sha_tensor(v) for k,v in agent.actor.state_dict().items()}
        if path.exists():
            if not resume:raise FileExistsError('Use --resume to retain existing pilot checkpoints')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if ck['config']!=json.loads(json.dumps(config)):raise ValueError('Pilot checkpoint config mismatch')
            agent.load_state_dict(ck['agent']);replay.load_state_dict(ck['replay']);history=ck['history'];start=ck['episode']
            initial=ck['initial'];restore_rng(ck['rng'])
        for index in range(start,len(manifest)):
            t=time.monotonic()
            ep=collect_episode(agent.actor,flight,manifest[index],200,random_actions=index<2)
            replay.add_episode(ep.transitions); updates=[]
            if index>=2:
                for _ in range(20):updates.append(agent.update(replay.sample(16,4)))
            row={'episode':index+1,'steps':ep.metrics['steps'],'total_physical_reward':ep.total_physical_reward,
                 'prefix_flight_reward':ep.prefix_flight_reward,'metrics':ep.metrics,'updates':updates,
                 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':time.monotonic()-t}
            history.append(row)
            checkpoint={'config':json.loads(json.dumps(config)),'episode':index+1,'agent':agent.state_dict(),
                        'replay':replay.state_dict(),'history':history,'rng':snapshot_rng(),'initial':initial}
            tmp=path.with_suffix('.tmp');torch.save(checkpoint,tmp);os.replace(tmp,path)
            write_json(folder/'history.json',history)
            print('pilot',variant,'episode',index+1,'updates',agent.updates,'seconds',round(row['seconds'],1),flush=True)


def sha_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def evaluate(resume: bool):
    require_gates('test_results.json','parity_results.json','math_checks.json','smoke_results.json')
    from .trainer import TeamSAC
    from .collector import collect_episode,frozen_flight_actor
    flight=frozen_flight_actor(); rows=[]
    for variant in VARIANTS:
        ck=torch.load(OUT/'training'/variant/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        if ck['episode']!=20:raise RuntimeError('Pilot must finish 20 episodes before final evaluation')
        agent=TeamSAC(variant);agent.load_state_dict(ck['agent'])
        for wind in (11,23,4800):
            for seed in (42,43,44):
                dest=OUT/'evaluation'/variant/f'w{wind}_s{seed}.json'
                if resume and dest.exists():row=json.loads(dest.read_text())
                else:
                    ep=collect_episode(agent.actor,flight,{'wind_seed':wind,'terrain_seed':seed,'infra_seed':seed},500,greedy=True)
                    row={'variant':variant,'wind_seed':wind,'eval_seed':seed,**ep.metrics}
                    write_json(dest,row);print('eval',variant,wind,seed,flush=True)
                rows.append(row)
    keys=list(rows[0]);path=OUT/'per_run_metrics.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in rows)
    names=('coverage_auc','final_coverage','repeated_visit_ratio','uav_energy_j','integrated_fleet_deficit','inactive_offload_count','energy_exhausted')
    aggregates=[]
    for v in VARIANTS:
        group=[r for r in rows if r['variant']==v]; entry={'variant':v,'valid_runs':len(group)}
        for key in names:
            x=[float(r[key]) for r in group];entry[key+'_mean']=float(np.mean(x));entry[key+'_std']=float(np.std(x,ddof=1))
        aggregates.append(entry)
    with (OUT/'aggregate_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=aggregates[0].keys());w.writeheader();w.writerows(aggregates)
    write_json(OUT/'evaluation_manifest.json',{'completed':len(rows),'expected':36,'variants':list(VARIANTS)})


def report():
    require_gates('test_results.json','parity_results.json','math_checks.json','smoke_results.json')
    manifest=json.loads((OUT/'evaluation_manifest.json').read_text())
    if manifest['completed']!=36:raise RuntimeError('Evaluation is incomplete')
    rows=list(csv.DictReader((OUT/'aggregate_metrics.csv').open()))
    lines=['# LAMP prototype report','', 'Short pilots only; no claim of superiority, novelty, stability or global path coordination.',
           'All 36 evaluation rollouts use fixed pretrained CNN actions and final pilot checkpoints. ', '',
           '| Variant | Coverage | Coverage AUC | Repeat ratio | UAV energy J | Deficit |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['variant']} | {float(r['final_coverage_mean']):.4f} | {float(r['coverage_auc_mean']):.4f} | {float(r['repeated_visit_ratio_mean']):.4f} | {float(r['uav_energy_j_mean']):.0f} | {float(r['integrated_fleet_deficit_mean']):.1f} |")
    lines+=['','## Training and event diagnostics','', 'Event metrics are sampled training-batch diagnostics, not calibration on independent data. Counts include both independent critics.']
    histories={}
    for variant in VARIANTS:
        folder=OUT/'training'/variant; history=json.loads((folder/'history.json').read_text());histories[variant]=history
        ck=torch.load(folder/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        counts=[sum(u['event_counts'][i] for e in history for u in e['updates']) for i in range(8)]
        upd=[u for e in history for u in e['updates']]
        ce=[u['event_ce'] for u in upd if u['event_ce'] is not None]; brier=[u['event_brier'] for u in upd if u['event_brier'] is not None]
        accuracy=[u['event_accuracy'] for u in upd if u['event_accuracy'] is not None]
        changed=sum(sha_tensor(v)!=ck['initial'][k] for k,v in ck['agent']['actor'].items())
        majority=max(counts)/sum(counts) if sum(counts) else None
        lines.append(f"- {variant}: episodes={len(history)}, env_steps={sum(e['steps'] for e in history)}, updates={len(upd)}, "
                     f"seconds={sum(e['seconds'] for e in history):.1f}, peak_rss_bytes={max(e['peak_rss_bytes'] for e in history)}, "
                     f"actor_tensors_changed={changed}, event_counts={counts}, event_CE={float(np.mean(ce)) if ce else None}, "
                     f"event_Brier={float(np.mean(brier)) if brier else None}, batch_accuracy={float(np.mean(accuracy)) if accuracy else None}, "
                     f"majority_class_reference={majority}.")
    lines+=['','## Boundaries', '', 'The dense parity check independently reconstructs the pinned upstream operations; it does not execute official torch_scatter.',
            'The default xi=1 has no claimed geometric stability guarantee. Validation-selected or formal learning outcomes are not established.',
            'The V1.5 environment updates uncertainty in flight before offload feasibility; this differs from the paper description and was not changed.',
            'The stopped Dynamic HCSAC run has a known temperature configuration concern and is not repaired by this work.']
    (OUT/'PROTOTYPE_REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for variant,h in histories.items():
        values=[np.mean([u['critic_loss'] for u in e['updates']]) if e['updates'] else np.nan for e in h]
        ax.plot(range(1,len(h)+1),values,label=variant)
    ax.set(xlabel='Episode',ylabel='Mean critic loss');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots()
    for variant,h in histories.items():ax.plot(range(1,len(h)+1),[e['metrics']['mean_hidden_norm'] for e in h],label=variant)
    ax.set(xlabel='Episode',ylabel='Mean hidden norm');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'hidden_norm.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots()
    for variant in ('lamp_plain','lamp_lifecycle','lamp_lifecycle_ecc'):
        h=histories[variant]; data=[x for e in h for x in e['metrics']['damping_by_status_samples']['active']]
        if data:ax.hist(data,bins=20,alpha=.4,label=variant)
    ax.set(xlabel='Active-node mean damping',ylabel='Observations');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'status_damping.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots();ax.bar([r['variant'] for r in rows],[float(r['final_coverage_mean']) for r in rows]);ax.tick_params(axis='x',rotation=25);ax.set_ylabel('Final coverage');fig.tight_layout();fig.savefig(OUT/'core_results.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('test','smoke','math-check','pilot','evaluate','report'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    try:
        {'test':tests,'smoke':smoke,'math-check':math_check,'pilot':lambda:pilot(args.resume),
         'evaluate':lambda:evaluate(args.resume),'report':report}[args.stage]()
    except Exception:
        with (OUT/'failures.jsonl').open('a') as f:f.write(json.dumps({'stage':args.stage,'traceback':traceback.format_exc()})+'\n')
        raise

if __name__=='__main__':main()
