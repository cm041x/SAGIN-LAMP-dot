"""Fixed-budget stability pilot; only writes the separate stability directory."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .policy import VARIANTS,encode
from .trainer import TeamSAC
from .collector import collect_episode
from .diagnose import OUT,ROOT,jwrite,fixed_q
from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
from Our_experiment.HCSAC.lamp_method.sequence_replay import SequenceReplay
from Our_experiment.HCSAC.training.training_split import build_split,episode_manifest
from Our_experiment.HCSAC.training.checkpointing import module_hash

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint():
    paths=sorted(p for p in (ROOT/'Our_experiment/HCSAC/flight_stability').glob('*.py') if not p.name.startswith('._'))
    paths+=sorted(p for p in (ROOT/'Our_experiment/HCSAC/flight_residual').glob('*.py') if not p.name.startswith('._'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()

def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
def restore_rng(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def tests():
    all_suites=[]
    for pat in ('test_dynamic*.py','test_lamp*.py','test_flight_residual.py','test_flight_stability.py'):
        command=[sys.executable,'-m','unittest','discover','-s','tests','-p',pat,'-v']
        p=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        name=pat.replace('*','all')+'.log';(OUT/name).write_text(p.stdout)
        print(p.stdout[-500:],flush=True);all_suites.append({'pattern':pat,'returncode':p.returncode,'log':name})
    jwrite(OUT/'test_results.json',{'passed':all(s['returncode']==0 for s in all_suites),'suites':all_suites})
    if any(s['returncode'] for s in all_suites):raise RuntimeError('Unit/regression tests failed')

def td_and_returns(agent,episode):
    """On-policy one-step SARSA residual and realized discounted return MSE."""
    hidden1=hidden2=None;q=[];ts=episode.transitions
    with torch.no_grad():
        for tr in ts:
            h1,hidden1=encode(agent.q1.encoder,tr['snapshot'],hidden1)
            h2,hidden2=encode(agent.q2.encoder,tr['snapshot'],hidden2)
            a=tr['actions'];mask=tr['snapshot'].action_mask
            q.append([float(agent.q1.head(h1,a,mask)[0]),float(agent.q2.head(h2,a,mask)[0])])
    q=np.asarray(q);rewards=np.asarray([x['reward'] for x in ts]);returns=np.zeros(len(ts));v=0.
    for i in range(len(ts)-1,-1,-1):v=rewards[i]+.99*v*(not ts[i]['terminal']);returns[i]=v
    next_q=np.vstack((q[1:],np.zeros((1,2))))
    targets=rewards[:,None]+.99*next_q*np.asarray([not x['terminal'] for x in ts])[:,None]
    return {'steps':len(ts),'one_step_sarsa_td_mse_per_critic':((q-targets)**2).mean(0).tolist(),
            'discounted_realized_return_mse_per_critic':((q-returns[:,None])**2).mean(0).tolist(),
            'mean_discounted_return':float(returns.mean())}

def setup(options):
    split=build_split();training=episode_manifest(3001,13,split)
    val=[split['validation_combinations'][i] for i in (0,3,6)]
    test=[{'wind_seed':w,'terrain_seed':42,'infra_seed':42} for w in (11,23,4800)]
    assert not ({x['wind_seed'] for x in training}&set(split['excluded_validation_wind_seeds']))
    assert not ({x['wind_seed'] for x in training}&set(split['excluded_test_wind_seeds']))
    cfg={'train_seed':3001,'episodes':10,'max_episode_steps':500,'updates_per_episode':20,
         'critic_warm_start_episodes':2,'critic_warm_start_updates':20,'held_out_training_episodes':1,
         'unroll_length':16,'sequence_batch_size':4,'actor_lr':.0003,'critic_lr':.0003,
         'gamma':.99,'tau':.005,'alpha':.01,'temperature_mode':'fixed',
         'residual_mode':options.residual_mode,'residual_bound':options.residual_bound,'input_layernorm':True,'readout_layernorm':True,
         'xi':options.xi,'kl_probe_steps':[1,6,11,16],'kl_limit':.01,'kl_max_attempts':3,'kl_retry_scale':.5,
         'collapse_rule':{'window_last_steps':100,'dominant_action_fraction_min':.9,'coverage_shortfall_min':.20,'minimum_scenarios':2},
         'validation_scenarios':val,'test_scenarios':test,'training_manifest':training[:10],
         'warm_scenarios':training[10:12],'heldout_training_scenario':training[12],
         'source_fingerprint':fingerprint(),'cnn_sha256':sha(ROOT/'Our_experiment/HCSAC/data/sac_model_fly.pt')}
    existing=OUT/'experiment_config.json'
    if existing.exists() and json.loads(existing.read_text())!=cfg:raise ValueError('Configuration or source changed; preserve prior run')
    jwrite(existing,cfg);return cfg

def cache_frozen(cnn,cfg):
    path=OUT/'frozen_training_cache.pt'
    if path.exists():
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['config']!=cfg:raise ValueError('frozen warmstart cache mismatch')
        return data['episodes']
    eps=[]
    for case in cfg['warm_scenarios']+[cfg['heldout_training_scenario']]:
        ep=collect_episode(None,cnn,case,500,greedy=True);eps.append(ep)
        print('frozen critic data',case['wind_seed'],ep.metrics['steps'],flush=True)
    save(path,{'config':cfg,'episodes':eps});return eps

def evaluate(actor,cnn,scenarios,agent=None):
    result=[]
    for s in scenarios:
        e=collect_episode(actor,cnn,s,500,greedy=True)
        saturations=[];spans=[];h1=h2=None
        if agent is not None:
            with torch.no_grad():
                for tr in e.transitions:
                    snap=tr['snapshot'];q1,h1=encode(agent.q1.encoder,snap,h1);q2,h2=encode(agent.q2.encoder,snap,h2)
                    action=torch.where(snap.action_mask,tr['actions'],torch.full_like(tr['actions'],8))
                    action=torch.cat((action,torch.full((4,),8,dtype=torch.long)))
                    h=torch.cat((q1,agent.q1.head.action_embedding(action)),dim=-1)
                    pre=agent.q1.head.node_fusion[0](h)
                    saturations.append(float((pre.abs()>3).float().mean()))
                    diagnostic=fixed_q(agent,snap,tr['actions'],q1,q2)
                    if diagnostic:spans.append(diagnostic['span'])
        e.metrics['mean_critic_tanh_saturation']=float(np.mean(saturations)) if saturations else None
        e.metrics['mean_critic_q_span']=float(np.mean(spans)) if spans else None
        result.append({'scenario':s,'metrics':e.metrics,'actions':e.actions,'trajectory':e.trajectory})
    return result

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def train(options):
    if not (OUT/'diagnostics/validation_diagnosis.json').exists():raise RuntimeError('Run read-only diagnosis first')
    if not json.loads((OUT/'test_results.json').read_text())['passed']:raise RuntimeError('Tests failed')
    if not json.loads((OUT/'preflight.json').read_text())['passed']:raise RuntimeError('Preflight failed')
    cfg=setup(options);cnn=frozen_flight_actor();cnn_hash=module_hash(cnn)
    frozen=cache_frozen(cnn,cfg)
    basepath=OUT/'validation/frozen_cnn_reference.json'
    if basepath.exists():base=json.loads(basepath.read_text())
    else:base=evaluate(None,cnn,cfg['validation_scenarios']);jwrite(basepath,base)
    probe={'snapshots':[t['snapshot'] for t in frozen[0].transitions[:16]],'indices':[0,5,10,15]}
    for variant in VARIANTS:
        folder=OUT/'training'/variant;folder.mkdir(parents=True,exist_ok=True);last=folder/'last_checkpoint.pt'
        random.seed(3001);np.random.seed(3001);torch.manual_seed(3001)
        agent=TeamSAC(variant,residual_bound=cfg['residual_bound'],bounded=cfg['residual_mode']=='bounded',xi=cfg['xi']);replay=SequenceReplay();history=[];start=0
        if last.exists():
            ck=torch.load(last,map_location='cpu',weights_only=False)
            if ck['config']!=cfg:raise ValueError('stale checkpoint; do not reuse optimizer')
            agent.load_state_dict(ck['agent']);replay.load_state_dict(ck['replay']);history=ck['history'];start=ck['episode'];restore_rng(ck['rng'])
            if ck.get('stopped_early'):print(variant,'already stopped for collapse',flush=True);continue
        else:
            save(folder/'initial_checkpoint.pt',{'config':cfg,'agent':agent.state_dict(),'episode':0})
            warm_before={'train':[td_and_returns(agent,e) for e in frozen[:2]],'heldout':td_and_returns(agent,frozen[2])}
            for e in frozen[:2]:replay.add_episode(e.transitions)
            w=[]
            for i in range(cfg['critic_warm_start_updates']):
                w.append(agent.update(replay.sample(16,4),actor_enabled=False))
                if (i+1)%5==0:print('critic warmstart',variant,i+1,flush=True)
            warm_after={'train':[td_and_returns(agent,e) for e in frozen[:2]],'heldout':td_and_returns(agent,frozen[2])}
            jwrite(folder/'warmstart_metrics.json',{'before':warm_before,'after':warm_after,'updates':w})
            save(folder/'warmstart_checkpoint.pt',{'config':cfg,'agent':agent.state_dict(),'replay':replay.state_dict(),'episode':0,'history':[],'rng':rng()})
            initial_val=evaluate(agent.actor,cnn,cfg['validation_scenarios'],agent);jwrite(OUT/'validation'/variant/'episode_000.json',initial_val)
        for index in range(start,10):
            began=time.monotonic();e=collect_episode(agent.actor,cnn,cfg['training_manifest'][index],500)
            replay.add_episode(e.transitions);updates=[]
            for i in range(20):
                updates.append(agent.update(replay.sample(16,4),probe=probe))
                if (i+1)%5==0:print('train',variant,'episode',index+1,'update',i+1,flush=True)
            if module_hash(cnn)!=cnn_hash:raise RuntimeError('CNN changed')
            row={'episode':index+1,'metrics':e.metrics,'updates':updates,'seconds':time.monotonic()-began};history.append(row)
            jwrite(folder/'episodes'/f'{index+1:03}.json',{'metrics':e.metrics,'actions':e.actions,'trajectory':e.trajectory})
            stopped=False
            if (index+1)%2==0:
                val=evaluate(agent.actor,cnn,cfg['validation_scenarios'],agent);jwrite(OUT/'validation'/variant/f'episode_{index+1:03}.json',val)
                failures=[]
                for actual,ref in zip(val,base):
                    m=actual['metrics'];counts=m['direction_counts_last100'];total=sum(counts)
                    dominant=max(counts)/max(1,total);shortfall=ref['metrics']['final_coverage']-m['final_coverage']
                    failures.append(dominant>=.9 and shortfall>=.20)
                stopped=sum(failures)>=2
                if stopped:jwrite(folder/'early_stop.json',{'episode':index+1,'reason':'direction_collapse_and_coverage_loss','scenario_flags':failures})
            state={'config':cfg,'agent':agent.state_dict(),'replay':replay.state_dict(),'history':history,'episode':index+1,'rng':rng(),'stopped_early':stopped,'cnn_hash':cnn_hash}
            save(last,state);jwrite(folder/'history.json',history)
            if (index+1)%2==0:save(folder/f'checkpoint_{index+1:03}.pt',state)
            print('completed',variant,index+1,'seconds',round(row['seconds'],1),'early_stop',stopped,flush=True)
            if stopped:break

def final_test_and_report(options):
    cfg=setup(options);cnn=frozen_flight_actor();rows=[]
    for name in ('frozen_cnn_reference',*VARIANTS):
        actor=None;agent=None
        if name!='frozen_cnn_reference':
            ck=torch.load(OUT/'training'/name/'last_checkpoint.pt',map_location='cpu',weights_only=False)
            if ck['config']!=cfg:raise ValueError('checkpoint configuration mismatch')
            torch.manual_seed(3001);agent=TeamSAC(name,residual_bound=cfg['residual_bound'],bounded=cfg['residual_mode']=='bounded',xi=cfg['xi']);agent.load_state_dict(ck['agent']);actor=agent.actor
        dest=OUT/'test'/f'{name}.json'
        if dest.exists():data=json.loads(dest.read_text())
        else:data=evaluate(actor,cnn,cfg['test_scenarios'],agent);jwrite(dest,data)
        for item in data:
            m=item['metrics'];rows.append({'policy':name,'wind_seed':item['scenario']['wind_seed'],
                **{k:m[k] for k in ('final_coverage','coverage_auc','total_team_reward','discounted_team_reward',
                    'action_change_ratio','mean_policy_kl_to_cnn','mean_residual_range','mean_raw_hidden_rms','mean_actor_tanh_saturation','mean_critic_tanh_saturation','mean_critic_q_span')},
                'direction_counts':json.dumps(m['direction_counts'])})
        print('test complete',name,flush=True)
    write_csv(OUT/'independent_test.csv',rows)
    valrows=[]
    for name in VARIANTS:
        for p in sorted((OUT/'validation'/name).glob('episode_*.json')):
            if p.name.startswith('._'):continue
            for x in json.loads(p.read_text()):
                m=x['metrics'];valrows.append({'policy':name,'episode':int(p.stem.split('_')[1]),'wind_seed':x['scenario']['wind_seed'],
                    **{k:m[k] for k in ('final_coverage','coverage_auc','total_team_reward','discounted_team_reward',
                         'action_change_ratio','mean_policy_kl_to_cnn','mean_residual_range','mean_raw_hidden_rms','mean_actor_tanh_saturation','mean_critic_tanh_saturation','mean_critic_q_span')},
                    'direction_counts':json.dumps(m['direction_counts'])})
    for x in json.loads((OUT/'validation/frozen_cnn_reference.json').read_text()):
        m=x['metrics'];valrows.append({'policy':'frozen_cnn_reference','episode':0,'wind_seed':x['scenario']['wind_seed'],
            **{k:m[k] for k in ('final_coverage','coverage_auc','total_team_reward','discounted_team_reward',
                 'action_change_ratio','mean_policy_kl_to_cnn','mean_residual_range','mean_raw_hidden_rms','mean_actor_tanh_saturation','mean_critic_tanh_saturation','mean_critic_q_span')},
            'direction_counts':json.dumps(m['direction_counts'])})
    write_csv(OUT/'validation_metrics.csv',valrows)
    snapshot=json.loads((OUT/'previous_outputs_sha256.json').read_text());old=ROOT/'outputs/lamp_flight_residual_pilot'
    current={str(p.relative_to(old)) for p in old.rglob('*') if p.is_file() and not any(q.startswith('._') for q in p.relative_to(old).parts)}
    expected={x['path'] for x in snapshot['files']};bad=[x['path'] for x in snapshot['files'] if not (old/x['path']).is_file() or sha(old/x['path'])!=x['sha256']]
    passed=current==expected and not bad and cfg['cnn_sha256']==sha(ROOT/'Our_experiment/HCSAC/data/sac_model_fly.pt')
    jwrite(OUT/'audit.json',{'passed':passed,'previous_file_count':len(expected),'previous_file_set_equal':current==expected,'changed_files':bad,'cnn_unchanged':cfg['cnn_sha256']==sha(ROOT/'Our_experiment/HCSAC/data/sac_model_fly.pt'),'test_rollouts':len(rows)})
    if not passed:raise RuntimeError('Prior data or CNN integrity failure')
    print('report and audit complete',flush=True)

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('test','train','evaluate'),required=True)
    p.add_argument('--residual-mode',choices=('bounded','unbounded'),default='bounded')
    p.add_argument('--residual-bound',type=float,default=1.);p.add_argument('--xi',type=float,default=2.)
    a=p.parse_args()
    {'test':lambda:tests(),'train':lambda:train(a),'evaluate':lambda:final_test_and_report(a)}[a.stage]()

if __name__=='__main__':main()
