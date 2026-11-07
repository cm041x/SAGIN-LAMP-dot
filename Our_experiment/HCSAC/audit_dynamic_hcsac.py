"""Independent audit; missing evidence or incomplete runs produce a failed audit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import ROOT,HCSAC,TrainingMode,initialize_agents,new_env
from Our_experiment.HCSAC.training.checkpointing import module_hash,file_hash
from Our_experiment.HCSAC.training.training_split import build_split

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--training-dir',required=True);p.add_argument('--evaluation-dir',required=True);a=p.parse_args()
    train=Path(a.training_dir);train=train if train.is_absolute() else ROOT/train
    ev=Path(a.evaluation_dir);ev=ev if ev.is_absolute() else ROOT/ev;ev.mkdir(parents=True,exist_ok=True)
    results={};errors=[]
    def check(name,condition,detail=''):
        results[name]={'passed':bool(condition),'detail':detail}
        if not condition:errors.append(name)
    paths=[(mode,seed,train/mode/f'seed_{seed}') for mode in ('scratch','finetune') for seed in (3001,3002,3003)]
    complete=[(m,s,p) for m,s,p in paths if (p/'run_manifest.json').exists() and json.loads((p/'run_manifest.json').read_text()).get('episodes')==1000]
    check('formal_training_runs',len(complete)==6,f'{len(complete)}/6')
    paired=True
    for seed in (3001,3002,3003):
        x=train/'scratch'/f'seed_{seed}'/'episode_manifest.csv';y=train/'finetune'/f'seed_{seed}'/'episode_manifest.csv'
        if not x.exists() or not y.exists() or x.read_bytes()!=y.read_bytes():paired=False
    check('shared_episode_manifests',paired)
    split=build_split();tr={x for v in split['train_wind_seed_pools'].values() for x in v};va={x['wind_seed'] for x in split['validation_combinations']};te={x['wind_seed'] for x in split['test_combinations']}
    check('disjoint_wind_splits',tr.isdisjoint(va|te) and va.isdisjoint(te))
    baseline={p:h for h,p in (line.split('  ',1) for line in (ROOT/'SOURCE_BASELINE_SHA256.txt').read_text().splitlines())}
    model_paths=['Our_experiment/HCSAC/data/sac_model_fly.pt','Our_experiment/HCSAC/data/sac_model_offload.pt']
    check('original_checkpoints_unchanged',all(file_hash(ROOT/p)==baseline[p] for p in model_paths))
    original=Path('/Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-HybridDRLGA-main')
    check('original_source_eight_sha256',all(file_hash(original/p)==h for p,h in baseline.items()))
    loaded=finite=fingerprints=masks=True; evidence_complete=True; validation_ok=True
    fly_original,off_original=initialize_agents(TrainingMode.FINETUNE,'cpu')
    original_actor=(module_hash(fly_original.actor),module_hash(off_original.actor))
    for mode,seed,folder in complete:
        try:
            ini=json.loads((folder/'initial_model_fingerprint.json').read_text());fin=json.loads((folder/'final_model_fingerprint.json').read_text())
            initial=(ini['fly_actor_sha256'],ini['offload_actor_sha256']);final=(fin['fly_actor_sha256'],fin['offload_actor_sha256'])
            fingerprints &= (initial==original_actor if mode=='finetune' else initial!=original_actor) and all(a!=b for a,b in zip(initial,final))
            x,y=initialize_agents(TrainingMode.SCRATCH,'cpu');x.actor.load_state_dict(torch.load(folder/'best_fly_actor.pt',map_location='cpu',weights_only=True),strict=True);y.actor.load_state_dict(torch.load(folder/'best_offload_actor.pt',map_location='cpu',weights_only=True),strict=True)
            env=new_env(2,{'wind_seed':11,'terrain_seed':42,'infra_seed':42})
            with torch.no_grad():loaded &= tuple(x.actor(torch.tensor(env._get_obs()[:1],dtype=torch.float32)).shape)==(1,8) and tuple(y._graph(y.actor,[env.get_obs_2()]).shape)==(1,6,5)
            with (folder/'validation_history.csv').open() as f:
                vals=list(csv.DictReader(f))
            if len(vals)!=20 or [int(v['episode']) for v in vals]!=list(range(50,1001,50)):
                validation_ok=False
            if vals:
                chosen=max(vals,key=lambda v:(float(v['coverage_auc']),float(v['final_coverage']),-float(v['repeated_visit_ratio']),-float(v['total_uav_energy_consumed_j'])))
                if int(chosen['episode'])!=json.loads((folder/'run_manifest.json').read_text())['best_episode']:validation_ok=False
            with (folder/'train_history.csv').open() as f:
                for row in csv.DictReader(f):
                    for key in ('fly_actor_loss','fly_critic_loss','off_actor_loss','off_critic_loss','fly_alpha','off_alpha'):
                        if row[key] and not np.isfinite(float(row[key])):finite=False
            ck=torch.load(folder/'last_checkpoint.pt',map_location='cpu',weights_only=False)
            for entry in ck['gcn_replay']['items']:
                graph,_,_,_,active,_,_,_,_,_=entry
                mask=graph.node_type==0;slots=graph.slot_id[mask]
                if sorted(slots.tolist())!=list(range(6)):
                    masks=False;break
                for i in range(6):
                    node=torch.nonzero((graph.node_type==0)&(graph.slot_id==i),as_tuple=False).flatten()
                    if len(node)!=1 or (bool(active[i]) and graph.x[int(node[0]),1]!=1):masks=False
                    if (not bool(entry[5][i]) or entry[7]) and not bool(entry[6][i]):masks=False
                if not masks:break
            graph_masks={(e[8],e[9]):e[4] for e in ck['gcn_replay']['items']}
            for entry in ck['cnn_replay']['items']:
                if len(entry)<9:
                    evidence_complete=False;break
                slot,episode,step,witness=entry[5],entry[6],entry[7],entry[8]
                if not witness.get('active_before') or slot not in witness.get('executed_slots',[]):
                    evidence_complete=False;break
                other=graph_masks.get((episode,step))
                if other is not None and not bool(other[slot]):evidence_complete=False;break
        except Exception as e:
            loaded=False;errors.append(f'{mode}/{seed}: {e}')
    check('actor_loading_shapes',len(complete)==6 and loaded)
    check('finetune_and_scratch_fingerprints',len(complete)==6 and fingerprints)
    check('finite_recorded_losses',len(complete)==6 and finite)
    check('validation_schedule_and_selection',len(complete)==6 and validation_ok)
    check('gcn_slot_and_active_masks',len(complete)==6 and masks)
    check('cnn_replay_independent_active_provenance',len(complete)==6 and evidence_complete,
          'CNN entries must record active-before and executed-slot witnesses, cross-checked against matching graph masks when retained.')
    raw=ev/'raw_results.jsonl';rows=[json.loads(x) for x in raw.read_text().splitlines()] if raw.exists() else []
    check('formal_evaluation_count',len(rows)==210,f'{len(rows)}/210')
    check('unique_run_ids',len({r['run_id'] for r in rows})==210)
    if rows:
        models={r['policy_name'] for r in rows};conditions={(w,s) for w in (11,23,4800) for s in range(42,52)}
        check('paired_test_conditions',len(models)==7 and all({(r['wind_seed'],r['eval_seed']) for r in rows if r['policy_name']==m}==conditions for m in models))
        check('inactive_offloading',all(r['inactive_uav_offload_count']==0 for r in rows))
        checkpoint_provenance=True
        for row in rows:
            if row['policy_name']=='original_zero_shot':
                expected=[file_hash(HCSAC/'data/sac_model_fly.pt'),file_hash(HCSAC/'data/sac_model_offload.pt')]
                if row['checkpoint_path']!='original checkpoints':checkpoint_provenance=False
            else:
                folder=train/row['training_mode']/f"seed_{row['training_seed']}"
                if not (folder/'run_manifest.json').exists():checkpoint_provenance=False;continue
                manifest=json.loads((folder/'run_manifest.json').read_text())
                expected=[file_hash(folder/'best_fly_actor.pt'),file_hash(folder/'best_offload_actor.pt')]
                if Path(row['checkpoint_path'])!=folder or row['best_episode']!=manifest['best_episode'] or row['validation_coverage_auc']!=manifest['best_validation'][0]:checkpoint_provenance=False
            if row['checkpoint_sha256']!=expected:checkpoint_provenance=False
        check('validation_selected_checkpoint_provenance',checkpoint_provenance)
        check('evaluation_environment_identity',len({r['environment_config_hash'] for r in rows})==1 and len({r['code_fingerprint'] for r in rows})==1)
        trajectories_valid=True; per_run_valid=True
        for row in rows:
            saved=ev/'runs'/(row['run_id']+'.json')
            if not saved.exists():per_run_valid=False;continue
            data=json.loads(saved.read_text())
            if data['metrics']!=row:per_run_valid=False
            if len(data['trajectory'])!=row['steps'] or any(not np.isfinite(v) or not 0<=v<=360000 for step in data['trajectory'] for v in step['energies']):trajectories_valid=False
        check('rollout_energy_and_trajectory',trajectories_valid)
        check('per_run_summary_matches_raw',per_run_valid)
        csvpath=ev/'per_run_metrics.csv'
        if csvpath.exists():
            with csvpath.open() as f:csvrows=list(csv.DictReader(f))
            csv_by_id={r['run_id']:r for r in csvrows};same=len(csvrows)==len(rows) and set(csv_by_id)=={r['run_id'] for r in rows}
            for row in rows:
                if row['run_id'] not in csv_by_id:continue
                cr=csv_by_id[row['run_id']]
                for key,value in row.items():
                    got=cr.get(key,'')
                    if value is None:ok=got==''
                    elif isinstance(value,list):ok=json.loads(got)==value
                    elif isinstance(value,(int,float)):ok=got!='' and np.isclose(float(got),value,rtol=1e-10,atol=1e-10)
                    else:ok=got==str(value)
                    if not ok:same=False
            check('raw_csv_values',same)
        else:check('raw_csv_values',False)
    else:
        for k in ('paired_test_conditions','inactive_offloading','validation_selected_checkpoint_provenance','evaluation_environment_identity','rollout_energy_and_trajectory','per_run_summary_matches_raw','raw_csv_values'):check(k,False,'No evaluation results')
    check('formal_policy_names', {r['policy_name'] for r in rows} == {'original_zero_shot'} | {f'{m}_seed_{s}' for m in ('scratch','finetune') for s in (3001,3002,3003)})
    report={'passed':not errors,'checks':results,'errors':errors}
    (ev/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (ev/'audit_report.md').write_text('# Dynamic HCSAC audit\n\n'+ '\n'.join(f"- {'PASS' if r['passed'] else 'FAIL'} {k}: {r['detail']}" for k,r in results.items())+'\n')
    print(json.dumps(report,indent=2));
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
