"""Two-level evaluation analysis: checkpoint seeds and paired environment means."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import ROOT

METRICS=['final_coverage','coverage_auc','repeated_visit_ratio','coverage_gain_per_active_uav_step',
 'mean_pairwise_visit_overlap','integrated_fleet_deficit','target_met_ratio','total_uav_energy_consumed_j','target_contention_index']
LOWER={'repeated_visit_ratio','mean_pairwise_visit_overlap','integrated_fleet_deficit','total_uav_energy_consumed_j','target_contention_index'}

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def save(path,rows,fields=None):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def float_or_none(x):
    try:return float(x) if x not in (None,'') else None
    except ValueError:return None

def evaluate_tables(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['policy_name'],r['training_mode'],r['training_seed'],r['wind_class'])].append(r)
    checkpoints=[]
    for (policy,mode,seed,wind),rs in sorted(groups.items()):
        row={'policy_name':policy,'training_mode':mode,'training_seed':seed,'wind_class':wind,'valid_rollouts':len(rs)}
        for k in METRICS:
            vals=[float_or_none(r.get(k)) for r in rs];vals=[v for v in vals if v is not None]
            row[k+'_mean']=float(np.mean(vals)) if vals else None;row[k+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        checkpoints.append(row)
    environment=defaultdict(list)
    for r in rows:environment[(r['training_mode'],r['wind_seed'],r['eval_seed'])].append(r)
    mode_conditions=[]
    for (mode,wind,seed),rs in sorted(environment.items()):
        row={'training_mode':mode,'wind_seed':wind,'eval_seed':seed,'wind_class':rs[0]['wind_class'],'training_seed_count':len(rs)}
        for k in METRICS:
            vals=[float_or_none(r.get(k)) for r in rs];vals=[x for x in vals if x is not None];row[k]=float(np.mean(vals)) if vals else None
        mode_conditions.append(row)
    modegroups=defaultdict(list)
    for r in mode_conditions:modegroups[(r['training_mode'],r['wind_class'])].append(r)
    modes=[]
    for (mode,wind),rs in sorted(modegroups.items()):
        row={'training_mode':mode,'wind_class':wind,'environment_conditions':len(rs)}
        for k in METRICS:
            v=[r[k] for r in rs if r[k] is not None];row[k+'_mean']=float(np.mean(v)) if v else None;row[k+'_std']=float(np.std(v,ddof=1)) if len(v)>1 else None
        modes.append(row)
    index={(r['training_mode'],r['wind_seed'],r['eval_seed']):r for r in mode_conditions}
    pairs=[]
    for first,second in [('scratch','zero_shot'),('finetune','zero_shot'),('finetune','scratch')]:
        for metric in METRICS:
            d=[]
            for m,w,s in index:
                if m!=first or (second,w,s) not in index:continue
                x=index[(first,w,s)][metric];y=index[(second,w,s)][metric]
                if x is not None and y is not None:d.append(x-y)
            if d:
                good=np.array(d)*(-1 if metric in LOWER else 1)
                pairs.append({'comparison':first+' vs '+second,'metric':metric,'valid_pair_count':len(d),'mean_paired_difference':float(np.mean(d)),
                    'median_paired_difference':float(np.median(d)),'wins':int((good>1e-9).sum()),'ties':int((np.abs(good)<=1e-9).sum()),'losses':int((good < -1e-9).sum())})
    variability=[]
    for mode in ('scratch','finetune'):
        row={'training_mode':mode}
        for metric in METRICS:
            per_seed=[]
            for seed in sorted({r['training_seed'] for r in rows if r['training_mode']==mode}):
                values=[float_or_none(r.get(metric)) for r in rows if r['training_mode']==mode and r['training_seed']==seed];values=[v for v in values if v is not None]
                if values:per_seed.append(float(np.mean(values)))
            row[metric+'_seed_mean']=float(np.mean(per_seed)) if per_seed else None
            row[metric+'_seed_std']=float(np.std(per_seed,ddof=1)) if len(per_seed)>1 else None
        variability.append(row)
    return checkpoints,mode_conditions,modes,pairs,variability

def plots(out: Path):
    training=read(out/'aggregate_training.csv') if (out/'aggregate_training.csv').exists() else []
    val=read(out/'aggregate_validation.csv') if (out/'aggregate_validation.csv').exists() else []
    conditions=read(out/'mode_environment_means.csv');checkpoints=read(out/'evaluation_per_run.csv')
    def curve(filename,field,source,mode=None):
        fig,ax=plt.subplots(figsize=(8,4));drawn=False
        for key in sorted({(r['training_mode'],r['training_seed']) for r in source}):
            if mode and key[0]!=mode:continue
            rs=[r for r in source if (r['training_mode'],r['training_seed'])==key and float_or_none(r.get(field)) is not None]
            if rs:ax.plot([int(r['episode']) for r in rs],[float(r[field]) for r in rs],label=f'{key[0]} {key[1]}');drawn=True
        ax.set(xlabel='Episode',ylabel=field);ax.grid(alpha=.2)
        if drawn:ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/filename,dpi=140);plt.close(fig)
    for mode in ('scratch','finetune'):curve(mode+'_training_return.png','total_return',training,mode)
    for name,field in [('fly_actor_loss','fly_actor_loss'),('fly_critic_loss','fly_critic_loss'),('offload_actor_loss','off_actor_loss'),('offload_critic_loss','off_critic_loss')]:curve(name+'.png',field,training)
    fig,ax=plt.subplots(figsize=(8,4))
    for mode,seed in sorted({(r['training_mode'],r['training_seed']) for r in training}):
        rs=[r for r in training if r['training_mode']==mode and r['training_seed']==seed]
        for field,style in [('fly_alpha','-'),('off_alpha','--')]:
            ax.plot([int(r['episode']) for r in rs],[float(r[field]) for r in rs],style,label=f'{mode} {seed} {field}')
    ax.set(xlabel='Episode',ylabel='alpha');ax.legend(fontsize=6);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'alpha_curves.png',dpi=140);plt.close(fig)
    for name,field in [('validation_coverage_auc','coverage_auc'),('validation_final_coverage','final_coverage'),('validation_repeated_visit','repeated_visit_ratio')]:curve(name+'.png',field,val)
    for name,field in [('final_coverage_by_policy','final_coverage'),('coverage_auc_by_policy','coverage_auc'),('repeated_visit_by_policy','repeated_visit_ratio'),('coverage_gain_per_active_step','coverage_gain_per_active_uav_step'),('pairwise_overlap_by_policy','mean_pairwise_visit_overlap'),('fleet_deficit_by_policy','integrated_fleet_deficit'),('target_met_ratio_by_policy','target_met_ratio'),('energy_by_policy','total_uav_energy_consumed_j'),('contention_by_policy','target_contention_index')]:
        fig,ax=plt.subplots(figsize=(8,4));labels=['zero_shot','scratch','finetune'];v=[]
        for label in labels:
            x=[float(r[field]) for r in conditions if r['training_mode']==label and float_or_none(r.get(field)) is not None];v.append(float(np.mean(x)) if x else np.nan)
        ax.bar(labels,v);ax.set_ylabel(field);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=140);plt.close(fig)
    for filename,xkey in [('coverage_vs_energy','total_uav_energy_consumed_j'),('coverage_vs_repeated_visit','repeated_visit_ratio')]:
        fig,ax=plt.subplots(figsize=(6,4))
        for mode in ('zero_shot','scratch','finetune'):
            rs=[r for r in conditions if r['training_mode']==mode and float_or_none(r.get(xkey)) is not None]
            if rs:ax.scatter([float(r[xkey]) for r in rs],[float(r['final_coverage']) for r in rs],label=mode,alpha=.65)
        ax.set(xlabel=xkey,ylabel='final_coverage');ax.legend();fig.tight_layout();fig.savefig(out/(filename+'.png'),dpi=140);plt.close(fig)
    if (out/'example_trajectory.csv').exists():
        t=read(out/'example_trajectory.csv');modes=['zero_shot','scratch','finetune'];fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
        for ax,mode in zip(axes,modes):
            for i in range(6):
                rs=[r for r in t if r['training_mode']==mode and int(r['slot'])==i];ax.plot([float(r['x']) if r['status']=='active' else np.nan for r in rs],[float(r['y']) if r['status']=='active' else np.nan for r in rs],lw=.7)
            ax.set(title=mode,xlim=(-.5,19.5),ylim=(-.5,19.5));ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'zero_shot_scratch_finetune_example_trajectory.png',dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--training-dir',required=True);p.add_argument('--evaluation-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--scratch-dir');p.add_argument('--finetune-dir');a=p.parse_args()
    def path(s):
        q=Path(s);return q if q.is_absolute() else ROOT/q
    train=path(a.training_dir);ev=path(a.evaluation_dir);out=path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    training=[];validation=[];fingerprints=[]
    for mode in ('scratch','finetune'):
        base=path(a.scratch_dir) if mode=='scratch' and a.scratch_dir else path(a.finetune_dir) if mode=='finetune' and a.finetune_dir else train
        for folder in sorted((base/mode).glob('seed_*')):
            if not (folder/'train_history.csv').exists():continue
            seed=folder.name.split('_')[-1]
            training.extend({'training_mode':mode,'training_seed':seed,**r} for r in read(folder/'train_history.csv'))
            if (folder/'validation_history.csv').exists():validation.extend({'training_mode':mode,'training_seed':seed,**r} for r in read(folder/'validation_history.csv'))
            initial=json.loads((folder/'initial_model_fingerprint.json').read_text());final=json.loads((folder/'final_model_fingerprint.json').read_text())
            fingerprints.append({'training_mode':mode,'training_seed':seed,'initial_fly_sha256':initial['fly_actor_sha256'],'final_fly_sha256':final['fly_actor_sha256'],'initial_offload_sha256':initial['offload_actor_sha256'],'final_offload_sha256':final['offload_actor_sha256'],
                'initial_fly_critic_sha256':initial.get('fly_critic_1_sha256'),'final_fly_critic_sha256':final.get('fly_critic_1_sha256'),
                'initial_offload_critic_sha256':initial.get('offload_critic_1_sha256'),'final_offload_critic_sha256':final.get('offload_critic_1_sha256')})
    save(out/'aggregate_training.csv',training);save(out/'aggregate_validation.csv',validation);save(out/'model_fingerprints.csv',fingerprints)
    rows=read(ev/'per_run_metrics.csv');checkpoints,conditions,modes,pairs,variability=evaluate_tables(rows)
    save(out/'evaluation_per_run.csv',rows);save(out/'mode_environment_means.csv',conditions);save(ev/'aggregate_by_checkpoint.csv',checkpoints);save(ev/'aggregate_by_mode.csv',modes);save(ev/'paired_comparisons.csv',pairs)
    save(out/'aggregate_evaluation.csv',modes);save(out/'paired_comparisons.csv',pairs);save(out/'training_seed_variability.csv',variability)
    example=[]
    for mode in ('zero_shot','scratch','finetune'):
        matches=[r for r in rows if r['training_mode']==mode and r['wind_seed']=='4800' and r['eval_seed']=='42']
        if not matches:continue
        chosen=sorted(matches,key=lambda r:r['policy_name'])[0];trajectory=json.loads((ev/'runs'/(chosen['run_id']+'.json')).read_text())['trajectory']
        for t in trajectory:
            for slot in range(6):example.append({'training_mode':mode,'slot':slot,'step':t['step'],'x':t['positions'][slot][0],'y':t['positions'][slot][1],'status':t['statuses'][slot]})
    save(out/'example_trajectory.csv',example)
    plots(out)
    lines=['# Dynamic HCSAC diagnostic','',f'{len(rows)} evaluation rollouts; aggregation first averages training seeds within each environment condition.','', '| Mode | Wind | Coverage | AUC | Repeat | Deficit | Energy |','|---|---|---:|---:|---:|---:|---:|']
    for r in modes:lines.append(f"| {r['training_mode']} | {r['wind_class']} | {r['final_coverage_mean']:.4f} | {r['coverage_auc_mean']:.4f} | {r['repeated_visit_ratio_mean']:.4f} | {r['integrated_fleet_deficit_mean']:.2f} | {r['total_uav_energy_consumed_j_mean']:.0f} |")
    lines+=['','## Paired model comparisons','','Differences are mode means within each shared environment, first minus second. The mode mean uses available training seeds; formal acceptance requires three per mode and 30 conditions. No statistical significance is claimed.']
    for r in pairs:
        lines.append(f"- {r['comparison']} / {r['metric']}: mean Δ {r['mean_paired_difference']:.6f}, median Δ {r['median_paired_difference']:.6f}, wins/ties/losses {r['wins']}/{r['ties']}/{r['losses']} (n={r['valid_pair_count']}).")
    lines+=['','## Training and evidence limits','']
    for mode in ('scratch','finetune'):
        for seed in sorted({r['training_seed'] for r in validation if r['training_mode']==mode}):
            series=[r for r in validation if r['training_mode']==mode and r['training_seed']==seed]
            if series:
                first,last=series[0],series[-1]
                lines.append(f"- {mode} seed {seed}: validation AUC episode {first['episode']}={float(first['coverage_auc']):.4f}; episode {last['episode']}={float(last['coverage_auc']):.4f}. This change alone does not prove convergence.")
    lines+=['','Compare validation curves to assess learning speed, rather than selecting on evaluation. Training-seed means and between-seed standard deviations are in training_seed_variability.csv; actor/critic fingerprints are in model_fingerprints.csv.',
            'Coverage, AUC, repeats, spatial overlap, energy, target contention and fleet deficit are separately reported above and in the CSV. Their paired changes do not establish causal mechanisms.',
            'Repeated search by itself does not prove that global state is necessary; an explicit controlled observation ablation is needed. Fleet activation, return and charging remain controlled by the fixed rules, so an unchanged deficit cannot be attributed to a learned scheduling decision.',
            'The original HCSAC architecture can only be called a formal baseline after six completed runs, 210 matched evaluation episodes and a passed audit. These diagnostics alone cannot establish a need for a learned fleet scheduler, global state or handover; compare those proposals against the retrained baselines after validation.']
    (out/'DYNAMIC_HCSAC_DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n');print(f'Analyzed {len(rows)} rollouts')
if __name__=='__main__':main()
