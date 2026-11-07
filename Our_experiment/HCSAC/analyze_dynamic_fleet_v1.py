"""Analyze saved V1 CSVs; plots are reconstructed from persisted CSV data."""
import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Our_experiment.HCSAC.fleet_metrics import request_accounting

ROOT=Path(__file__).resolve().parents[2]
MODES=['no_replacement','naive_threshold','target_aware']
PAIRS=[('target_aware','naive_threshold'),('target_aware','no_replacement'),('naive_threshold','no_replacement')]
DIRECTIONS={'integrated_fleet_deficit':'lower','below_target_steps':'lower','integrated_fleet_surplus':'lower',
 'target_met_ratio':'higher','final_coverage':'higher','coverage_auc':'higher','mean_charging_queue_length':'lower',
 'mean_waiting_charge_steps':'lower','mean_replacement_delay_steps':'lower','return_flight_energy_j':'lower',
 'mean_target_recovery_delay_steps':'lower','total_uav_energy_consumed_j':'lower'}

def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as f: return list(csv.DictReader(f))

def number(x: str | None) -> float | None:
    if x is None or x=='': return None
    try:
        value=float(x)
        return value if np.isfinite(value) else None
    except ValueError: return None

def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(rows)

def grouped(rows: list[dict]) -> list[dict]:
    groups=defaultdict(list)
    for r in rows: groups[(r['scheduler_mode'],r['wind_class'],r['wind_seed'])].append(r)
    excluded={'wind_seed','eval_seed','trajectory_seed','terrain_seed','infra_seed'}
    keys=[k for k in rows[0] if k not in excluded and any(number(r.get(k)) is not None for r in rows)]
    out=[]
    for (mode,wind_class,wind_seed),group in sorted(groups.items()):
        record={'scheduler_mode':mode,'wind_class':wind_class,'wind_seed':wind_seed,'number_of_valid_runs':len(group)}
        for key in keys:
            values=[v for r in group if (v:=number(r.get(key))) is not None]
            for stat,val in [('mean',np.mean(values) if values else None),('std',np.std(values,ddof=1) if len(values)>1 else None),
                             ('median',np.median(values) if values else None),('min',min(values) if values else None),
                             ('max',max(values) if values else None),('number_of_valid_runs',len(values))]:
                record[f'{key}_{stat}']=float(val) if val is not None else None
        out.append(record)
    return out

def pair_comparisons(rows: list[dict]) -> list[dict]:
    index={(r['scheduler_mode'],r['wind_seed'],r['eval_seed']):r for r in rows}
    out=[]
    for first,second in PAIRS:
        for wind in sorted({r['wind_seed'] for r in rows},key=int):
            for metric,direction in DIRECTIONS.items():
                diffs=[]; wind_class=''
                for seed in sorted({r['eval_seed'] for r in rows if r['wind_seed']==wind},key=int):
                    a=index.get((first,wind,seed)); b=index.get((second,wind,seed))
                    if a is None or b is None: continue
                    x=number(a.get(metric)); y=number(b.get(metric)); wind_class=a['wind_class']
                    if x is not None and y is not None: diffs.append(x-y)
                if not diffs: continue
                better=np.array(diffs)*(1 if direction=='higher' else -1)
                out.append({'comparison':f'{first} vs {second}','wind_class':wind_class,'wind_seed':wind,'metric':metric,
                    'difference':'first minus second','better_direction':direction,'number_of_pairs':len(diffs),
                    'mean_paired_difference':float(np.mean(diffs)),'median_paired_difference':float(np.median(diffs)),
                    'wins':int(np.sum(better>1e-9)),'ties':int(np.sum(np.abs(better)<=1e-9)),'losses':int(np.sum(better < -1e-9))})
    return out

def plots(directory: Path) -> None:
    # Inputs are re-read from disk after aggregation; no rollout memory is used.
    rows=read_csv(directory/'per_run_metrics.csv'); agg=read_csv(directory/'aggregate_metrics.csv')
    trajectories=read_csv(directory/'trajectory.csv')
    winds=sorted({r['wind_seed'] for r in rows},key=int)
    labels={r['wind_seed']:r['wind_class'] for r in rows}; colors=['#6b7280','#e69f00','#0072b2']
    wind='4800' if '4800' in winds else winds[0]
    common=sorted(set.intersection(*[{r['eval_seed'] for r in rows if r['wind_seed']==wind and r['scheduler_mode']==m} for m in MODES if any(r['scheduler_mode']==m for r in rows)]),key=int)
    seed='42' if '42' in common else common[0]
    fig,ax=plt.subplots(figsize=(9,4))
    for mode,color in zip(MODES,colors):
        samples=[r for r in trajectories if r['wind_seed']==wind and r['eval_seed']==seed and r['scheduler_mode']==mode]
        if samples: ax.plot([int(r['step']) for r in samples],[int(r['search_active_count']) for r in samples],label=mode,color=color)
    target=int(float(rows[0]['target_search_uavs']))
    ax.axhline(target,ls='--',color='black',lw=.8,label=f'target {target}'); ax.set(xlabel='Step',ylabel='Search-active UAVs',title=f'{labels[wind]} / seed {seed}'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(directory/'fleet_count_example.png',dpi=150); plt.close(fig)
    def bars(filename: str,metric: str,ylabel: str):
        fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(winds)); width=.24
        for idx,(mode,color) in enumerate(zip(MODES,colors)):
            vals=[]
            for w in winds:
                r=next((r for r in agg if r['scheduler_mode']==mode and r['wind_seed']==w),None)
                vals.append(number(r.get(metric+'_mean')) if r else None)
            ax.bar(x+(idx-1)*width,[v if v is not None else np.nan for v in vals],width,label=mode,color=color)
        ax.set_xticks(x,[labels[w] for w in winds]); ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(directory/filename,dpi=150); plt.close(fig)
    bars('deficit_by_method.png','integrated_fleet_deficit','Integrated deficit (UAV-steps)')
    bars('surplus_by_method.png','integrated_fleet_surplus','Integrated surplus (UAV-steps)')
    bars('target_met_ratio.png','target_met_ratio','Fraction of steps at or above target')
    bars('coverage_by_method.png','final_coverage','Final coverage')
    bars('replacement_delay.png','mean_replacement_delay_steps','Mean observed replacement delay (steps)')
    fig,ax=plt.subplots(figsize=(7,5))
    for mode,color in zip(MODES,colors):
        rs=[r for r in rows if r['scheduler_mode']==mode]
        ax.scatter([float(r['integrated_fleet_deficit']) for r in rs],[float(r['final_coverage']) for r in rs],label=mode,color=color,alpha=.65)
    ax.set(xlabel='Integrated fleet deficit (UAV-steps)',ylabel='Final coverage'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(directory/'coverage_vs_deficit.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4)); x=np.arange(len(winds)); width=.24
    for idx,(mode,color) in enumerate(zip(MODES,colors)):
        subset=[next((r for r in agg if r['scheduler_mode']==mode and r['wind_seed']==w),None) for w in winds]
        for ax,metric in zip(axes,('mean_charging_queue_length_mean','mean_waiting_charge_steps_mean')):
            ax.bar(x+(idx-1)*width,[(number(r.get(metric)) if r and number(r.get(metric)) is not None else np.nan) for r in subset],width,label=mode,color=color)
    for ax,title in zip(axes,('Mean queue length','Mean observed wait (steps)')):
        ax.set_xticks(x,[labels[w] for w in winds]); ax.set_ylabel(title); ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(directory/'charging_queue_by_method.png',dpi=150); plt.close(fig)
    if all(number(r.get('total_uav_energy_consumed_j')) and number(r.get('final_coverage')) is not None for r in rows):
        fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(winds)); width=.24
        for idx,(mode,color) in enumerate(zip(MODES,colors)):
            vals=[]
            for w in winds:
                rs=[r for r in rows if r['scheduler_mode']==mode and r['wind_seed']==w]
                vals.append(np.mean([float(r['final_coverage'])/float(r['total_uav_energy_consumed_j']) for r in rs]) if rs else np.nan)
            ax.bar(x+(idx-1)*width,vals,width,label=mode,color=color)
        ax.set_xticks(x,[labels[w] for w in winds]); ax.set_ylabel('Coverage / UAV energy (1/J)'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(directory/'energy_efficiency.png',dpi=150); plt.close(fig)

def report(directory: Path, rows: list[dict], aggregate: list[dict], pairs: list[dict]) -> None:
    metrics=['integrated_fleet_deficit','integrated_fleet_surplus','below_target_steps','target_met_ratio','final_coverage','mean_charging_queue_length','mean_waiting_charge_steps','mean_replacement_delay_steps','return_flight_energy_j','charger_utilization_ratio']
    table=['| Wind | Mode | Deficit | Surplus | Below steps | Target met | Coverage | Queue | Wait | Replacement delay | Return energy J | Charger use |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v): return 'NA' if v is None else f'{v:.3f}'
    for r in aggregate:
        table.append('| '+r['wind_class']+' | '+r['scheduler_mode']+' | '+' | '.join(fmt(number(str(r.get(m+'_mean','')))) for m in metrics)+' |')
    pairlines=[]
    for r in pairs:
        if r['comparison']=='target_aware vs naive_threshold' and r['metric'] in ('integrated_fleet_deficit','integrated_fleet_surplus','target_met_ratio','final_coverage'):
            pairlines.append(f"- {r['wind_class']}, {r['metric']}: mean Δ {r['mean_paired_difference']:.3f}, median Δ {r['median_paired_difference']:.3f}; wins/ties/losses {r['wins']}/{r['ties']}/{r['losses']} (n={r['number_of_pairs']}).")
    def paired_summary(metric: str) -> str:
        subset=[r for r in pairs if r['comparison']=='target_aware vs naive_threshold' and r['metric']==metric]
        if not subset:
            return 'no paired observations'
        n=sum(r['number_of_pairs'] for r in subset)
        mean=sum(r['mean_paired_difference']*r['number_of_pairs'] for r in subset)/n
        wins=sum(r['wins'] for r in subset); ties=sum(r['ties'] for r in subset); losses=sum(r['losses'] for r in subset)
        return f'mean paired Δ {mean:.3f}; wins/ties/losses {wins}/{ties}/{losses} across {n} pairs'
    def group_mean(mode: str, metric: str) -> float | None:
        vals=[number(r.get(metric)) for r in rows if r['scheduler_mode']==mode]
        vals=[v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    def val(mode: str, metric: str) -> str:
        v=group_mean(mode,metric)
        return 'NA' if v is None else f'{v:.3f}'
    target_rows=[r for r in rows if r['scheduler_mode']=='target_aware']
    one_or_less=sum(number(r.get('min_search_active_count')) is not None and number(r['min_search_active_count'])<=1 for r in target_rows)
    def wind_value(label: str, metric: str) -> str:
        group=next((r for r in aggregate if r['scheduler_mode']=='target_aware' and r['wind_class']==label),None)
        if group is None: return 'NA'
        value=group.get(metric+'_mean')
        return 'NA' if value is None else f'{float(value):.3f}'
    assessments=[
        '1. **Deficit:** '+paired_summary('integrated_fleet_deficit')+'. The paired values are descriptive; no significance test was run.',
        '2. **Below target:** '+paired_summary('below_target_steps')+'.',
        '3. **Overdeployment:** '+paired_summary('integrated_fleet_surplus')+'. Surplus includes deliberate handover and is not by itself proof of an unnecessary launch.',
        '4. **Target availability:** '+paired_summary('target_met_ratio')+'.',
        '5. **Coverage:** '+paired_summary('final_coverage')+'. This comparison measures the simulated outcome, not a causal effect of fleet count alone.',
        '6. **Repeated search and contention:** mean repeated-visit ratio is '+val('target_aware','repeated_visit_ratio')+' for target-aware and '+val('naive_threshold','repeated_visit_ratio')+' for naive. Collision and contention effects were not separately measured; causation cannot be inferred.',
        '7. **Charging pressure:** mean charger utilization is '+val('target_aware','charger_utilization_ratio')+' and mean observed queue wait is '+val('target_aware','mean_waiting_charge_steps')+' steps for target-aware. A capacity ablation is needed before calling charging the main bottleneck.',
        '8. **Wind:** target-aware total return energy averages '+wind_value('Moderate Wind','return_flight_energy_j')+' J under moderate wind and '+wind_value('Strong Wind','return_flight_energy_j')+' J under strong wind; observed replacement delays are '+wind_value('Moderate Wind','mean_replacement_delay_steps')+' and '+wind_value('Strong Wind','mean_replacement_delay_steps')+' steps. The comparison is descriptive and includes changed trajectories and return counts.',
        '9. **Learning:** the paired diagnostics alone do not establish a need for or benefit from a learned controller.',
        '10. **Rule sufficiency:** whether the rule is sufficient depends on the target operating objective; define held-out conditions and stronger rule baselines before proposing V2.',
        '11. **Minimum fleet:** '+str(one_or_less)+'/'+str(len(target_rows))+' target-aware runs reached at most one search-active UAV at least once; the V0 minimum-count failure is not eliminated.'
    ]
    body='''# V1 diagnostic report

Deterministic pretrained actor rollouts use the nominal 600 W, 90% efficiency charging model, full initial batteries, identical paired seed configuration and no retraining. These are simulation diagnostics; 600 W is a nominal value, not verified hardware. Each mode uses the same initial deployments and random seeds. Differences in trajectory after a scheduling decision are expected. Target-met means at least four active aircraft. Coverage AUC is mean coverage over executed steps. Reported waits and delays use completed observations; unfinished intervals are right-censored and recorded in the CSV. Unresolved requests count both returns without a replacement and unassigned active requests; the rollout original active-pending count is retained separately. UAV local/communication energy uses the original model's feasibility/fallback rule; counts distinguish requested versus effective targets.

'''+ '\n'.join(table)+'''\n\n## Paired differences

Differences are target-aware minus naive, matched by wind seed and eval seed. Wins are defined by the better direction in `paired_comparisons.csv`; no statistical significance test is performed.

'''+ '\n'.join(pairlines)+'''\n\n## Interpretation boundaries

1. Compare deficit and below-target means and paired differences above; the data alone do not establish statistical significance.
2. Compare surplus and above-target steps in the CSV: overlap can be intentional handover, so surplus is not automatically an erroneous launch.
3. Coverage and deficit are reported together; more active UAVs can revisit cells or compete for SAGIN service. This diagnostic does not isolate the causal contribution of either mechanism.
4. Queue length, waiting time and charger utilization indicate charging pressure. A capacity-one bottleneck cannot be established causally without a capacity ablation.
5. Return energy and replacement delay by wind class are descriptive; only paired comparisons within a wind class use identical samples. Changes across wind classes do not establish causation.
6. Rule scheduling is a necessary baseline. These results alone do not establish that a learned scheduler is necessary or that it would outperform the rules. A learning-based V2 needs a defined objective, stronger baselines, held-out conditions and a controlled ablation.
7. `mean_replacement_delay_steps` excludes unassigned requests; recovery delays exclude ongoing gaps. Observe the corresponding counts and unresolved/censored fields before interpreting these means.
8. The offload energy split mirrors the original service feasibility checks and attributes local fallback to local computation; requested and effective target counts are separately retained. No physical hardware validation is claimed.
'''
    body += '\n## Answers from these diagnostics\n\n' + '\n\n'.join(assessments) + '\n'
    (directory/'V1_DIAGNOSTIC_REPORT.md').write_text(body)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',default='Our_experiment/HCSAC/data/dynamic_fleet_v1'); a=p.parse_args()
    directory=Path(a.input_dir); directory=directory if directory.is_absolute() else ROOT/directory
    # Preserve the rollout's original active-pending count, and derive unserved
    # requests from the complete event stream. Never discard the original data.
    if not (directory/'raw_results_original.jsonl').exists():
        shutil.copy2(directory/'raw_results.jsonl',directory/'raw_results_original.jsonl')
        shutil.copy2(directory/'per_run_metrics.csv',directory/'per_run_metrics_original.csv')
    rows=[]
    for path in sorted((directory/'runs').glob('*.json')):
        if path.name.startswith('._'): continue
        run=json.loads(path.read_text())
        row=dict(run['metrics'])
        row['rollout_original_unresolved_return_requests']=row.get('rollout_original_unresolved_return_requests',row['unresolved_return_requests'])
        row.update(request_accounting(run['events']))
        run['metrics']=row
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(run,allow_nan=False,ensure_ascii=False)+'\n')
        os.replace(tmp,path)
        rows.append(row)
    if not rows: raise SystemExit('No saved rollout metrics')
    with (directory/'raw_results.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
    save_csv(directory/'per_run_metrics.csv',
             [{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in rows],list(rows[0]))
    manifest_path=directory/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['metric_postprocessing']='unresolved requests count requests that returned without a binding plus still-active unassigned requests; original active-pending count preserved in *_original files and rollout_original_unresolved_return_requests'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    rows=read_csv(directory/'per_run_metrics.csv')
    aggregate=grouped(rows); pairs=pair_comparisons(rows)
    save_csv(directory/'aggregate_metrics.csv',aggregate,list(aggregate[0]))
    save_csv(directory/'paired_comparisons.csv',pairs,list(pairs[0]) if pairs else ['comparison','metric'])
    traj=[]
    for pth in sorted((directory/'runs').glob('*.json')):
        if pth.name.startswith('._'): continue
        run=json.loads(pth.read_text()); m=run['metrics']
        traj.extend({'scheduler_mode':m['scheduler_mode'],'wind_seed':m['wind_seed'],'eval_seed':m['eval_seed'],**item} for item in run['trajectory'])
    save_csv(directory/'trajectory.csv',traj,list(traj[0]))
    plots(directory); report(directory,rows,aggregate,pairs)
    first=rows[0]
    summary={'number_of_valid_runs':len(rows), 'charging_power_w':number(first.get('charging_power_w')),
             'charging_efficiency':number(first.get('charging_efficiency')),
             'step_duration_seconds':number(first.get('step_duration_seconds')),
             'charged_energy_per_step_j':number(first.get('charged_energy_per_step_j')),
             'total_grid_energy_input_j':sum(number(r.get('total_grid_energy_input_j')) or 0 for r in rows),
             'grid_energy_scope':'sum across all listed rollouts; per-run values in per_run_metrics.csv'}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'Analyzed {len(rows)} runs; {len(aggregate)} groups; {len(pairs)} paired rows in {directory}')

if __name__=='__main__': main()
