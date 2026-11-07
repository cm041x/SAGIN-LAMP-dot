"""Regenerate V1.5 paired tables and plots solely from persisted artifacts."""
import argparse
from collections import defaultdict
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def number(x):
    if x is None or x=='':return None
    try:
        a=float(x);return a if np.isfinite(a) else None
    except (ValueError,TypeError):return None

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def write(path,rows,columns):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def config_name(row,stage):
    return f"t{row['target_search_uavs']}_c{row['charging_capacity']}" if stage=='capacity' else row['method']

def aggregate(rows,stage):
    groups=defaultdict(list)
    for r in rows:groups[(config_name(r,stage),r['wind_class'],r['wind_seed'])].append(r)
    excluded={'wind_seed','eval_seed','terrain_seed','infra_seed','trajectory_seed','target_search_uavs','charging_capacity'}
    unavailable={'same_cell_move_conflict_attempts','simultaneous_same_region_count','failed_or_dropped_task_count'}
    keys=[k for k in rows[0] if k not in excluded and (k in unavailable or any(number(r.get(k)) is not None for r in rows))]
    out=[]
    for (name,wind,seed),group in sorted(groups.items()):
        first=group[0]; row={'configuration':name,'wind_class':wind,'wind_seed':seed,'valid_run_count':len(group)}
        if stage=='capacity':row.update(target_search_uavs=first['target_search_uavs'],charging_capacity=first['charging_capacity'])
        else:row['method']=first['method']
        for key in keys:
            v=[z for r in group if (z:=number(r.get(key))) is not None]
            for suffix,x in [('mean',np.mean(v) if v else None),('std',np.std(v,ddof=1) if len(v)>1 else None),('median',np.median(v) if v else None),('min',min(v) if v else None),('max',max(v) if v else None),('valid_run_count',len(v))]:
                row[key+'_'+suffix]=float(x) if x is not None else None
        out.append(row)
    return out

DIRECTION={'integrated_fleet_deficit':'lower','integrated_fleet_surplus':'lower','runs_reaching_one_uav':'lower',
 'target_met_ratio':'higher','final_coverage':'higher','coverage_auc':'higher','repeated_visit_ratio':'lower',
 'coverage_gain_per_active_uav_step':'higher','mean_pairwise_visit_overlap':'lower','active_pair_proximity_steps':'lower',
 'same_cell_move_conflict_attempts':'lower','sequential_blocked_move_attempts':'lower','target_contention_index':'lower','mean_charging_queue_length':'lower',
 'mean_waiting_charge_steps':'lower','total_uav_energy_consumed_j':'lower','charger_utilization_ratio':'neutral'}

def pairs(rows,stage):
    comparisons=[('t4_c2','t4_c1'),('t3_c1','t4_c1'),('t3_c2','t3_c1'),('t3_c2','t4_c2')] if stage=='capacity' else [('target_aware_handover','target_aware'),('target_aware_handover','naive_threshold'),('target_aware','naive_threshold')]
    index={(config_name(r,stage),r['wind_seed'],r['eval_seed']):r for r in rows}; out=[]
    for a,b in comparisons:
        for wind in sorted({r['wind_seed'] for r in rows},key=int):
            for key,direction in DIRECTION.items():
                values=[]; label=''
                for seed in sorted({r['eval_seed'] for r in rows if r['wind_seed']==wind},key=int):
                    x=index.get((a,wind,seed));y=index.get((b,wind,seed))
                    if not x or not y:continue
                    v1=number(x.get(key));v2=number(y.get(key));label=x['wind_class']
                    if v1 is not None and v2 is not None:values.append(v1-v2)
                if not values:continue
                oriented=np.array(values)*(1 if direction=='higher' else -1)
                out.append({'comparison':a+' vs '+b,'wind_class':label,'wind_seed':wind,'metric':key,'better_direction':direction,
                    'difference':'first minus second','valid_pair_count':len(values),'mean_paired_difference':float(np.mean(values)),
                    'median_paired_difference':float(np.median(values)),
                    'wins':int((oriented>1e-9).sum()) if direction!='neutral' else None,
                    'ties':int((np.abs(oriented)<=1e-9).sum()),'losses':int((oriented < -1e-9).sum()) if direction!='neutral' else None})
    return out

def plot_bar(out,rows,stage,filename,metric,ylabel,*,as_sum=False):
    names=['t3_c1','t3_c2','t4_c1','t4_c2'] if stage=='capacity' else ['naive_threshold','target_aware','target_aware_handover']
    winds=sorted({r['wind_seed'] for r in rows},key=int);labels={r['wind_seed']:r['wind_class'] for r in rows}
    fig,ax=plt.subplots(figsize=(9,4.4));x=np.arange(len(winds));width=.8/len(names)
    for i,name in enumerate(names):
        vals=[]
        for w in winds:
            v=[number(r.get(metric)) for r in rows if config_name(r,stage)==name and r['wind_seed']==w]
            v=[z for z in v if z is not None]
            vals.append((sum(v) if as_sum else np.mean(v)) if v else np.nan)
        ax.bar(x+(i-(len(names)-1)/2)*width,vals,width,label=name)
    ax.set_xticks(x,[labels[w] for w in winds]);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/filename,dpi=140);plt.close(fig)

def plots(out,stage):
    rows=read(out/'per_run_metrics.csv')
    capacity=[('deficit_capacity_target.png','integrated_fleet_deficit','Integrated deficit (UAV-steps)'),('target_met_capacity_target.png','target_met_ratio','Fraction at or above target'),('charging_queue_capacity.png','mean_charging_queue_length','Mean charging queue length'),('charging_wait_capacity.png','mean_waiting_charge_steps','Mean observed charging wait (steps)'),('coverage_capacity_target.png','final_coverage','Final coverage'),('repeated_visit_capacity_target.png','repeated_visit_ratio','Repeat visit ratio'),('coverage_gain_per_active_step.png','coverage_gain_per_active_uav_step','New cells per active UAV-step')]
    handover=[('coverage_by_method.png','final_coverage','Final coverage'),('repeated_visit_by_method.png','repeated_visit_ratio','Repeat visit ratio'),('coverage_gain_per_active_step.png','coverage_gain_per_active_uav_step','New cells per active UAV-step'),('pairwise_overlap_by_method.png','mean_pairwise_visit_overlap','Mean pairwise Jaccard overlap'),('proximity_conflict_by_method.png','active_pair_proximity_steps','Nearby active-pair steps'),('contention_index_by_method.png','target_contention_index','Same-target contention proxy'),('handover_success_rate.png','handover_success_ratio','Fraction reaching waypoint'),('handover_travel_steps.png','mean_handover_travel_steps','Mean waypoint travel steps')]
    for fn,key,label in (capacity if stage=='capacity' else handover):plot_bar(out,rows,stage,fn,key,label)
    if stage=='capacity':
        plot_bar(out,rows,stage,'runs_reaching_one_uav.png','runs_reaching_one_uav','Runs reaching one active UAV',as_sum=True)
        names=['t3_c1','t3_c2','t4_c1','t4_c2']; data=[[float(r['min_search_active_count']) for r in rows if config_name(r,stage)==n] for n in names]
        fig,ax=plt.subplots(figsize=(7,4));ax.boxplot(data,labels=names,showmeans=True);ax.set(ylabel='Minimum search-active UAVs');fig.tight_layout();fig.savefig(out/'min_active_count_distribution.png',dpi=140);plt.close(fig)
    else:
        methods=['naive_threshold','target_aware','target_aware_handover'];wind='4800' if any(r['wind_seed']=='4800' for r in rows) else rows[0]['wind_seed'];seed='42' if any(r['eval_seed']=='42' and r['wind_seed']==wind for r in rows) else rows[0]['eval_seed']
        matches=[next((r for r in rows if r['method']==m and r['wind_seed']==wind and r['eval_seed']==seed),None) for m in methods]
        fig,ax=plt.subplots(figsize=(9,4))
        for m,r in zip(methods,matches):
            if r:
                t=json.loads((out/'runs'/r['run_id']/'fleet_timeline.json').read_text());ax.plot([x['step'] for x in t],[x['search_active_count'] for x in t],label=m)
        ax.axhline(4,ls='--',color='black',lw=.7);ax.set(xlabel='Step',ylabel='Search-active UAVs',title=f'wind {wind}, seed {seed}');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'fleet_count_example.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
        for ax,m,r in zip(axes,methods,matches):
            if r:
                t=json.loads((out/'runs'/r['run_id']/'trajectory.json').read_text())
                for u in range(6):
                    x=[s['positions'][u][0] if s['statuses'][u]=='active' else np.nan for s in t];y=[s['positions'][u][1] if s['statuses'][u]=='active' else np.nan for s in t];ax.plot(x,y,lw=.7,label=f'UAV {u}')
            ax.set(title=m,xlabel='x',xlim=(-.5,19.5),ylim=(-.5,19.5));ax.grid(alpha=.2)
        axes[0].set_ylabel('y');axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'trajectory_handover_example.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,5));r=matches[-1]
        if r:
            es=[json.loads(x) for x in (out/'runs'/r['run_id']/'events.jsonl').read_text().splitlines()]
            starts=[e for e in es if e['event_type']=='handover_target_assigned'];moves=[e for e in es if e['event_type']=='handover_move']
            for e in moves:
                a=e['metadata']['from_position'];b=e['position'];ax.plot([a[0],b[0]],[a[1],b[1]],color='#0072b2',alpha=.35)
            if starts:ax.scatter([e['metadata']['target'][0] for e in starts],[e['metadata']['target'][1] for e in starts],marker='*',s=90,color='red',label='Waypoints')
        ax.set(xlabel='x',ylabel='y',xlim=(-.5,19.5),ylim=(-.5,19.5),title=f'Handover wind {wind}, seed {seed}');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'handover_waypoint_example.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        for m in methods:
            rs=[r for r in rows if r['method']==m];ax.scatter([float(r['repeated_visit_ratio']) for r in rs],[float(r['final_coverage']) for r in rs],label=m,alpha=.65)
        ax.set(xlabel='Repeated visit ratio',ylabel='Final coverage');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'coverage_vs_repeated_visit.png',dpi=140);plt.close(fig)

def report(out,rows,stage,pairrows):
    names=['t3_c1','t3_c2','t4_c1','t4_c2'] if stage=='capacity' else ['naive_threshold','target_aware','target_aware_handover']
    keys=['integrated_fleet_deficit','integrated_fleet_surplus','target_met_ratio','final_coverage','repeated_visit_ratio','coverage_gain_per_active_uav_step','mean_pairwise_visit_overlap','active_pair_proximity_steps','target_contention_index','mean_charging_queue_length','mean_waiting_charge_steps','total_uav_energy_consumed_j']
    def avg(name,key):
        vals=[number(r.get(key)) for r in rows if config_name(r,stage)==name];vals=[x for x in vals if x is not None]
        return float(np.mean(vals)) if vals else None
    def fmt(x):return 'NA' if x is None else f'{x:.4f}'
    lines=['# V1.5 '+stage+' diagnostic','',f'{len(rows)} new V1.5 runs; deterministic pretrained actors, fixed existing GA deployment and 600 W / 0.9 efficiency. Differences are descriptive, not significance claims.','', '| Configuration | '+ ' | '.join(keys)+' |','|---|'+'---:|'*len(keys)]
    for n in names:lines.append('| '+n+' | '+' | '.join(fmt(avg(n,k)) for k in keys)+' |')
    lines+=['','## Paired results','', 'Differences are first minus second, paired within wind and eval seed. Full means, medians and wins/ties/losses are in paired_comparisons.csv.']
    for pair in sorted({r['comparison'] for r in pairrows}):
        lines.append('\n### '+pair)
        for metric in ('integrated_fleet_deficit','final_coverage','repeated_visit_ratio','coverage_gain_per_active_uav_step','mean_pairwise_visit_overlap','active_pair_proximity_steps','target_contention_index'):
            z=[r for r in pairrows if r['comparison']==pair and r['metric']==metric]
            if z:
                n=sum(r['valid_pair_count'] for r in z);mean=sum(r['mean_paired_difference']*r['valid_pair_count'] for r in z)/n
                lines.append(f"- {metric}: pooled mean Δ {mean:.4f}; wins/ties/losses {sum(r['wins'] for r in z)}/{sum(r['ties'] for r in z)}/{sum(r['losses'] for r in z)} across {n} pairs.")
    if stage=='capacity':
        lines+=['','## Interpretation','', 'Runs reaching exactly one search-active UAV by configuration:']
        for n in names:lines.append(f"- {n}: {sum(int(r['runs_reaching_one_uav']) for r in rows if config_name(r,stage)==n)}/{sum(config_name(r,stage)==n for r in rows)}.")
        lines+=['','Capacity and target changes are paired on seeds; compare deficit, waiting, utilization and coverage together. Target-relative deficit uses a different target in target=3 versus target=4, so its magnitude alone does not establish better service. Charger utilization/idle metrics are available in the CSV. Results do not prove that a real system should use two chargers or that one resource is the sole bottleneck.']
    else:
        hand=[r for r in rows if r['method']=='target_aware_handover']; assignments=sum(int(r['total_handover_assignments']) for r in hand);success=sum(int(r['successful_handover_targets']) for r in hand)
        lines+=['','## Handover and limitations','',f'Across {len(hand)} handover rollouts, {success}/{assignments} assigned waypoints were reached. Mean per-rollout waypoint flight energy: {fmt(avg("target_aware_handover","mean_handover_energy_j"))} J; travel steps: {fmt(avg("target_aware_handover","mean_handover_travel_steps"))}. Compare total flight energy with controls; waypoint energy alone is not incremental cost.',
                f"Runs reaching one active UAV: {sum(int(r['runs_reaching_one_uav']) for r in hand)}/{len(hand)}.",
                'The same-target measure includes local processing and is a contention proxy, not a measured network congestion rate. The requested simultaneous same-cell attempt count is unavailable: the rollout records sequential blocked moves but not all simultaneous attempted destinations; `sequential_blocked_move_attempts` is retained separately as a partial proxy. Pairwise Jaccard excludes pairs where neither UAV visited a cell. The original service model does not record dropped tasks, and no persistent multi-region ownership is defined: these two metrics are unavailable rather than assumed zero.',
                'Compare wind-specific paired rows before claiming consistent behavior across winds. Coverage, overlap and energy jointly describe the handover tradeoff; these diagnostics alone do not establish causal responsibility or justify a learned V2.']
    (out/('V15_CAPACITY_DIAGNOSTIC_REPORT.md' if stage=='capacity' else 'V15_HANDOVER_DIAGNOSTIC_REPORT.md')).write_text('\n'.join(lines)+'\n')

def main(stage):
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);a=p.parse_args();out=Path(a.input_dir);out=out if out.is_absolute() else ROOT/out
    # The simulator records blocked moves under sequential execution, not all
    # simultaneous same-destination attempts. Preserve it as a partial proxy.
    if not (out/'raw_results_original.jsonl').exists():
        shutil.copy2(out/'raw_results.jsonl',out/'raw_results_original.jsonl')
        shutil.copy2(out/'per_run_metrics.csv',out/'per_run_metrics_original.csv')
    normalized=[]
    for summary in sorted((out/'runs').glob('*/summary.json')):
        if any(part.startswith('._') for part in summary.parts):continue
        m=json.loads(summary.read_text())
        if 'sequential_blocked_move_attempts' not in m:
            m['sequential_blocked_move_attempts']=m['same_cell_move_conflict_attempts']
        m['same_cell_move_conflict_attempts']=None
        summary.write_text(json.dumps(m,allow_nan=False)+'\n')
        normalized.append(m)
    with (out/'raw_results.jsonl').open('w') as f:
        for m in normalized:f.write(json.dumps(m,allow_nan=False)+'\n')
    write(out/'per_run_metrics.csv',[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in m.items()} for m in normalized],list(normalized[0]))
    manifest_path=out/'run_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['metric_postprocessing']='Full simultaneous same-cell attempt count unavailable; original sequential blocked-move count preserved as sequential_blocked_move_attempts; original raw files retained.'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    rows=read(out/'per_run_metrics.csv'); groups=aggregate(rows,stage);pairrows=pairs(rows,stage)
    write(out/'aggregate_metrics.csv',groups,list(groups[0]));write(out/'paired_comparisons.csv',pairrows,list(pairrows[0]))
    plots(out,stage);report(out,rows,stage,pairrows);print('Analyzed',len(rows),'runs',len(groups),'groups',len(pairrows),'paired rows')
