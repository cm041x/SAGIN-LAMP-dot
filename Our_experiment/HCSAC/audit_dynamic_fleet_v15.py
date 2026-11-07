"""Independent audit of saved V1.5 episodes and aggregated run lists."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def verify_directory(path: Path) -> dict:
    config=json.loads((path/'experiment_config.json').read_text()); manifest=json.loads((path/'run_manifest.json').read_text())
    raw=[json.loads(x) for x in (path/'raw_results.jsonl').read_text().splitlines()]
    with (path/'per_run_metrics.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    errors=[]; ids=set(); combinations_seen=set(); csv_by_id={r['run_id']:r for r in rows}
    def check(test: bool, msg: str) -> None:
        if not test: errors.append(msg)
    check(len(raw)==manifest['expected_runs']==manifest['completed_runs'], 'run count mismatch')
    check(len(rows)==len(raw)==len(csv_by_id),'CSV count/duplicate')
    for m in raw:
        rid=m['run_id']; run=path/'runs'/rid
        combo=(m['method'],m['target_search_uavs'],m['charging_capacity'],m['wind_seed'],m['eval_seed'])
        check(combo not in combinations_seen,f'{rid}: duplicate paired configuration')
        combinations_seen.add(combo)
        check(rid not in ids,f'{rid}: duplicate');ids.add(rid)
        try:
            saved=json.loads((run/'summary.json').read_text()); events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
            traj=json.loads((run/'trajectory.json').read_text()); fleet=json.loads((run/'fleet_timeline.json').read_text())
            off=json.loads((run/'offload_timeline.json').read_text()); hand=json.loads((run/'handover_timeline.json').read_text())
            visits=np.load(run/'visit_maps.npz')['visits']
        except Exception as e:
            errors.append(f'{rid}: read error {e}');continue
        check(saved==m,f'{rid}: raw/summary mismatch')
        row=csv_by_id.get(rid,{})
        for k,v in m.items():
            got=row.get(k,'')
            if v is None: same=got==''
            elif isinstance(v,(list,dict)): same=json.loads(got)==v
            elif isinstance(v,(int,float)): same=got!='' and math.isclose(float(got),v,rel_tol=1e-10,abs_tol=1e-10)
            else: same=got==str(v)
            check(same,f'{rid}: CSV {k}')
        check(m['experiment_type']==config['stage'] and m['code_fingerprint']==config['code_fingerprint'] and config['code_fingerprint'][:12] in rid,f'{rid}: provenance')
        check(m['wind_seed'] in config['wind_seeds'] and m['eval_seed'] in config['eval_seeds'],f'{rid}: seeds')
        expected_id=(f"{config['stage']}_{m['method']}_t{m['target_search_uavs']}_c{m['charging_capacity']}"
                     f"_p{m['charging_power_w']:g}_w{m['wind_seed']}_s{m['eval_seed']}_{config['code_fingerprint'][:12]}")
        check(rid==expected_id,f'{rid}: run_id fields')
        check(m['initial_positions']==config['initial_positions'],f'{rid}: initial deployment')
        check(m['terrain_seed']==m['infra_seed']==m['trajectory_seed']==m['eval_seed'],f'{rid}: pairing')
        check([m['method'],m['target_search_uavs'],m['charging_capacity']] in config['configs'],f'{rid}: config')
        check(m['charging_power_w']==config['charging_power_w'] and m['charging_efficiency']==config['charging_efficiency'],f'{rid}: power')
        check(len(traj)==len(fleet)==len(off)==len(hand)==m['steps'],f'{rid}: step lengths')
        check(m['inactive_uav_offload_count']==0 and m['charging_overload_steps']==0,f'{rid}: offload/charge')
        check(all(t['charging_count']<=m['charging_capacity'] for t in fleet),f'{rid}: charging capacity')
        check(all(0<=v<=m['initial_energy_j'] and math.isfinite(v) for t in traj for v in t['energies']),f'{rid}: energy')
        check(not any(e['event_type']=='energy_exhausted_before_return' for e in events),f'{rid}: exhaustion')
        check(visits.shape==(6,20,20) and int(visits.sum())==m['total_visit_count'],f'{rid}: visits')
        prev=m['initial_positions']
        for t in traj:
            for a,b in zip(prev,t['positions']):
                check(max(abs(a[0]-b[0]),abs(a[1]-b[1]))<=1,f'{rid}: teleport step {t["step"]}')
            prev=t['positions']
        binding={}; waypoint={}
        for e in events:
            typ=e['event_type']; i=e['uav_id']; meta=e['metadata']
            if typ=='replacement_assigned':
                replacement=meta['replacement_uav_id'];check(i not in binding and replacement not in binding.values(),f'{rid}: duplicate binding');binding[i]=replacement
            elif typ=='replacement_binding_released': binding.pop(i,None)
            elif typ=='handover_target_assigned':
                check(i not in waypoint and meta['owner'] not in waypoint.values(),f'{rid}: duplicate waypoint');waypoint[i]=meta['owner']
            elif typ=='handover_released':
                check(i in waypoint,f'{rid}: release without waypoint');waypoint.pop(i,None)
            elif typ=='handover_move':
                a=meta['from_position'];b=e['position'];check(max(abs(a[0]-b[0]),abs(a[1]-b[1]))==1,f'{rid}: illegal waypoint move')
        check(not waypoint,f'{rid}: open waypoint')
        if m['handover_mode']=='none': check(not any(e['event_type'].startswith('handover_') or e['event_type']=='responsibility_region_created' for e in events),f'{rid}: handover pollution')
        check(sum(x['active_decisions'] for x in off)==m['active_offload_decisions'],f'{rid}: offload timeline')
    expected={(c[0],c[1],c[2],w,seed) for c in config['configs'] for w in config['wind_seeds'] for seed in config['eval_seeds']}
    check(combinations_seen==expected,'missing or extra experiment configuration/seed combinations')
    failures=[json.loads(line) for line in (path/'failed_runs.jsonl').read_text().splitlines() if line.strip()]
    recovered={f['run_id'] for f in failures if f.get('run_id') in ids}
    manifest['failed_attempts']=len(failures)
    manifest['retried_run_count']=len(recovered)
    manifest['unresolved_failed_run_count']=len({f['run_id'] for f in failures if f.get('run_id') not in ids})
    manifest['retry_count_definition']='Distinct run IDs with recorded failed attempts and a later completed summary.'
    (path/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result={'expected_runs':manifest['expected_runs'],'audited_runs':len(raw),'errors':errors,'all_passed':not errors}
    (path/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (path/'audit_report.md').write_text('# V1.5 audit\n\n'+f"Audited {len(raw)}/{manifest['expected_runs']} runs; {'PASS' if not errors else 'FAIL'}.\n\n"+'\n'.join('- '+e for e in errors)+'\n')
    return result

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--capacity-dir',default='Our_experiment/HCSAC/data/dynamic_fleet_v15_capacity');p.add_argument('--handover-dir',default='Our_experiment/HCSAC/data/dynamic_fleet_v15_handover');a=p.parse_args()
    results=[]; paths=[]
    for name in (a.capacity_dir,a.handover_dir):
        path=Path(name);path=path if path.is_absolute() else ROOT/path; paths.append(path)
        r=verify_directory(path);results.append(r);print(path, len(r['errors']),'errors,',r['audited_runs'],'runs')
        for e in r['errors'][:10]:print(e)
    capacity=[json.loads(x) for x in (paths[0]/'raw_results.jsonl').read_text().splitlines()]
    handover=[json.loads(x) for x in (paths[1]/'raw_results.jsonl').read_text().splitlines()]
    aa={(m['wind_seed'],m['eval_seed']):m for m in capacity if m['target_search_uavs']==4 and m['charging_capacity']==1}
    bb={(m['wind_seed'],m['eval_seed']):m for m in handover if m['method']=='target_aware' and m['target_search_uavs']==4 and m['charging_capacity']==1}
    cross=[]
    if aa.keys()!=bb.keys(): cross.append('Cross-stage baseline seed sets differ')
    for key in aa.keys() & bb.keys():
        x={k:v for k,v in aa[key].items() if k not in ('run_id','experiment_type')}
        y={k:v for k,v in bb[key].items() if k not in ('run_id','experiment_type')}
        if x!=y: cross.append(f'{key}: cross-stage baseline summaries differ')
        for name in ('trajectory.json','fleet_timeline.json','offload_timeline.json','events.jsonl'):
            if (paths[0]/'runs'/aa[key]['run_id']/name).read_bytes() != (paths[1]/'runs'/bb[key]['run_id']/name).read_bytes():
                cross.append(f'{key}: cross-stage {name} differs')
    for path,r in zip(paths,results):
        r['cross_stage_baseline_pairs_checked']=len(aa.keys() & bb.keys())
        r['cross_stage_errors']=cross
        r['all_passed']=r['all_passed'] and not cross
        (path/'audit.json').write_text(json.dumps(r,indent=2)+'\n')
        with (path/'audit_report.md').open('a') as f:f.write(f"\nCross-stage matched baselines: {len(aa.keys() & bb.keys())}; errors: {len(cross)}.\n")
    print('Cross-stage matching baseline pairs:',len(aa.keys() & bb.keys()),'errors:',len(cross))
    if not all(r['all_passed'] for r in results):
        for e in cross[:10]:print(e)
        raise SystemExit(1)
if __name__=='__main__': main()
