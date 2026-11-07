"""Audit persisted V1 runs without modifying simulation results."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def audit_run(run: dict) -> list[str]:
    m=run['metrics']; events=run['events']; trajectory=run['trajectory']; errors=[]
    def check(condition: bool, message: str) -> None:
        if not condition: errors.append(message)
    check(len(trajectory)==m['steps'],'step count')
    check(m['duplicate_replacement_attempts']==0,'duplicate replacements')
    check(m['inactive_uav_offload_count']==0,'inactive offload')
    check(m['charging_overload_steps']==0,'charging capacity')
    check(all(e['energy'] is None or (math.isfinite(e['energy']) and 0<=e['energy']<=m['initial_energy_j']) for e in events),'event energy')
    check(not any(e['event_type']=='energy_exhausted_before_return' for e in events),'exhaustion before return')
    check(abs(m['local_compute_energy_j']+m['communication_energy_j']-m['offload_energy_j'])<1e-4,'offload energy split')
    check(abs(m['search_flight_energy_j']+m['return_flight_energy_j']+m['offload_energy_j']-m['total_uav_energy_consumed_j'])<1e-4,'total energy')
    counts=[t['search_active_count'] for t in trajectory]; target=m['target_search_uavs']
    check(sum(max(0,target-n) for n in counts)==m['integrated_fleet_deficit'],'deficit')
    check(sum(max(0,n-target) for n in counts)==m['integrated_fleet_surplus'],'surplus')
    check(m['total_launch_events']==sum(e['event_type']=='reserve_launched' for e in events),'launch count')
    check(m['active_offload_decisions']==sum(m['offload_target_counts'])==sum(m['effective_offload_target_counts']),'offload counts')
    check(m['local_processing_count']==m['effective_offload_target_counts'][0],'effective local count')
    check(m['terrain_seed']==m['eval_seed'] and m['infra_seed']==m['eval_seed'] and m['trajectory_seed']==m['eval_seed'],'shared seed mapping')
    if m['scheduler_mode']=='no_replacement': check(m['total_launch_events']==0,'no-replacement launch')
    if m['scheduler_mode']=='target_aware':
        deficits={e['step']:e['metadata']['launch_deficit'] for e in events if e['event_type']=='target_deficit_detected'}
        launches=Counter(e['step'] for e in events if e['event_type']=='reserve_launched')
        check(all(step in deficits and n<=deficits[step] for step,n in launches.items()),'launch exceeds projected deficit')
        check(all(e['metadata']['activation_reason'] in ('replace_pending_return','restore_target_count','post_charge_replenishment') for e in events if e['event_type']=='reserve_launched'),'launch reason')
    binding={}
    for e in events:
        if e['event_type']=='replacement_assigned':
            owner=e['uav_id']; replacement=e['metadata']['replacement_uav_id']
            check(owner not in binding and replacement not in binding.values(),'non-unique binding')
            binding[owner]=replacement
        elif e['event_type']=='replacement_binding_released': binding.pop(e['uav_id'],None)
    return errors

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',default='Our_experiment/HCSAC/data/dynamic_fleet_v1'); a=p.parse_args()
    directory=Path(a.input_dir); directory=directory if directory.is_absolute() else ROOT/directory
    problems={}; count=0
    for path in sorted((directory/'runs').glob('*.json')):
        if path.name.startswith('._'): continue
        count+=1; errors=audit_run(json.loads(path.read_text()))
        if errors: problems[path.stem]=errors
    manifest=json.loads((directory/'run_manifest.json').read_text())
    result={'audited_runs':count,'expected_runs':manifest['expected_runs'],'problems':problems,'all_passed':not problems and count==manifest['expected_runs']}
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['all_passed']: raise SystemExit(1)

if __name__=='__main__': main()
