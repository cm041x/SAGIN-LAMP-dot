"""Transparent per-rollout spatial and resource contention diagnostics."""
from itertools import combinations
from typing import Any
import numpy as np

def runs_reaching_one(counts: list[int]) -> int:
    return int(any(n==1 for n in counts))

def coverage_gain_per_active_step(newly_covered: int, active_steps: int) -> float:
    return newly_covered/active_steps if active_steps else 0.0

def pairwise_visit_overlap(visit_maps: np.ndarray) -> tuple[float | None,float | None]:
    masks=visit_maps>0; values=[]
    for i,j in combinations(range(len(masks)),2):
        union=np.logical_or(masks[i],masks[j]).sum()
        if union: values.append(float(np.logical_and(masks[i],masks[j]).sum()/union))
    return (float(np.mean(values)),float(max(values))) if values else (None,None)

def proximity_pairs(positions: list[tuple[int,int]]) -> int:
    return sum(max(abs(a[0]-b[0]),abs(a[1]-b[1]))<=1 for a,b in combinations(positions,2))

def contention_index(timeline: list[dict[str,Any]]) -> float:
    total=sum(t['active_decisions'] for t in timeline)
    return sum(sum(max(0,n-1) for n in t['requested']) for t in timeline)/max(1,total)

def summarize_v15(env, counts: list[int], initial_uncertainty: float) -> dict[str,Any]:
    newly=int(initial_uncertainty-env.uncertainty_matrix.sum()); visits=env.total_visit_count
    avg_overlap,max_overlap=pairwise_visit_overlap(env.visit_maps)
    times=env.offload_timeline
    peak=[max(x['requested']) if x['requested'] else 0 for x in times]
    completed=env.handover_records; successful=sum(h['reason']=='target_reached' for h in completed)
    event_types=[e['event_type'] for e in env.events]
    return {'runs_reaching_one_uav':runs_reaching_one(counts), 'newly_covered_cells':newly,
        'total_active_uav_steps':visits, 'coverage_gain_per_active_uav_step':coverage_gain_per_active_step(newly,visits),
        'total_visit_count':visits,'repeated_visit_count':visits-newly,
        'repeated_visit_ratio':(visits-newly)/visits if visits else 0.0,
        'unique_cells_visited':int((env.visit_maps.sum(axis=0)>0).sum()),
        'same_cell_move_conflict_attempts':env.conflict_attempts,'active_pair_proximity_steps':int(env.proximity_steps),
        'mean_pairwise_visit_overlap':avg_overlap,'max_pairwise_visit_overlap':max_overlap,
        'simultaneous_same_region_count':None,
        'active_offload_decisions':sum(t['active_decisions'] for t in times),
        'requested_offload_target_counts':[sum(t['requested'][i] for t in times) for i in range(5)],
        'effective_offload_target_counts':[sum(t['effective'][i] for t in times) for i in range(5)],
        'local_fallback_count':sum(t['local_fallback_count'] for t in times),
        'failed_or_dropped_task_count':None, 'inactive_uav_offload_count':0,
        'mean_same_target_concurrency':float(np.mean(peak)) if peak else 0.0,
        'max_same_target_concurrency':max(peak,default=0),'congestion_steps':sum(p>1 for p in peak),
        'target_contention_index':contention_index(times),
        'total_handover_assignments':event_types.count('handover_target_assigned'),
        'successful_handover_targets':successful,
        'handover_success_ratio':successful/len(completed) if completed else None,
        'mean_handover_travel_steps':float(np.mean([h['travel_steps'] for h in completed])) if completed else None,
        'max_handover_travel_steps':max((h['travel_steps'] for h in completed),default=None),
        'mean_handover_energy_j':float(np.mean([h['energy_j'] for h in completed])) if completed else None,
        'waypoint_timeout_count':sum(h['reason']=='timeout' for h in completed),
        'fallback_global_target_count':event_types.count('fallback_global_target'),
        'no_valid_handover_target_count':event_types.count('no_valid_handover_target'),
        'mean_target_uncertainty_reduction':float(np.mean([h['initial_uncertainty']-h['final_uncertainty'] for h in completed])) if completed else None,
        'post_handover_unique_coverage_gain':env.post_handover_unique_coverage_gain}
